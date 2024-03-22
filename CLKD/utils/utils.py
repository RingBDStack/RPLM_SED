import torch
import dgl
import numpy as np
import os
from utils.reader import SocialDataset,graph_statistics
from sklearn.cluster import KMeans
from sklearn import metrics

def generateMasks(length, data_split, train_i, i, validation_percent=0.1, test_percent=0.2, save_path=None):
    # verify total number of nodes
    assert length == data_split[i]
    if train_i == i:
        # randomly shuffle the graph indices
        train_indices = torch.randperm(length)
        # get total number of validation indices
        n_validation_samples = int(length * validation_percent)
        # sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        n_test_samples = n_validation_samples + int(length * test_percent)
        test_indices = train_indices[n_validation_samples:n_test_samples]
        train_indices = train_indices[n_test_samples:]

        if save_path is not None:
            torch.save(validation_indices, save_path + '/validation_indices.pt')
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(test_indices, save_path + '/test_indices.pt')
            validation_indices = torch.load(save_path + '/validation_indices.pt')
            train_indices = torch.load(save_path + '/train_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return train_indices, validation_indices, test_indices
    # If is in inference(prediction) epochs, generate test indices
    else:
        test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
        if save_path is not None:
            torch.save(test_indices, save_path + '/test_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return test_indices



def getdata(embedding_save_path, data_path, data_split, train_i, i, args, src=None, tgt=None):
    save_path_i = embedding_save_path + '/block_' + str(i)
    if not os.path.isdir(save_path_i):
        os.mkdir(save_path_i)
    # load data
    data = SocialDataset(data_path, i)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    in_feats = features.shape[1]  # feature dimension

    g = dgl.DGLGraph(data.matrix,
                     readonly=True)
    num_isolated_nodes = graph_statistics(g, save_path_i)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.readonly(readonly_state=True)
    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
    g = g.to(device)

    mask_path = save_path_i + '/masks'
    if not os.path.isdir(mask_path):
        os.mkdir(mask_path)

    if train_i == i:
        train_indices, validation_indices, test_indices = generateMasks(len(labels), data_split, train_i, i,
                                                                        args.validation_percent,
                                                                        args.test_percent,
                                                                        mask_path)
    else:
        test_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent,
                                     args.test_percent,
                                     mask_path)
    if args.use_cuda:
        features, labels = features.cuda(), labels.cuda()
        test_indices = test_indices.cuda()
        if train_i == i:
            train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()
    # features = F.normalize(features, p=2, dim=1)

    g.ndata['h'] = features
    if args.mode == 4:
        tranfeatures = np.load(
            data_path + '/' + str(i) + '/' + "-".join([src, tgt, 'features']) + '.npy')
        tranfeatures = torch.FloatTensor(tranfeatures)
        # tranfeatures = F.normalize(tranfeatures, p=2, dim=1)
        if args.use_cuda:
            tranfeatures = tranfeatures.cuda()
        g.ndata['tranfeatures'] = tranfeatures

    if train_i == i:
        return save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices
    else:
        return save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices


# Compute the representations of all the nodes in g using model
def extract_embeddings(g, model, num_all_samples, labels, args, device):
    with torch.no_grad():
        model.eval()
        select_indices = torch.LongTensor(range(0, num_all_samples)).to(device)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, select_indices, sampler,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
        )
        labels = labels.cpu().detach()
        fea_list = []
        nid_list = []
        label_list = []
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            blocks = [b.to(device) for b in blocks]
            extract_features = model(blocks,args)
            extract_features = extract_features.cpu().detach()
            extract_nids = blocks[-1].dstdata[dgl.NID].data.cpu()  # node ids
            extract_labels = labels[extract_nids]  # labels of all nodes
            fea_list.append(extract_features.numpy())
            nid_list.append(extract_nids.numpy())
            label_list.append(extract_labels.numpy())

        for b in blocks:
            del b
        del input_nodes, output_nodes
        del select_indices
        # assert batch_id == 0
        # extract_nids = extract_nids.data.cpu().numpy()
        # extract_features = extract_features.data.cpu().numpy()
        # extract_labels = extract_labels.data.cpu().numpy()
        extract_features = np.concatenate(fea_list, axis=0)
        extract_labels = np.concatenate(label_list, axis=0)
        extract_nids = np.concatenate(nid_list, axis=0)
        # generate train/test mask
        A = np.arange(num_all_samples)
        # print("A", A)
        # assert (A == extract_nids).all()

    return (extract_nids, extract_features, extract_labels)


def mutual_extract_embeddings(g, model, peer, src, tgt, num_all_samples, labels, args, device):
    with torch.no_grad():
        model.eval()
        peer.eval()
        select_indices = torch.LongTensor(range(0, num_all_samples)).to(device)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, select_indices, sampler,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
        )
        fea_list = []
        nid_list = []
        label_list = []
        labels = labels.cpu().detach()
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            blocks = [b.to(device) for b in blocks]
            extract_features1 = model(blocks,args)
            if (args.mode == 2 and args.add_mapping):
                print("** add linear tran peer feature **", src, tgt)
                extract_features2 = peer(blocks, args, True, src=src, tgt=tgt)  # representations of all nodes
            elif args.mode == 4:
                print("** add nonlinear tran peer feature **", src, tgt)
                extract_features2 = peer(blocks, args, True)  # representations of all nodes
            else:
                print("** add feature **")
                extract_features2 = peer(blocks, args)
            extract_nids = blocks[-1].dstdata[dgl.NID].cpu()
            extract_labels = labels[extract_nids]  # labels of all nodes
            extract_features1 = extract_features1.cpu().detach()
            extract_features2 = extract_features2.cpu().detach()
            extract_features = torch.cat((extract_features1, extract_features2), 1).numpy()
            # extract_features = extract_features1.numpy()
            fea_list.append(extract_features)
            nid_list.append(extract_nids.numpy())
            label_list.append(extract_labels.numpy())

        for b in blocks:
            del b
        del input_nodes, output_nodes, select_indices

        # assert batch_id == 0
        # extract_nids = extract_nids.data.cpu().numpy()
        # extract_features1 = extract_features1.data.cpu().detach()
        # extract_features2 = extract_features2.data.cpu().detach()
        # extract_features = torch.cat((extract_features1,extract_features2),1).numpy()
        # extract_labels = extract_labels.data.cpu().detach().numpy()
        extract_features = np.concatenate(fea_list, axis=0)
        extract_labels = np.concatenate(label_list, axis=0)
        extract_nids = np.concatenate(nid_list, axis=0)
        # generate train/test mask
        A = np.arange(num_all_samples)
        # print("A", A)
        # assert (A == extract_nids).all()

    return (extract_nids, extract_features, extract_labels)


def save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, path, counter):
    np.savetxt(path + '/features_' + str(counter) + '.tsv', extract_features, delimiter='\t')
    np.savetxt(path + '/labels_' + str(counter) + '.tsv', extract_labels, fmt='%i', delimiter='\t')
    with open(path + '/labels_tags_' + str(counter) + '.tsv', 'w') as f:
        f.write('label\tmessage_id\ttrain_tag\n')
        for (label, mid, train_tag) in zip(extract_labels, extract_nids, extract_train_tags):
            f.write("%s\t%s\t%s\n" % (label, mid, train_tag))
    print("Embeddings after inference epoch " + str(counter) + " saved.")


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def run_kmeans(extract_features, extract_labels, indices, metric, isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    ari = metrics.adjusted_rand_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic')
    print("nmi:", nmi, 'ami:', ami, 'ari:', ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if metric == 'ari':
        print('use ari')
        value = ari
    if metric == 'ami':
        print('use ami')
        value = ami
    # Return number  of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, value)


def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, metrics, is_validation=True,
             file_name='evaluate.txt'):
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, metrics)
    if is_validation:
        split = 'validation'
    else:
        split = 'test'
    message += '\tNumber of ' + split + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + split + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + split + ' '
    message += metrics+ ': '
    message += str(value)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, metrics,
                                                save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + split + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + split + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + split + f' {metrics}: '
        message += str(value)
    message += '\n'
    global NMI
    global AMI
    global ARI
    print("*********************************")
    with open(save_path + f'/{file_name}', 'a') as f:
        f.write(message)
        f.write('\n')
        f.write("NMI " + str(NMI) + " AMI " + str(AMI) + ' ARI ' + str(ARI))
    print(message)

    return value


