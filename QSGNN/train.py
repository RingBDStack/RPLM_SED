from load_data import getdata
from model import GAT
import torch.optim as optim
import time
import numpy as np
import torch
import os
import dgl
from sklearn import metrics
from sklearn.cluster import KMeans
from utils import pairwise_sample
from model import EDNN, simNN
from utils import edl_digamma_loss, relu_evidence, edl_mse_loss, edl_log_loss
import torch.nn.functional as F
from load_data import SocialDataset


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def run_kmeans(extract_features, extract_labels, indices, args,isoPath=None):
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
    print("nmi:",nmi,'ami:',ami,'ari:',ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if args.metrics =='ari':
        print('use ari')
        value = ari
    if args.metrics=='ami':
        print('use ami')
        value = ami
    # Return number  of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, value)


def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, args, is_validation=True):
    message = ''
    message += '\nEpoch '
    message += str(epoch+1)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, args)
    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode +' '
    message += args.metrics +': '
    message += str(value)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, value= run_kmeans(extract_features, extract_labels, indices, args,
                                              save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' value: '
        message += str(value)
    message += '\n'
    global NMI
    global AMI
    global ARI
    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI))
    print(message)

    all_value_save_path = "/".join(save_path.split('/')[0:-1])
    print(all_value_save_path)

    with open(all_value_save_path + '/evaluate.txt', 'a') as f:
        f.write("block "+ save_path.split('/')[-1])
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI) + '\n')

    return value


def extract_embeddings(g, model, num_all_samples, args):
    with torch.no_grad():
        model.eval()
        indices = torch.LongTensor(np.arange(0,num_all_samples,1))
        if args.use_cuda:
            indices = indices.cuda()
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, graph_sampler=sampler,
            batch_size=num_all_samples,
            indices = indices,
            shuffle=False,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            extract_labels = blocks[-1].dstdata['labels']
            extract_features = model(blocks)

        assert batch_id == 0
        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()

    return (extract_features, extract_labels)

def initial_train(i, args, data_split, metrics,embedding_save_path, loss_fn, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices = getdata(
        embedding_save_path, data_split, i, args)

    if model is None:  # Construct the initial model
        model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    if args.use_cuda:
        model.cuda()

    # Optimizer
    optimizer = optim.Adam([{"params":model.parameters()},lr=args.lr, weight_decay=1e-4)

    # Start training
    message = "\n------------ Start initial training ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # record the highest validation nmi ever got for early stopping
    best_vali_value = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_value = []
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        label_center = {}
        for l in set(extract_labels):
            l_indices = np.where(extract_labels==l)[0]
            l_feas = extract_features[l_indices]
            l_cen = np.mean(l_feas,0)
            label_center[l] = l_cen


        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )


        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']

            start_batch = time.time()
            model.train()
            # forward
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).


            # loss_outputs = loss_fn(pred, batch_labels)
            # loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            dis = torch.empty([0, 1]).cuda()
            for l in set(batch_labels.cpu().data.numpy()):
                label_indices = torch.where(batch_labels==l)
                l_center = torch.FloatTensor(label_center[l]).cuda()
                dis_l = (pred[label_indices] - l_center).pow(2).sum(1).unsqueeze(-1)
                dis = torch.cat([dis,dis_l],0)

            if args.add_pair:
                pairs, pair_labels, pair_matrix = pairwise_sample(pred, batch_labels)
                if args.use_cuda:
                    pairs = pairs.cuda()
                    pair_matrix = pair_matrix.cuda()
                    # pair_labels = pair_labels.unsqueeze(-1).cuda()

                pos_indices = torch.where(pair_labels > 0)
                neg_indices = torch.where(pair_labels == 0)
                neg_ind = torch.randint(0, neg_indices[0].shape[0], [5*pos_indices[0].shape[0]]).cuda()
                neg_dis = (pred[pairs[neg_indices[0][neg_ind], 0]] - pred[pairs[neg_indices[0][neg_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                pos_dis = (pred[pairs[pos_indices[0], 0]] - pred[pairs[pos_indices[0], 1]]).pow(2).sum(1).unsqueeze(-1)
                pos_dis = torch.cat([pos_dis]*5,0)
                pairs_indices = torch.where(torch.clamp(pos_dis + args.a - neg_dis, min=0.0)>0)
                loss = torch.mean(torch.clamp(pos_dis + args.a - neg_dis, min=0.0)[pairs_indices[0]]) 

                label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda()
                pred = F.normalize(pred, 2, 1)
                pair_out = torch.mm(pred,pred.t())
                if args.add_ort:
                    pair_loss = (pair_matrix - pair_out).pow(2).mean()
                    print("pair loss:",loss,"pair orthogonal loss:  ",100*pair_loss)
                    loss += 100 * pair_loss

            losses.append(loss.item())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        np.save(save_path_i + '/features_' + str(epoch) + '.npy', extract_features)
        np.save(save_path_i + '/labels_' + str(epoch) + '.npy', extract_labels)

        validation_value = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                  save_path_i, args, True)
        all_vali_value.append(validation_value)

        # Early stop
        if validation_value > best_vali_value:
            best_vali_value = validation_value
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)

        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # Save all validation nmi
    np.save(save_path_i + '/all_vali_value.npy', np.asarray(all_vali_value))
    # Save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')
    # Load the best model of the current block
    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded.")


    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    label_center = {}
    for l in set(extract_labels):
        l_indices = np.where(extract_labels == l)[0]
        l_feas = extract_features[l_indices]
        l_cen = np.mean(l_feas, 0)
        label_center[l] = l_cen
    label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda()
    torch.save(label_center_emb,save_path_i + '/models/center.pth')

    if args.add_pair:
        return model, label_center_emb
    else:
        return model


def continue_train(i, data_split, metrics, embedding_save_path, loss_fn, model, label_center_emb,args):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(
        embedding_save_path, data_split, i, args)

    if i%1!=0:
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        test_value = evaluate(extract_features, extract_labels, test_indices, 0, num_isolated_nodes,
                              save_path_i, args, True)
        return model


    else:
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        test_value = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                              save_path_i, args, True)
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Start fine tuning
        message = "\n------------ Start fine tuning ------------\n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)

        # record the time spent in seconds on each batch of all training/maintaining epochs
        seconds_train_batches = []
        # record the time spent in mins on each epoch
        mins_train_epochs = []
        for epoch in range(args.finetune_epochs):
            start_epoch = time.time()
            losses = []
            total_loss = 0
            for metric in metrics:
                metric.reset()

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, test_indices, sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                )

            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
                blocks = [b.to(device) for b in blocks]
                batch_labels = blocks[-1].dstdata['labels']

                start_batch = time.time()
                model.train()
                label_center_emb.to(device)

                # forward
                pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
                pred = F.normalize(pred, 2, 1)
                rela_center_vec = torch.mm(pred,label_center_emb.t())
                rela_center_vec = F.normalize(rela_center_vec,2,1)
                entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
                entropy = torch.sum(entropy,dim=1)
                value,old_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=True)
                value,novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest = False)
                print(old_indices.shape,novel_indices.shape)
                pair_matrix = torch.mm(rela_center_vec,rela_center_vec.t())

                pairs,pair_labels,_ = pairwise_sample(F.normalize(pred, 2, 1), batch_labels)

                if args.use_cuda:
                    pairs.cuda()
                    pair_labels.cuda()
                    pair_matrix.cuda()
                    # initial_pair_matrix.cuda()
                    model.cuda()

                neg_values, novel_neg_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=False)
                pos_values, novel_pos_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=True)
                neg_values, old_neg_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=False)
                pos_values, old_pos_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=True)

                old_row = torch.LongTensor([[i] * args.oldnum for i in old_indices])
                old_row = old_row.reshape(-1).cuda()
                novel_row = torch.LongTensor([[i] * args.novelnum for i in novel_indices])
                novel_row = novel_row.reshape(-1).cuda()
                row = torch.cat([old_row,novel_row])
                neg_ind = torch.cat([old_neg_ind.reshape(-1),novel_neg_ind.reshape(-1)])
                pos_ind = torch.cat([old_pos_ind.reshape(-1),novel_pos_ind.reshape(-1)])
                neg_distances = (pred[row] - pred[neg_ind]).pow(2).sum(1).unsqueeze(-1)
                pos_distances = (pred[row] - pred[pos_ind]).pow(2).sum(1).unsqueeze(-1)

                loss = torch.mean(torch.clamp(pos_distances + args.a - neg_distances, min=0.0))


                losses.append(loss.item())
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_seconds_spent = time.time() - start_batch
                seconds_train_batches.append(batch_seconds_spent)
                # end one batch

            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.finetune_epochs, total_loss)
            mins_spent = (time.time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_train_epochs.append(mins_spent)

            extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
            # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
            test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                  save_path_i, args, True)



        # Save model
        model_path = save_path_i + '/models'
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        p = model_path + '/finetune.pt'
        torch.save(model.state_dict(), p)
        print('finetune model saved after epoch ', str(epoch))

        # Save time spent on epochs
        np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
        print('Saved mins_train_epochs.')
        # Save time spent on batches
        np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
        print('Saved seconds_train_batches.')

        return model

