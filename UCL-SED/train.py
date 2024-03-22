import torch.optim as optim
import torch
import dgl
from utils import make_onehot, DS_Combin
from losses import relu_evidence, common_loss, EUC_loss
from sklearn.metrics import classification_report
import numpy as np
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import torch.nn as nn


def extract_results(g_dict, views, labels, model, args, train_indices=None):
    with torch.no_grad():
        model.eval()
        out_list = []
        emb_list = []
        nids_list = []
        all_indices = torch.LongTensor(range(0, labels.shape[0]))
        if args.use_cuda:
            all_indices = all_indices.cuda()
        print(all_indices)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g_dict[views[0]], all_indices, sampler,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            extract_indices = blocks[-1].dstdata[dgl.NID]
            if args.use_cuda:
                extract_indices = extract_indices.cuda()
            blocks_dict = {}
            blocks_dict[views[0]] = blocks
            for v in views[1:]:
                blocks_v = list(dgl.dataloading.NodeDataLoader(
                    g_dict[v], extract_indices, sampler,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                ))[0][2]
                blocks_dict[v] = blocks_v
            for v in views:
                blocks_dict[v] = [b.to(device) for b in blocks_dict[v]]

            out, emb = model(blocks_dict)
            out_list.append(out)
            emb_list.append(emb)
            nids_list.append(extract_indices)

    # assert batch_id==0
    all_out = {}
    all_emb = {}
    for v in views:
        all_out[v] = []
        all_emb[v] = []
        for out, emb in zip(out_list, emb_list):
            all_out[v].append(out[v])
            all_emb[v].append(emb[v])
        if args.use_cuda:
            all_out[v] = torch.cat(all_out[v]).cpu()
            all_emb[v] = torch.cat(all_emb[v]).cpu()
        else:
            all_out[v] = torch.cat(all_out[v])
            all_emb[v] = torch.cat(all_emb[v])

    extract_nids = torch.cat(nids_list)
    if args.use_cuda:
        extract_nids = extract_nids.cpu()

    return all_out, all_emb, extract_nids


def train_model(model, g_dict, views, features, times, labels, epoch, criterion, mask_path, save_path, args):
    train_indices = torch.load(mask_path + "train_indices.pt")
    val_indices = torch.load(mask_path + "val_indices.pt")
    test_indices = torch.load(mask_path + "test_indices.pt")
    classes = len(set(labels))
    ori_labels = labels

    labels = make_onehot(labels, classes)
    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
    if args.use_cuda:
        model = model.cuda()
        features = features.cuda()
        times = times.cuda()
        labels = labels.cuda()
        train_indices = train_indices.cuda()
        val_indices = val_indices.cuda()
        test_indices = test_indices.cuda()

    for v in views:
        if args.use_cuda:
            g_dict[v] = g_dict[v].to(device)
        g_dict[v].ndata['features'] = features
        g_dict[v].ndata['t'] = times

    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.005)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if args.mode == 0:
        message = "----------begin training---------\n"
        with open(save_path + "log.txt", 'w') as f:
            f.write(message)

        best_vali = 0
        test_acc_in_best_e = 0
        best_epoch = 0
        test_acc_list = []
        label_u = torch.FloatTensor(np.ones(classes))

        for e in range(epoch):
            print(label_u)

            _, GNN_out_fea, extract_nids = extract_results(g_dict, views, labels, model, args)
            extract_labels = ori_labels[extract_nids]
            label_center = {}
            for v in views:
                label_center[v] = []
            for l in range(classes):
                l_indices = torch.LongTensor(np.where(extract_labels == l)[0].reshape(-1))
                for v in views:
                    l_feas = GNN_out_fea[v][l_indices]
                    l_cen = torch.mean(l_feas, dim=0)
                    label_center[v].append(l_cen)

            for v in views:
                label_center[v] = torch.stack(label_center[v], dim=0)
                label_center[v] = F.normalize(label_center[v], 2, 1)
    
                if args.use_cuda:
                    label_center[v] = label_center[v].cuda()
                    label_u = label_u.cuda()

            losses = []
            total_loss = 0
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader = dgl.dataloading.NodeDataLoader(
                g_dict[views[0]], train_indices, sampler,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )

            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                print("batch_id:", batch_id)
                device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
                batch_indices = blocks[-1].dstdata[dgl.NID]
                if args.use_cuda:
                    batch_indices = batch_indices.cuda()
                blocks_dict = {}
                blocks_dict[views[0]] = blocks
                for v in views[1:]:
                    blocks_v = list(dgl.dataloading.NodeDataLoader(
                        g_dict[v], batch_indices, sampler,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                    ))[0][2]
                    blocks_dict[v] = blocks_v

                for v in views:
                    blocks_dict[v] = [b.to(device) for b in blocks_dict[v]]

                batch_labels = labels[batch_indices]
                batch_ori_labels = torch.LongTensor(ori_labels).cuda()[batch_indices]
                model.train()
                out, emb = model(blocks_dict)

                view_contra_loss = 0
                e_loss = 0
                if args.use_uncertainty:
                    alpha = []
                    true_labels = torch.LongTensor(ori_labels).cuda()[batch_indices]
                    for i, v in enumerate(views):
                        emb[v] = F.normalize(emb[v], 2, 1)
                        batch_center = label_center[v][batch_ori_labels]

                        view_contra_loss += torch.mean(-torch.log(
                            (torch.exp(torch.sum(torch.mul(emb[v], batch_center), dim=1)) - 0.1 * label_u[
                                batch_ori_labels]) / (
                                torch.sum(torch.exp(torch.mm(emb[v], label_center[v].T)),
                                          dim=1))))  # *label_u[batch_ori_labels])


                        alpha_v = relu_evidence(out[v]) + 1
                        alpha.append(alpha_v)

                    comb_alpha, comb_u = DS_Combin(alpha=alpha, classes=classes)

                    e_loss = EUC_loss(comb_alpha, comb_u, true_labels, e)
                    loss = e_loss + criterion(comb_alpha, batch_labels, true_labels, e, classes, 100,device) + 2* view_contra_loss  

                else:
                    batch_labels = torch.argmax(batch_labels, 1)
                    for i, v in enumerate(views):
                        if i == 0:
                            comb_out = out[v]
                        else:
                            comb_out += out[v]
                        emb[v] = F.normalize(emb[v], 2, 1)
                        batch_center = label_center[v][batch_ori_labels]
                        view_contra_loss += torch.mean(-torch.log(
                            (torch.exp(torch.sum(torch.mul(emb[v], batch_center), dim=1))) / (
                                torch.sum(torch.exp(torch.mm(emb[v], label_center[v].T)), dim=1))))
                    loss = criterion(comb_out,batch_labels)  # + view_contra_loss

                com_loss = 0
                for i in range(len(emb) - 1):
                    for j in range(i + 1, len(emb)):
                        com_loss += common_loss(emb[views[i]], emb[views[j]])
                loss += 1 * com_loss
                print("com_loss:", 1 * com_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                total_loss += loss.item()
                print(loss)

            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(e + 1, args.epoch, total_loss)
            print(message)
            with open(save_path + '/log.txt', 'a') as f:
                f.write(message)
                f.write("\n")
            out, emb, nids = extract_results(g_dict, views, labels, model, args)
            extract_labels = ori_labels[nids]
            if args.use_uncertainty:
                alpha = []
                for out_v in out.values():
                    evi_v = relu_evidence(out_v)
                    alpha_v = evi_v + 1
                    alpha.append(alpha_v)
                comb_out, comb_u = DS_Combin(alpha=alpha, classes=classes)

                train_labels = extract_labels[train_indices.cpu().numpy()]
                train_u = comb_u[train_indices].cpu().numpy()
                train_i_u = []
                for i in range(classes):
                    i_indices = np.where(train_labels == i)
                    i_u = np.mean(train_u[i_indices])
                    train_i_u.append(i_u)
                label_u = torch.FloatTensor(train_i_u).cuda()
                # print("label_u:",label_u)



            else:
                for i, out_v in enumerate(out.values()):
                    if i == 0:
                        comb_out = out_v
                    else:
                        comb_out += out_v
            _, val_pred = torch.max(comb_out[val_indices], 1)
            val_labels = torch.IntTensor(extract_labels[val_indices.cpu().numpy()])
            val_f1 = f1_score(val_labels.cpu().numpy(), val_pred.cpu().numpy(), average='macro')
            val_match = torch.reshape(torch.eq(val_pred, val_labels).float(), (-1, 1))
            val_acc = torch.mean(val_match)

            _, test_pred = torch.max(comb_out[test_indices], 1)
            test_labels = torch.IntTensor(extract_labels[test_indices.cpu().numpy()])
            test_f1 = f1_score(test_labels.numpy(), test_pred.numpy(), average='macro')
            test_match = torch.reshape(torch.eq(test_pred, test_labels).float(), (-1, 1))
            test_acc = torch.mean(test_match)
            # t = classification_report(test_labels.cpu().numpy(), test_pred.cpu().numpy(), target_names=[i for i in range(classes)])
            message = "val_acc: %.4f val_f1:%.4f  test_acc: %.4f test_f1:%.4f" % (val_acc, val_f1, test_acc, test_f1)
            print(message)
            with open(save_path + '/log.txt', 'a') as f:
                f.write(message)

            test_acc_list.append(test_acc)

            if val_acc > best_vali:
                best_vali = val_acc
                best_epoch = e + 1
                test_acc_in_best_e = test_acc
                p = save_path + 'best.pt'
                torch.save(model.state_dict(), p)

        np.save(save_path + "testacc.npy", np.array(test_acc_list))
        message = "best epoch:%d  test_acc:%.4f" % (best_epoch, test_acc_in_best_e)
        print(message)
        with open(save_path + '/log.txt', 'a') as f:
            f.write(message)

    else:
        model.load_state_dict(torch.load(save_path + '/best.pt'))
        model.eval()
        out, emb, nids = extract_results(g_dict, views, labels, model, args)
        extract_labels = ori_labels[nids]
        if args.use_uncertainty:
            alpha = []
            for v in ['h','u','e']:
                evi_v = relu_evidence(out[v])
                alpha_v = evi_v + 1
                alpha.append(alpha_v)
            comb_out, comb_u = DS_Combin(alpha=alpha, classes=classes)
        else:
            for i, v in enumerate(views):
                if i == 0:
                    comb_out = out[v]
                else:
                    comb_out += out[v]

        _, test_pred = torch.max(comb_out[test_indices], 1)
        test_labels = torch.IntTensor(extract_labels[test_indices.cpu().numpy()])
        if args.use_uncertainty:
            test_u = comb_u[test_indices].cpu().numpy()
            test_match = torch.reshape(torch.eq(test_pred, test_labels).float(), (-1, 1))
            test_i_u = []
            for i in range(classes):
                i_indices = np.where(test_labels.cpu().numpy() == i)
                i_u = np.mean(test_u[i_indices])
                test_i_u.append(i_u)

        test_f1 = f1_score(test_labels.cpu().numpy(), test_pred.cpu().numpy(), average='macro')
        test_match = torch.reshape(torch.eq(test_pred, test_labels).float(), (-1, 1))
        test_acc = torch.mean(test_match)
        t = classification_report(test_labels.cpu().numpy(), test_pred.cpu().numpy())
        message = "test_acc: %.4f test_f1:%.4f" % (test_acc, test_f1)
        print(message)

        p, r, f1, s = precision_recall_fscore_support(test_labels.cpu().numpy(), test_pred.cpu().numpy(),
                                                      labels=np.array([i for i in range(classes)]))
 




