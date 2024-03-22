import torch
import torch.nn.functional as F
import dgl
from scipy import sparse
import os
import numpy as np


def make_onehot(input, classes):
    input = torch.LongTensor(input).unsqueeze(1)
    result = torch.zeros(len(input),classes).long()
    result.scatter_(dim=1,index=input.long(),src=torch.ones(len(input),classes).long())
    return result

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def DS_Combin(alpha, classes):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """

    def DS_Combin_two(alpha1, alpha2, classes):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a, u_a

    if len(alpha)==1:
        S = torch.sum(alpha[0], dim=1, keepdim=True)
        u = classes / S
        return alpha[0],u
    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_a,u_a = DS_Combin_two(alpha[0], alpha[1], classes)
        else:
            alpha_a,u_a = DS_Combin_two(alpha_a, alpha[v + 1], classes)
    return alpha_a,u_a



def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees()
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()
    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "w") as f:
        f.write(message)
    return num_isolated_nodes


def get_dgl_data(dataset, views,noise):
    g_dict = {}
    path = "../data/{}/".format(dataset)
    features = torch.FloatTensor(np.load(path + "features.npy"))
    times = np.load(path + "time.npy")
    times = torch.FloatTensor(((times - times.min()).astype('timedelta64[D]') / np.timedelta64(1, 'D')))
    labels = np.load(path + "label.npy")
    for v in views:
        if v == "h":
            matrix = sparse.load_npz(path + "s_tweet_tweet_matrix_{}.npz".format(v+noise))
            #matrix = np.load(path + "matrix_{}.npy".format(v+noise))
        else:
            matrix = sparse.load_npz(path+"s_tweet_tweet_matrix_{}.npz".format(v))
        g = dgl.DGLGraph(matrix, readonly=True)
        save_path_v = path + v
        if not os.path.exists(save_path_v):
            os.mkdir(save_path_v)
        num_isolated_nodes = graph_statistics(g, save_path_v)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.readonly(readonly_state=True)
        # g.ndata['features'] = features
        # # g.ndata['labels'] = labels
        # g.ndata['times'] = times
        g_dict[v] = g
    return g_dict, times, features, labels



def split_data(length, train_p, val_p, test_p):
    indices = torch.randperm(length)
    val_samples = int(length * val_p)
    val_indices = indices[:val_samples]
    test_samples = val_samples + int(length * test_p)
    test_indeces = indices[val_samples:test_samples]
    train_indices = indices[test_samples:]
    return train_indices, val_indices, test_indeces

def ava_split_data(length, train_p, val_p, test_p, labels, classes):
    indices = torch.randperm(length)
    labels = torch.LongTensor(labels[indices])

    train_indices = []
    test_indices = []
    val_indices = []

    for l in range(classes):
        l_indices = torch.LongTensor(np.where(labels.numpy() == l)[0].reshape(-1))
        val_indices.append(l_indices[:20].reshape(-1,1))
        test_indices.append(l_indices[20:50].reshape(-1,1))
        train_indices.append(l_indices[50:].reshape(-1,1))

    val_indices = indices[torch.cat(val_indices,dim=0).reshape(-1)]
    test_indices = indices[torch.cat(test_indices,dim=0).reshape(-1)]
    train_indices = indices[torch.cat(train_indices,dim=0).reshape(-1)]
    print(train_indices.shape,val_indices.shape,test_indices.shape)
    print(train_indices)
    return train_indices, val_indices, test_indices



