import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl

class Tem_Agg_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual):
        super(Tem_Agg_Layer, self).__init__()
        # equation (1) reference: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.temporal_fc = nn.Linear(out_dim, 1, bias=False)
        self.reset_parameters()
        self.use_residual = use_residual

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)


    def edge_attention(self, edges):
        deltas = edges.src['t'] - edges.dst['t']
        deltas = deltas.cpu().detach().numpy()
        # weights = self.time_similarity(deltas)
        weights = -abs(deltas)
        return {'e': torch.tensor(weights).unsqueeze(1).cuda()}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(torch.exp(self.temporal_fc(nodes.mailbox['z']) * nodes.mailbox['e'] / 500), dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        #h = torch.sum(nodes.mailbox['z'],dim=1)
        return {'h': h}

    def forward(self, blocks, layer_id):
        h = blocks[layer_id].srcdata['features']
        z = self.fc(h)
        blocks[layer_id].srcdata['z'] = z
        z_dst = z[:blocks[layer_id].number_of_dst_nodes()]

        blocks[layer_id].dstdata['z'] = z_dst
        blocks[layer_id].apply_edges(self.edge_attention)

        blocks[layer_id].update_all(  # block_id – The block to run the computation.
                         self.message_func,  # Message function on the edges.
                         self.reduce_func)  # Reduce function on the node.

        if self.use_residual:
            return z_dst + blocks[layer_id].dstdata['h']  # residual connection
        return blocks[layer_id].dstdata['h']



class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,use_residual=False):
        super(GNN, self).__init__()
        self.layer1 = Tem_Agg_Layer(in_dim, hidden_dim,use_residual)
        self.layer2 = Tem_Agg_Layer(hidden_dim, out_dim,use_residual)

    def forward(self, blocks):
        h = self.layer1(blocks, 0)
        h = F.elu(h)
        # print(h.shape)
        blocks[1].srcdata['features'] = h
        h = self.layer2(blocks, 1)
        #h = F.normalize(h, p=2, dim=1)
        return h


class EDNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_dropout=True):
        super(EDNN,self).__init__()
        self.use_dropout = use_dropout
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        if self.use_dropout:
            hidden = F.dropout(hidden, training=self.training)
        out = self.fc2(hidden)
        return out



class ETGNN(nn.Module):
    def __init__(self, GNN_in_dim, GNN_h_dim, GNN_out_dim, E_h_dim, E_out_dim, views):
        super(ETGNN,self).__init__()
        self.views = views
        self.GNN = GNN(GNN_in_dim, GNN_h_dim,GNN_out_dim)
        self.EDNNs = nn.ModuleList([EDNN(GNN_out_dim, E_h_dim, E_out_dim) for v in self.views])

    def forward(self, blocks_dict, is_EDNN_input = False, i=None, emb_v = None):
        out = dict()
        if not is_EDNN_input:
            emb = dict()
            for i,v in enumerate(self.views):
                emb[v] = self.GNN(blocks_dict[v])
                out[v] = self.EDNNs[i](emb[v])
            return out,emb
        else:
            out = self.EDNNs[i](emb_v)
            return out











