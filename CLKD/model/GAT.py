import torch.nn as nn
import torch
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # compute attention
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # aggregate weighted features
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, blocks, layer_id):
        h = blocks[layer_id].srcdata['h']
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


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual))
        self.merge = merge

    def forward(self, blocks, layer_id):
        head_outs = [attn_head(blocks, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

    def forward(self, blocks, args, trans=False, src=None, tgt=None):
        if trans:
            if args.mode == 4:
                features = blocks[0].srcdata['tranfeatures']
                print("This is nonlinear trans!")
                blocks[0].srcdata['h'] = features
            # note that the features contain two parts: 1) semantic part and 2) temporal part, here we only transform the semantic part
            if args.mode == 2 and args.add_mapping:
                features = blocks[0].srcdata['h'].cpu().detach()
                W = torch.from_numpy(
                    torch.load('./datasets/LinearTranWeight/spacy_{}_{}/best_mapping.pth'.format(src, tgt)))
                print("This is linear trans!")
                part1 = torch.index_select(features, 1, torch.tensor(range(0, args.word_embedding_dim)))
                part1 = torch.matmul(part1, torch.FloatTensor(W))
                part2 = torch.index_select(features, 1,
                                           torch.tensor(range(args.word_embedding_dim, features.size()[1])))
                features = torch.cat((part1, part2), 1).cuda()
                blocks[0].srcdata['h'] = features

        h = self.layer1(blocks, 0)
        h = F.elu(h)
        # print(h.shape)
        blocks[1].srcdata['h'] = h
        h = self.layer2(blocks, 1)
        # h = F.normalize(h, p=2, dim=1)
        return h
