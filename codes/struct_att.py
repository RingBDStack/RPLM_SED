#! -*- coding: utf-8 -*-

import torch
from torch import nn


class StructAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding
    """

    def __init__(self, feat_dim, hid_dim, att_head_num=1):
        """
        Initializes parameters suggested in paper
        Args:
            feat_dim:       {int} hidden dimension for lstm
            hid_dim:        {int} hidden dimension for the dense layer
            att_head_num:   {int} attention-hops or attention heads
        Returns:
            self
        Raises:
            Exception
        """
        super(StructAttention, self).__init__()
        self.W1 = torch.nn.Linear(feat_dim, hid_dim, bias=False)
        nn.init.xavier_normal_(self.W1.weight)

        self.W2 = torch.nn.Linear(hid_dim, att_head_num, bias=False)
        nn.init.xavier_normal_(self.W2.weight)

        self.att_head_num = att_head_num

    def forward(self, inpt, mask=None):
        """
        :param inpt: [len, bsz, dim]
        :param mask: [len, bsz]
        :return: [bsz, head_num, dim], [bsz, head_num, len]
        """
        hid = torch.tanh(self.W1(inpt))
        hid = self.W2(hid)

        if mask is not None:
            mask = mask.float().unsqueeze(-1).expand(-1, -1, self.att_head_num)
            mask = (1. - mask) * 1e10
            hid = hid - mask
        att = torch.softmax(hid, dim=0).permute(1, 2, 0)

        outp = att @ inpt.permute(1, 0, 2)

        return outp, att
