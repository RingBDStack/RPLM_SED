import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from struct_att import StructAttention

import torch



class PairPfxTuningEncoder(nn.Module):
    def __init__(self, plm_path,  from_config=False,
                 use_ctx_att=True, ctx_att_head_num=2):
        super().__init__()
        self.plm_path = plm_path

        if from_config:
            config = AutoConfig.from_pretrained(plm_path)
            self.plm = AutoModel.from_config(config)
        else:
            self.plm = AutoModel.from_pretrained(plm_path)

        self.plm_oupt_dim = self.plm.config.hidden_size

        self.plm_emb_dim = self.plm.embeddings.word_embeddings.embedding_dim

    def feat_size(self):
        return self.plm_oupt_dim
    def forward(self, inputs, types, mask):

        txt_emb = self.plm.embeddings(inputs)
        # embed= txt_emb #
        att_msk = mask[:, None, None, :]
        att_msk = (1.0 - att_msk.float()) * torch.finfo(torch.float).min
        plm_oupt = self.plm.encoder(txt_emb, att_msk, output_hidden_states=True)

        hidden = plm_oupt['last_hidden_state']
        msk = mask
        feat = hidden * msk.int().unsqueeze(-1)
        feat = feat.sum(dim=-2) / msk.sum(-1, keepdims=True)

        return feat
