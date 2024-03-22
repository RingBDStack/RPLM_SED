import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from struct_att import StructAttention

from peft import LoraConfig, get_peft_model, PeftType, PrefixTuningConfig,PrefixEncoder
import torch




class PairPfxTuningEncoder(nn.Module):
    def __init__(self, pmt_len,
                 plm_path, plm_tuning=False, from_config=False,
                 use_ctx_att=True, ctx_att_head_num=2):
        super().__init__()
        self.pfx_len = pmt_len
        self.plm_path = plm_path

        if from_config:
            config = AutoConfig.from_pretrained(plm_path)
            self.plm = AutoModel.from_config(config)
        else:
            self.plm = AutoModel.from_pretrained(plm_path)

        self.plm_oupt_dim = self.plm.config.hidden_size

        self.plm_emb_dim = self.plm.embeddings.word_embeddings.embedding_dim

        self.pfx_embedding = nn.Embedding(self.pfx_len * 2, self.plm_emb_dim)
        self.pfx_mask = torch.ones((1, self.pfx_len), dtype=torch.bool)

        self.linear = nn.Linear(self.plm_oupt_dim, self.plm_oupt_dim // 2)

        self.ctx_att = None
        if use_ctx_att:
            self.ctx_att = StructAttention( self.plm_oupt_dim // 2, self.plm_oupt_dim // 4, att_head_num=ctx_att_head_num)
        self.pair_cls = nn.Linear(2 * (self.plm_oupt_dim // 2), 1)
        LORA_R = 4
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.2

        peft_config = LoraConfig(
            peft_type=PeftType.ADALORA,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none"
        )

        # peft_config = PrefixTuningConfig(
        #     peft_type="PREFIX_TUNING",
        #     task_type="SEQ_2_SEQ_LM",
        #     num_virtual_tokens=20,
        #     token_dim=768,
        #     num_transformer_submodules=1,
        #     num_attention_heads=12,
        #     num_layers=12,
        #     encoder_hidden_size=768,
        # )

        self.plm = get_peft_model(self.plm, peft_config)

    def feat_size(self):
        return self.plm_oupt_dim // 2
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
    def forward(self, inputs, types, prompt, mask):
        bsz, txt_len = mask.size()

        pmt_msk = self.pfx_mask.to(inputs.device).expand(bsz, -1)
        ext_msk = torch.cat([pmt_msk, mask], dim=-1)
        # ext_msk =mask#

        pmt_emb = self.pfx_embedding(prompt)
        pmt_len = prompt.size(-1)
        txt_emb = self.plm.embeddings(inputs)
        embed = torch.cat([pmt_emb, txt_emb], dim=1)
        # embed= txt_emb #
        att_msk = ext_msk[:, None, None, :]
        att_msk = (1.0 - att_msk.float()) * torch.finfo(torch.float).min
        plm_oupt = self.plm.encoder(embed, att_msk, output_hidden_states=True)

        hidden = plm_oupt['last_hidden_state']
        # if self.ctx_att is not None:
        hidden = torch.tanh(self.linear(hidden))

        pmt_feat = hidden[:, :pmt_len, ...]
        tok_feat = hidden[:, pmt_len:, ...]
        # tok_feat = hidden#

        left_msk = (1 - types) * mask
        left_feat = tok_feat * left_msk.unsqueeze(-1)
        left_msk =  torch.cat([pmt_msk.int(), left_msk], dim=1)
        left_feat = torch.cat([pmt_feat, left_feat], dim=1)
        if self.ctx_att is None:
            left_feat = left_feat.sum(dim=-2) / left_msk.sum(-1, keepdims=True)
        else:
            left_feat, left_att = self.ctx_att(left_feat.permute(1, 0, 2), mask=left_msk.permute(1, 0))
            left_feat = torch.mean(left_feat, dim=1)

        right_msk = types * mask
        right_feat = tok_feat * right_msk.unsqueeze(-1)
        if self.ctx_att is None:
            right_feat = right_feat.sum(dim=-2) / right_msk.sum(-1, keepdims=True)
        else:
            right_feat, right_att = self.ctx_att(right_feat.permute(1, 0, 2), mask=right_msk.permute(1, 0))
            right_feat = torch.mean(right_feat, dim=1)

        cls_feat = torch.cat([left_feat, right_feat], dim=-1)

        logit = self.pair_cls(cls_feat).squeeze(dim=-1)

        return logit, left_feat
