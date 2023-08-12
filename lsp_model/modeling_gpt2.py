# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals


import logging
import copy
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# pytorch_pretrained_bert.__file__
# /misc/kfdata01/kf_grp/lchen/anaconda3/envs/sc-gpt/lib/python3.7/site-packages/pytorch_pretrained_bert
from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model, GPT2LMHead, Attention, Block, LayerNorm, MLP
# from pytorch_pretrained_bert.modeling_gpt2 import GPT2Model, GPT2LMHead, Attention, Block, LayerNorm, MLP
# from transformers import GPT2PreTrainedModel

logger = logging.getLogger(__name__)

# 调用顺序：Attention --> Block --> GPT2Model --> GPT2LMHeadModel --> train

class AttentionFP16(Attention):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(AttentionFP16, self).__init__(nx, n_ctx, config, scale)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)    # point out by Yen-Chun, FP16 overflow

        w = nn.Softmax(dim=-1)(w)
        # return torch.matmul(w, v)
        return torch.matmul(w, v), w

    # 重写：增加 output_attention
    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a, attention = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present, attention


class BlockFP16(Block):
    def __init__(self, n_ctx, config, scale=False):
        super(BlockFP16, self).__init__(n_ctx, config, scale)
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = AttentionFP16(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    # 重写：增加 output_attention
    def forward(self, x, layer_past=None):
        a, present, attention = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present, attention


class GPT2ModelFP16(GPT2Model):
    def __init__(self, config):
        super(GPT2ModelFP16, self).__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = BlockFP16(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.apply(self.init_weights)

    # 重写，增加 output_attention 参数
    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []

        all_self_attentions = []
        for block, layer_past in zip(self.h, past):
            # hidden_states, present = block(hidden_states, layer_past)
            hidden_states, present, attention = block(hidden_states, layer_past)
            presents.append(present)
            all_self_attentions.append(attention)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        # return hidden_states.view(*output_shape), presents
        return hidden_states.view(*output_shape), presents, all_self_attentions



class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2ModelFP16(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents, attentions = self.transformer(input_ids, position_ids, token_type_ids, past)
        # import pdb; pdb.set_trace()
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)) # [bz*seq_len, V], [bz*seq_len]
            loss1 = loss1.view(lm_labels.size(0), lm_labels.size(1)) # [bz, seq_len]，里边padding位置是0？
            label_size = torch.sum(lm_labels != -1, dim=1).type(loss1.type()) # [bz]

            # token-level loss = total_CE_loss / target_label_num
            loss = torch.sum(loss1)/torch.sum(label_size) 
            # seq-level ppl = exp(seq_level_loss)
            ppl = torch.exp(torch.mean(torch.sum(loss1, dim=1).float() / label_size.float()))
            # ppl = torch.mean(torch.exp(torch.sum(loss1, dim=1)/label_size))
            return loss, ppl, attentions
        return lm_logits, presents, attentions
    
    # 没用过吧
    # def forward_pointwise(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
    #     hidden_states, presents, attentions = self.transformer(input_ids, position_ids, token_type_ids, past, output_attention=True)
    #     # import pdb; pdb.set_trace()
    #     lm_logits = self.lm_head(hidden_states)
    #     if lm_labels is not None:
    #         # loss_fct = CrossEntropyLoss(ignore_index=-1)
    #         # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
    #         loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
    #         loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)),
    #                           lm_labels.view(-1))
    #         loss1 = loss1.view(lm_labels.size(0), lm_labels.size(1))
    #         label_size = torch.sum(lm_labels != -1, dim=1).type(loss1.type())
    #         loss1 = torch.sum(loss1, dim=1)/label_size
    #         ppl1 = torch.exp(loss1)

    #         return loss1, ppl1
    #     return lm_logits, presents
