# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer

from timm.models.vision_transformer import VisionTransformer, PatchEmbed


@dataclass
class Module_Data:
    main_pt_1: torch.Tensor = None # input
    main_pt_2: torch.Tensor = None # after sa
    main_pt_3: torch.Tensor = None # after ca
    main_pt_4: torch.Tensor = None # after ff
    res_pt_1: torch.Tensor = None # residual result of sa
    res_pt_2: torch.Tensor = None # residual result of ca
    res_pt_3: torch.Tensor = None # residual result of ff
    content: torch.Tensor = None
    sa_weights: torch.Tensor = None
    ca_weights: torch.Tensor = None
    

class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)        

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
                       tgt_key_padding_mask: Optional[Tensor], debug: bool):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        if debug:
            agg = Module_Data()
        else:
            agg = None
        
        if debug: agg.content = tgt_kv
        
        # O -> L
        if debug : agg.main_pt_1 = tgt
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)
        if debug : agg.res_pt_1 = tgt2
        if debug : agg.sa_weights = sa_weights
        tgt = tgt + self.dropout1(tgt2)
        if debug : agg.main_pt_2 = tgt

        # O -> V
        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        if debug : agg.res_pt_2 = tgt2
        if debug : agg.ca_weights = ca_weights
        tgt = tgt + self.dropout2(tgt2)
        if debug : agg.main_pt_3 = tgt

        # FF
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        if debug : agg.res_pt_3 = tgt2
        tgt = tgt + self.dropout3(tgt2)
        if debug : agg.main_pt_4 = tgt
        
        return tgt, agg

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True, debug: bool = False):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        # query_mask : Used in content -> pos.
        query, agg = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask, debug)
        if update_content:
            # content_mask : Used in content -> content.
            # content can be updated with the same decoder, with context as query instead of pos. The updated content
            # is used for content input for next decoder layer, if there are more than 1 deocder layers.
            # Basically, a self-attn casual mask with permutation ordering (including self) = LM
            # plus a cross-attn with no mask to memory = vis -> content.
            content, _ = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask, debug)
        return query, content, agg


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm, update_content):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.update_content = update_content

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):
        # query : pos
        # content : content
        # memory : memory
        # query_mask : query_mask
        # content_mask : content_mask
        # content_key_padding_mask : tgt_padding_mask
        aggs = []
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content, agg = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last and self.update_content)
            aggs.append(agg)
        query = self.norm(query)
        return query, aggs


class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=False)  # these disable the classifier head

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)
