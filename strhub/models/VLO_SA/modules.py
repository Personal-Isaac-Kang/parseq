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

from strhub.models.attention import MultiheadAttention


@dataclass
class Module_Data:
    sa_weights: torch.Tensor=None


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)
    

class GradientDisentangledTokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int, base_embedding):
        super().__init__()
        self.base_embedding = base_embedding
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        x = self.base_embedding(tokens)
        return x.detach() + math.sqrt(self.embed_dim) * self.embedding(tokens)
    

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
    

class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, vis, lan, pos, dummy, attn_mask:Optional[Tensor]=None, padding_mask:Optional[Tensor]=None, debug=False):
        aggs = []
        for i, dec_layer in enumerate(self.layers):
            vis, lan, pos, agg = dec_layer(vis, lan, pos, dummy, attn_mask, padding_mask, debug=debug)
            aggs.append(agg)
            
        vis = self.norm(vis)
        lan = self.norm(lan)
        pos = self.norm(pos)
        return vis, lan, pos, aggs
    

class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.mha_V = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mha_L = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mha_O = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ff = FeedForwardLayer(d_model, dim_feedforward, dropout, activation)
        
        self.norm_L = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_O = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.dummy_emb = torch.zeros((1, d_model))
        

    def forward(self, V:Tensor, L:Tensor, O:Tensor, dummy_emb:Tensor,
                attn_mask:Optional[Tensor]=None, padding_mask:Optional[Tensor]=None, debug=False):
        """
        Vision-Langauge-Position Transformer decoder.
        
        Dummy token is added to handle the softmax gradient error when all keys are masked.
        """
        L_V = V.shape[1]
        L_L = L.shape[1]
        L_O = O.shape[1]
        
        V_norm = V
        L_norm = self.norm_L(L)
        O_norm = self.norm_O(O)
        embs_norm = torch.cat([V_norm, L_norm, O_norm, dummy_emb], dim=1)
        
        # import ipdb; ipdb.set_trace(context=11) # #FF0000
        attn_mask_V, attn_mask_L, attn_mask_O, _ = torch.split(attn_mask, [L_V, L_L, L_O, 1], dim=0)
        
        # SA
        V_res, _ = self.mha_V(V_norm, embs_norm, embs_norm, attn_mask=attn_mask_V, key_padding_mask=padding_mask)
        L_res, _ = self.mha_L(L_norm, embs_norm, embs_norm, attn_mask=attn_mask_L, key_padding_mask=padding_mask)
        O_res, _ = self.mha_O(O_norm, embs_norm, embs_norm, attn_mask=attn_mask_O, key_padding_mask=padding_mask)
        V = V + self.dropout1(V_res)
        L = L + self.dropout1(L_res)
        O = O + self.dropout1(O_res)
        embs = torch.cat([V, L, O], dim=1)
        
        # FF
        embs_res = self.ff(embs)
        embs = embs + self.dropout2(embs_res)
        V, L, O = torch.split(embs, [L_V, L_L, L_O], dim=1)
        
        return V, L, O, None
    

class FeedForwardLayer(nn.Module):
    """Transformer position-wise feed-forward layer"""
    
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation='gelu', layer_norm_eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = transformer._get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))