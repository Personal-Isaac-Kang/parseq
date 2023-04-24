"""
MHA, mostly from pytorch nn.MultiHeadAttention
"""
import warnings
from typing import Optional, Tuple
import math

import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import Module
import torch.nn.functional as F


class MultiheadAttentionSep(Module):
    """MultiheadAttention with separately handled QKV calculation for different modalities. """

    __constants__ = ['batch_first']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, batch_first=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        assert self.batch_first, "Currently only implemented for batch first."
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight1 = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_weight2 = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))

        self.in_proj_bias1 = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.in_proj_bias2 = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight1)
        xavier_uniform_(self.in_proj_weight2)
        if self.in_proj_bias1 is not None:
            constant_(self.in_proj_bias1, 0.)
            constant_(self.in_proj_bias2, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, V: Tensor, L: Tensor, O: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        
        query1 = O
        query2 = O
        key1 = V
        key2 = L
        value1 = V
        value2 = L
        
        L_V = V.shape[1]
        L_L = L.shape[1]
        L_O = O.shape[1]
        attn_mask1 = attn_mask[-L_O:, :L_V] # OV
        attn_mask2 = attn_mask[-L_O:, L_V:L_V + L_L]
        if key_padding_mask is not None:
            key_padding_mask1 = key_padding_mask[:, :L_V]
            key_padding_mask2 = key_padding_mask[:, L_V:L_V + L_L]
        else:
            key_padding_mask1, key_padding_mask2 = None, None
        
        assert query1.dim() == 3, "Query should be batched."
        query1, query2, key1, value1, key2, value2 = [x.transpose(1, 0) for x in (query1, query2, key1, value1, key2, value2)]

        attn_output, attn_output_weights = multi_head_attention_forward(
            query1, query2, key1, key2, value1, value2, self.embed_dim, self.num_heads,
            self.in_proj_weight1, self.in_proj_weight2,
            self.in_proj_bias1, self.in_proj_bias2,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask1=key_padding_mask1, key_padding_mask2=key_padding_mask2,
            need_weights=need_weights,
            attn_mask1=attn_mask1, attn_mask2=attn_mask2,
            average_attn_weights=average_attn_weights)
        return attn_output.transpose(1, 0), attn_output_weights


def multi_head_attention_forward(
    query1: Tensor,
    query2: Tensor,
    key1: Tensor,
    key2: Tensor,
    value1: Tensor,
    value2: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight1: Optional[Tensor],
    in_proj_weight2: Optional[Tensor],
    in_proj_bias1: Optional[Tensor],
    in_proj_bias2: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask1: Optional[Tensor] = None,
    key_padding_mask2: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask1: Optional[Tensor] = None,
    attn_mask2: Optional[Tensor] = None,
    average_attn_weights: bool = True
) -> Tuple[Tensor, Optional[Tensor]]:
    # set up shape vars
    assert query1.shape == query2.shape, "Implemented assuming query1 == query2 == O"
    tgt_len, bsz, embed_dim = query1.shape
    src_len1, _, _ = key1.shape
    src_len2, _, _ = key2.shape
    assert embed_dim == embed_dim_to_check, f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    head_dim = embed_dim // num_heads

    # compute in-projection
    q1, k1, v1 = _in_projection_packed(query1, key1, value1, in_proj_weight1, in_proj_bias1)
    q2, k2, v2 = _in_projection_packed(query2, key2, value2, in_proj_weight2, in_proj_bias2)

    if attn_mask1 is not None:
        assert attn_mask1.is_floating_point() or attn_mask1.dtype == torch.bool, f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask1.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask1.dim() == 2:
            correct_2d_size = (tgt_len, src_len1)
            if attn_mask1.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask1.shape}, but should be {correct_2d_size}.")
            attn_mask1 = attn_mask1.unsqueeze(0)
            
    if attn_mask2 is not None:
        assert attn_mask2.is_floating_point() or attn_mask2.dtype == torch.bool, f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask2.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask2.dim() == 2:
            correct_2d_size = (tgt_len, src_len2)
            if attn_mask2.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask2.shape}, but should be {correct_2d_size}.")
            attn_mask2 = attn_mask2.unsqueeze(0)

    # prep key padding mask
    if key_padding_mask1 is not None and key_padding_mask1.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask1 in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask1 = key_padding_mask1.to(torch.bool)
    
    # prep key padding mask
    if key_padding_mask2 is not None and key_padding_mask2.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask2 in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask2 = key_padding_mask2.to(torch.bool)

    # reshape q, k, v for multihead attention and make em batch first
    q1 = q1.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    q2 = q2.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k1 = k1.contiguous().view(src_len1, bsz * num_heads, head_dim).transpose(0, 1)
    k2 = k2.contiguous().view(src_len2, bsz * num_heads, head_dim).transpose(0, 1)
    v1 = v1.contiguous().view(src_len1, bsz * num_heads, head_dim).transpose(0, 1)
    v2 = v2.contiguous().view(src_len2, bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len1 = k1.size(1)
    src_len2 = k2.size(1)
    src_len = src_len1 + src_len2

    # merge key padding and attention masks
    if key_padding_mask1 is not None:
        assert key_padding_mask1.shape == (bsz, src_len1), \
            f"expecting key_padding_mask1 shape of {(bsz, src_len1)}, but got {key_padding_mask1.shape}"
        key_padding_mask1 = key_padding_mask1.view(bsz, 1, 1, src_len1).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len1)
        if attn_mask1 is None:
            attn_mask1 = key_padding_mask1
        elif attn_mask1.dtype == torch.bool:
            attn_mask1 = attn_mask1.logical_or(key_padding_mask1)
        else:
            attn_mask1 = attn_mask1.masked_fill(key_padding_mask1, float("-inf"))
            
    # merge key padding and attention masks
    if key_padding_mask2 is not None:
        assert key_padding_mask2.shape == (bsz, src_len2), \
            f"expecting key_padding_mask2 shape of {(bsz, src_len2)}, but got {key_padding_mask2.shape}"
        key_padding_mask2 = key_padding_mask2.view(bsz, 1, 1, src_len2).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len2)
        if attn_mask2 is None:
            attn_mask2 = key_padding_mask2
        elif attn_mask2.dtype == torch.bool:
            attn_mask2 = attn_mask2.logical_or(key_padding_mask2)
        else:
            attn_mask2 = attn_mask2.masked_fill(key_padding_mask2, float("-inf"))

    # convert mask to float
    if attn_mask1 is not None and attn_mask1.dtype == torch.bool:
        new_attn_mask1 = torch.zeros_like(attn_mask1, dtype=q.dtype)
        new_attn_mask1.masked_fill_(attn_mask1, float("-inf"))
        attn_mask1 = new_attn_mask1
    
    # convert mask to float
    if attn_mask2 is not None and attn_mask2.dtype == torch.bool:
        new_attn_mask2 = torch.zeros_like(attn_mask2, dtype=q.dtype)
        new_attn_mask2.masked_fill_(attn_mask2, float("-inf"))
        attn_mask2 = new_attn_mask2

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # calculate attention and out projection
    attn_output, attn_output_weights = _scaled_dot_product_attention(q1, q2, k1, k2, v1, v2, attn_mask1, attn_mask2, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None


def _in_projection_packed(q, k, v, w, b):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
    q1: Tensor,
    q2: Tensor,
    k1: Tensor,
    k2: Tensor,
    v1: Tensor,
    v2: Tensor,
    attn_mask1: Optional[Tensor] = None,
    attn_mask2: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:

    B, Nt, E = q1.shape
    q1 = q1 / math.sqrt(E)
    q2 = q2 / math.sqrt(E)

    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask1 is not None:
        attn1 = torch.baddbmm(attn_mask1, q1, k1.transpose(-2, -1))
    else:
        attn1 = torch.bmm(q1, k1.transpose(-2, -1))
    if attn_mask2 is not None:
        attn2 = torch.baddbmm(attn_mask2, q2, k2.transpose(-2, -1))
    else:
        attn2 = torch.bmm(q2, k2.transpose(-2, -1))
    
    attn = torch.cat([attn1, attn2], dim=-1) # along Ns dim
    attn = F.softmax(attn, dim=-1)

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, torch.cat([v1, v2], dim=1))
    return output, attn