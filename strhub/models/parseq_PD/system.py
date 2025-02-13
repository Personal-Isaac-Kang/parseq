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
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .modules import DecoderLayer, Decoder, Encoder, TokenEmbedding
from .utils import ParallelDecoding

DEBUG_LAYER_INDEX = 0

@dataclass
class System_Data:
    sa_weights: torch.Tensor = None



class PARSeq_PD(CrossEntropySystem):
    """PARSeq with scheduled parallel decoding. """

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 mask_sampling_num: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, dec_iters: int, refine_iters: int, dropout: float,
                 head_char_emb_tying: bool, update_content: bool,
                 debug: bool = False, **kwargs: Any) -> None:
        self.debug = debug
        self.results = []
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay, self.debug)
        print('Model : PARSeq_PD')
        self.save_hyperparameters()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.dec_iters = dec_iters
        self.refine_iters = refine_iters

        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio)
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim), update_content=update_content)

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        
        self.K = mask_sampling_num

        # # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer))
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)
        if head_char_emb_tying:
            self.head.weight = self.text_embed.embedding.weight

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        # tgt : L(ids)
        # memory : V
        # tgt_mask : L-L Q-K mask
        # tgt_padding_mask : O-L, L-L K pad mask
        # tgt_query : O (optional)
        # tgt_query_mask : O-L Q-K mask
        
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        # query : O
        # content : L
        # memory : V
        # query_mask : query_mask
        # content_mask : content_mask
        # content_key_padding_mask : content_key_padding_mask
        
        # tgt_query : O
        # tgt_emb : L
        # memory : V
        # tgt_query_mask : query_mask
        # tgt_mask : content_mask
        # tgt_padding_mask : content_key_padding_mask
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)    

    def forward(self, images: Tensor, debug: bool = False) -> Tensor:
        bs = images.shape[0]
        S = self.max_label_length + 1
        memory = self.encode(images)
        pos_queries = self.pos_queries[:, :S].expand(bs, -1, -1)
        D = pos_queries.shape[-1]
        
        # unmasked regions of previous mask has already been decoded -> they should be untouched
        # the current mask has to be as subset of previous mask. the size of current mask is decided. exact mask is decided by conf.
        # conf of preivously decoded positions should be 1. top-(1-M_cur) confidence positions should be selected -> as long as size(1-M_cur) > size(1-M_prev) and (1-M_prev) regions are conf=1,
        # current mask will be subset previous mask. There are newly decoded positions.
        # when decoding, only L embs of previously decoded positions (excluding [B]) should be unmasked.
        # mask of L and O will be a bit different.  
        par_dec = ParallelDecoding(self.dec_iters, S)
        lan_ids = torch.full((bs, S + 1), self.bos_id, dtype=torch.long, device=self._device)
        O_mask = torch.ones(bs, S, dtype=torch.bool, device=self._device)
        L_mask = F.pad(O_mask, (1, 0), "constant", 0)
        
        def print_parallel_decoding_process(par_dec, t, lan_ids):
            if t == 1:
                print('iter | decode# | result')
            tokens = self.tokenizer._ids2tok(lan_ids, join=False)
            for i, token in enumerate(tokens):
                if i == 0:
                    pass
                elif len(token) != 3:
                    tokens[i] = f' {token} '
                elif token == '[B]':
                    tokens[i] = '   '
                else:
                    pass
            tokens = ''.join(tokens)
            print(f'{t:02d} | {par_dec.get_k(t):02d} | {tokens}')
        
        for t in range(1, self.dec_iters + 1):
            # gather masked O to use as Q, unmasked L to use as K, V 
            # only attention on previously decoded positions are allowed
            pos_queries_t = pos_queries[torch.where(O_mask)].reshape(bs, -1, D)
            pos_out, _ = self.decode(lan_ids, memory, tgt_padding_mask=L_mask, tgt_query=pos_queries_t)
            # tgt : L(ids)
            # memory : V
            # tgt_mask : L-L Q-K mask
            # tgt_padding_mask : O-L, L-L K pad mask
            # tgt_query : O (optional)
            # tgt_query_mask : O-L Q-K mask
            logits_t = self.head(pos_out)
            
            # Confidence values of most probable ids. Previously decoded positions are already filtered out.
            confs, conf_ids = logits_t.softmax(-1).max(-1)
            # Select top-k
            k = par_dec.get_k(t)
            topk, topk_ind = confs.topk(k, -1)
            # update decoded embs & mask & logits
            ys, xs = torch.where(O_mask)
            xs = xs.reshape(bs, -1)
            topk_pos = torch.gather(xs, 1, topk_ind) # topk position in pos_queries
            topk_ids = torch.gather(conf_ids, 1, topk_ind)
            lan_ids.scatter_(1, topk_pos + 1, topk_ids) # +1 for shifted input for L (prepend [B])
            O_mask.scatter_(1, topk_pos, torch.zeros_like(topk_ids).to(torch.bool))
            L_mask = F.pad(O_mask, (1, 0), "constant", 0)
            if t == 1:
                logits = torch.zeros(bs, S, logits_t.shape[-1], dtype=logits_t.dtype, device=self._device)
            b_ind = torch.arange(bs, device=logits.device).unsqueeze(1)
            logits[b_ind, topk_pos, :] = logits_t[b_ind, topk_ind, :]
            # print_parallel_decoding_process(par_dec, t, lan_ids[0])
        return logits, logits, None
    
    # def forward(self, images: Tensor, debug: bool = False) -> Tensor:
    #     bs = images.shape[0]
    #     num_steps = self.max_label_length + 1
    #     memory = self.encode(images)

    #     # Query positions up to `num_steps`
    #     pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

    #     # Special case for the forward permutation. Faster than using `generate_attn_masks()`
    #     tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

    #     tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
    #     tgt_in[:, 0] = self.bos_id

    #     logits = []
    #     for i in range(num_steps):
    #         j = i + 1  # next token index
    #         # Efficient decoding:
    #         # Input the context up to the ith token. We use only one query (at position = i) at a time.
    #         # This works because of the lookahead masking effect of the canonical (forward) AR context.
    #         # Past tokens have no access to future tokens, hence are fixed once computed.
    #         tgt_out, _aggs = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
    #                                 tgt_query_mask=query_mask[i:j, :j])
            
    #         if debug:
    #             _agg = _aggs[DEBUG_LAYER_INDEX]

    #         # the next token probability is in the output's ith token position
    #         p_i = self.head(tgt_out)
    #         logits.append(p_i)
    #         if j < num_steps:
    #             # greedy decode. add the next token index to the target input
    #             tgt_in[:, j] = p_i.squeeze().argmax(-1)
    #             # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
    #             if (tgt_in == self.eos_id).any(dim=-1).all():
    #                 break
    #     logits = torch.cat(logits, dim=1)
        
        
    #     if self.refine_iters:
    #         # For iterative refinement, we always use a 'cloze' mask.
    #         # We can derive it from the AR forward mask by unmasking the token context to the right.
    #         query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
    #         bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
    #         for i in range(self.refine_iters):
    #             # Prior context is the previous output.
    #             tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
    #             tgt_padding_mask = ((tgt_in == self.eos_id).cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
    #             tgt_out, _ = self.decode(tgt_in, memory, tgt_mask[:tgt_in.shape[1], :tgt_in.shape[1]], tgt_padding_mask,
    #                                   tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
    #             logits = self.head(tgt_out)
                
    #     return logits, logits, None



    def forward_logits_loss(self, images, labels):
        L_ids = self.tokenizer.encode(labels, self.device)
        L_ids = F.pad(L_ids, (0, self.max_label_length + 2 - L_ids.shape[1]), "constant", self.pad_id)
        L_ids_tgt = L_ids[:, 1:]
        logits, _, _ = self.forward(images)
        loss = F.cross_entropy(logits.flatten(end_dim=1), L_ids_tgt.flatten(), ignore_index=self.pad_id)
        loss_numel = len(L_ids_tgt)
        return logits, loss, logits, loss, loss_numel
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        bs = images.shape[0]
        memory = self.encode(images)
        L_ids = self.tokenizer.encode(labels, self._device)
        L_ids = F.pad(L_ids, (0, self.max_label_length + 2 - L_ids.shape[1]), "constant", self.pad_id)
        par_dec = ParallelDecoding(self.dec_iters, self.max_label_length + 1)
        K = self.K
        loss = 0
        for k in range(K):
            while True:
                L_mask, O_mask, r, n = par_dec.get_random_mask()
                if n > 1: break
            mask_ind = torch.where(O_mask)[0]
            L_ids_tgt = L_ids[:, 1:][:, mask_ind]
            pos_queries = self.pos_queries[:, :self.max_label_length + 1, :].expand(bs, -1, -1)
            D = pos_queries.shape[-1]
            O_embs = pos_queries[:, mask_ind, :]
            attn_mask_OL = L_mask.expand(len(mask_ind), -1).to(O_embs.device)
            out, _ = self.decode(L_ids, memory, tgt_query_mask=attn_mask_OL, tgt_query=O_embs)
            logits = self.head(out)
            loss += F.cross_entropy(logits.moveaxis(-1, 1), L_ids_tgt)
            self.log('loss', loss)
            if torch.isnan(loss):
                import ipdb; ipdb.set_trace(context=11) # #FF0000
        loss /= K
        results = torch.all(logits.argmax(-1) == L_ids_tgt, -1).tolist()
        self.results.extend(results)
        if len(self.results) > 10000:
            train_acc = sum(self.results) / len(self.results) * 100
            self.log('train_acc', train_acc)
            self.results = []
        return loss
