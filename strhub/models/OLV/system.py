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
from typing import Sequence, Any, Optional, List, Tuple
from dataclasses import dataclass

import os
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .modules import DecoderLayer, Decoder, Encoder, TokenEmbedding, GradientDisentangledTokenEmbedding
from .utils import AttentionMask

@dataclass
class System_Data:
    sa_weights_dec: Tensor = None
    sa_weights_ref: Tensor = None

class OLV(CrossEntropySystem):
    # TODO :  Implement PARSeq

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int, ref_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool, decode_ar: bool,
                 dropout: float, ref_iters: int,
                 debug: bool = False, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay, debug)
        print('Model : OLV')
        self.save_hyperparameters()
        self.debug = debug
        self.results_dec = []
        self.results_ref = []
        self.decode_ar = decode_ar
        
        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        
        self.max_label_length = max_label_length

        # Model
        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio)
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim), dropout=dropout)
        self.refiner = Decoder(decoder_layer, num_layers=ref_depth, norm=nn.LayerNorm(embed_dim), dropout=dropout) if ref_depth > 0 else None
        self.ref_iters = ref_iters
        
        # Character Embeddings / Heads
        self.char_embed_dec = TokenEmbedding(len(self.tokenizer), embed_dim)
        self.char_embed_ref = TokenEmbedding(len(self.tokenizer), embed_dim)
        self.char_head_dec = nn.Linear(embed_dim, len(self.tokenizer))
        self.char_head_dec.weight = self.char_embed_dec.embedding.weight
        self.char_head_ref = nn.Linear(embed_dim, len(self.tokenizer))
        self.char_head_ref.weight = self.char_embed_ref.embedding.weight
        self.rtd_head_ref = nn.Linear(embed_dim, 1)

        # Positional Embeddings
        self.pos_embed_dec_L = nn.Parameter(torch.Tensor(1, max_label_length + 2, embed_dim)) # +2 for [B], [E]
        self.pos_embed_dec_O = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim)) # +1 for [E]
        self.pos_embed_ref_L = nn.Parameter(torch.Tensor(1, max_label_length + 2, embed_dim))
        self.pos_embed_ref_O = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        
        # Modality Embeddings
        self.modal_embed  = nn.Parameter(torch.Tensor(1, 3, embed_dim))
        
        # Initialization
        named_apply(partial(init_weights, exclude=['encoder']), self) # Encoder has its own init.
        nn.init.trunc_normal_(self.pos_embed_dec_L, std=.02)
        nn.init.trunc_normal_(self.pos_embed_dec_O, std=.02)
        nn.init.trunc_normal_(self.pos_embed_ref_L, std=.02)
        nn.init.trunc_normal_(self.pos_embed_ref_O, std=.02)
        nn.init.trunc_normal_(self.modal_embed, std=.02)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {
            'char_embed_dec.embedding.weight',
            'char_embed_ref.embedding.weight',
            'pos_embed_dec_L',
            'pos_embed_dec_O'
            'pos_embed_ref_L',
            'pos_embed_ref_O',
            'modal_embed'
            }
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)
    
    def encode(self, img: Tensor):
        """
        Encodes image into a sequence of visual embeddings.
        """
        V = self.encoder(img)
        V = V + self.modal_embed[:, 0]
        return V
    
    def to_L(self, L_ids, module):
        """
        Converts tensor of language token ids to character embeddings.
        
        Args:
            L_ids (Tensor): [N, L_L] tensor of language token ids. Starts with [B] token.
        
        Returns:
            L (Tensor): [N, L_L, E] tensor of character embeddings.
        """
        bs, L_L = L_ids.shape
        if module == 'decoder':
            null_ctx = self.char_embed_dec(L_ids[:, :1])
            L = torch.cat([null_ctx, self.char_embed_dec(L_ids[:, 1:]) + self.pos_embed_dec_L[:, :L_L - 1]], dim=1)
            L = L +  self.modal_embed[:, 1]
        elif module == 'refiner':
            null_ctx = self.char_embed_ref(L_ids[:, :1])
            L = torch.cat([null_ctx, self.char_embed_ref(L_ids[:, 1:]) + self.pos_embed_dec_L[:, :L_L - 1]], dim=1)
            L = L + self.modal_embed[:, 1]
        else:
            raise Exception()
        return L
    
    def get_O(self, L_O, bs, module):
        """
        Gets Ordinal embeddings.
        """
        if module == 'decoder':
            O = self.pos_embed_dec_O[:, :L_O].expand(bs, -1, -1)
        elif module == 'refiner':
            O = self.pos_embed_ref_O[:, :L_O].expand(bs, -1, -1)
        else:
            raise Exception()
        O = O + self.modal_embed[:, 2]
        return O

    def forward(self, images: Tensor, validation: bool = False, debug: bool = False, DEC_IDX: int = 0, REF_IDX: int = 0) -> Tensor:
        """
        Forward-pass for test & val.
        
        Args:
            images (Tensor): input image
            validation (bool): Is validation step.
            debug: Is debug mode.
            DEC_IDX: Target debugging decoder index.
            REF_IDX: Target debugging refiner index.
            
        Returns:
            logits : Tensor of logits.
                     Shape [N, L_O, C] in case of validation,
                     Shape [N, <=L_O, C] in case of test.
        """
        if debug :
            agg_system = System_Data()
        else:
            agg_system = None
        
        testing = not validation
        bs = images.shape[0]
        L_L = self.max_label_length + 1 # +1 for [B]
        L_O = num_steps = self.max_label_length + 1 # +1 for [E]
        
        #@ decoder
        #* prepare embs
        
        V = self.encode(images)
        L_V = V.shape[1]
        
        # Initialize L_ids
        L_ids = torch.full((bs, L_L), self.pad_id, dtype=torch.long, device=self._device)
        L_ids[:, 0] = self.bos_id
        
        O_dec_in = self.get_O(L_O, bs, 'decoder')
        
        perms = self.gen_perms(L_ids)
        
        
        attn_mask = self.attn_mask.to(self._device)
        import ipdb; ipdb.set_trace(context=11) # #FF0000
        #* decoding
        logits_dec = []
        agg_dec_ts = []
        for i in range(num_steps):
            j = i + 1 # next token index
            L_dec_in = self.to_L(L_ids, 'decoder')
            select_indices = torch.arange(L_V).tolist() + (L_V + torch.arange(j)).tolist() + [L_V + L_L + i]
            attn_mask_t = attn_mask[select_indices][:, select_indices]
            V_dec_out, L_dec_out, O_dec_out, agg_dec_t = self.decoder(V, L_dec_in[:, :j], O_dec_in[:, i:j], attn_mask_t, debug=debug)
            agg_dec_ts.append(agg_dec_t)
            logits_dec_i = self.char_head_dec(O_dec_out)
            logits_dec.append(logits_dec_i)
            max_time_step = i
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                L_ids[:, j] = logits_dec_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if testing and (L_ids == self.eos_id).any(dim=-1).all():
                    break
        logits_dec = torch.cat(logits_dec, dim=1)
        logits = logits_dec
        
        if debug:
            sa_weights = []
            for t in range(max_time_step + 1):
                _sa_weights = agg_dec_ts[t][DEC_IDX].sa_weights[0]
                sa_weights.append(_sa_weights)
            sa_weights = torch.stack(sa_weights)
            agg_system.sa_weights_dec = sa_weights
        
        #@ refiner
        if self.refiner is not None and self.ref_iters > 0:
            #* sample sequence from decoder
            ids_sampled = self.tokenizer.sample(logits_dec, greedy=True, temp=1.0, pad_to_max_length=False, device=self._device)
            #* prepare embs
            L_S_L = ids_sampled.shape[1]
            L_ref_in = self.to_L(ids_sampled, 'refiner')
            L_S_O = min(ids_sampled.shape[1] + 5, self.max_label_length + 1)
            O_ref_in = self.pos_embed_ref_O[:, :L_S_O].expand(bs, -1, -1)
            O_ref_in = O_ref_in + self.modal_embed[:, 2]
            #* padding mask
            padding_mask_L = (ids_sampled == self.pad_id)
            padding_mask_VLO = F.pad(padding_mask_L, (L_V, L_S_O + 1), "constant", 0)
            #* attention mask
            attn_mask_refine = self.attn_mask_refine.to(self._device)
            select_indices = torch.arange(L_V).tolist() + (L_V + torch.arange(L_S_L)).tolist()\
                +  (L_V +  L_L + torch.arange(L_S_O)).tolist()
            attn_mask_refine_t = attn_mask_refine[select_indices][:, select_indices]
            #* refine
            V_ref_out, L_ref_out, O_ref_out, agg_ref = self.refine(V, L_ref_in, O_ref_in, attn_mask_refine_t, padding_mask_VLO, debug=debug)
            logits_ref = self.char_head_ref(O_ref_out)
            logits = logits_ref
            
            if debug:
                sa_weights = agg_ref[REF_IDX].sa_weights[0].unsqueeze(0)
                agg_system.sa_weights_ref = sa_weights
                
        return logits, logits_dec, agg_system
    
    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        """
        Forward-pass for validation.
        Override function defined in CrossEntropySystem, because initial prediction might be longer than target.
        Have the following properties.
        - Teacher forcing
        - Validation loss computation
        """
        logits, logits_inter, _ = self.forward(images, validation=True)
        # ids = self.tokenizer.encode(labels, self._device)
        # tgt_out = ids[:, 1:]  # Discard [B]
        # L_O = self.max_label_length + 1 # +1 for [E]
        # tgt_out = F.pad(tgt_out, (0, L_O - tgt_out.shape[1]), "constant", self.pad_id)
        # loss = nn.CrossEntropyLoss(ignore_index=self.pad_id)(logits.moveaxis(-1, 1), tgt_out)
        # loss_inter = nn.CrossEntropyLoss(ignore_index=self.pad_id)(logits_inter.moveaxis(-1, 1), tgt_out)
        # loss_numel = (tgt_out != self.pad_id).sum()
        loss = 0
        loss_inter = 0
        loss_numel = 1
        return logits, loss, logits_inter, loss_inter, loss_numel

    def gen_perms(self, L_ids):
        """
        Generate shared permutations for the whole batch.
        This works because the same attention mask can be used for the shorter sequences
        because of the padding mask.
        
        Args:
            L_ids: (bs, L) tensor of label ids. [B] and [E] are included.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = L_ids.shape[1] - 2
        # Special handling for 1-character sequences
        max_num_chars = 3
        # if max_num_chars == 1:
        #     return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        perms.extend([torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
        perms = torch.stack(perms)
        import ipdb; ipdb.set_trace(context=11) # #FF0000
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        # content_mask : Used in content -> content attention.
        # query starts from [B], ends with char_last. key starts from [B], ends with char_last.
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        # query mask : Used in content -> pos attention.
        # query starts from char_first, ends with [E]. key starts from [B], ends with char_last.c
        query_mask = mask[1:, :-1]
        return content_mask, query_mask
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # handle hydra x pytorch lightning bug
        if os.path.exists('./config'):
            shutil.rmtree('./config')
        for log_path in glob.glob('./*.log'):
            if 'ddp' in log_path:
                os.remove(log_path)
        
        images, labels = batch
        bs = images.shape[0]
        
        #* V embs
        V = self.encode(images)
        V = V + self.modal_embed[:, 0]
        L_V = V.shape[1]
        #@ decoding stage.
        #* L embs
        ids = self.tokenizer.encode(labels, self._device)
        L_L = self.max_label_length + 2 # +2 for [B], [E]
        L_O = self.max_label_length + 1 # +1 for [E]
        tgt_in = ids[:, :-1]
        tgt_out = ids[:, 1:]
        L_dec_in = self.to_L(tgt_in, 'decoder')
        #* O embs
        O_dec_in = self.pos_embed_dec_O[:, :L_O].expand(bs, -1, -1)
        O_dec_in = O_dec_in + self.modal_embed[:, 2]
        O_dec_in = O_dec_in[:, :tgt_out.shape[1]]
        #* padding mask
        padding_mask = (tgt_in == self.pad_id)
        padding_mask = F.pad(padding_mask, (L_V, O_dec_in.shape[1]), "constant", 0)
        #* attention mask
        attn_mask = self.attn_mask.to(self._device)
        select_indices = torch.arange(L_V).tolist() + (L_V + torch.arange(tgt_in.shape[1])).tolist()\
            + (L_V + L_L + torch.arange(tgt_out.shape[1])).tolist()
        attn_mask_t = attn_mask[select_indices][:, select_indices]
        #* decoding
        V_dec_out, L_dec_out, O_dec_out, agg_dec = self.decoder(V, L_dec_in, O_dec_in, attn_mask_t, padding_mask)
        logits_dec = self.char_head_dec(O_dec_out)
        loss_dec = nn.CrossEntropyLoss(ignore_index=self.pad_id)(logits_dec.moveaxis(-1, 1), tgt_out)
        probs_dec = logits_dec.softmax(-1)
        preds_dec, probs_dec_trunc = self.tokenizer.decode(probs_dec)
        results_dec = [(a == b) for (a, b) in zip(labels, preds_dec)]
        self.results_dec.extend(results_dec)
        if len(self.results_dec) > 10000:
            train_acc_dec = sum(self.results_dec) / len(self.results_dec) * 100
            self.log('train_acc_dec', train_acc_dec)
            self.results_dec = []
        
        #@ refinement stage.
        if self.refiner is not None:
            #* sample sequence from decoder
            ids_sampled = self.tokenizer.sample(logits_dec, greedy=self.dec_sampling_method == 'identity', temp=self.dec_sampling_temp, pad_to_max_length=False, device=self._device)
            ids_len = ids.shape[1]
            ids_sampled_len = ids_sampled.shape[1]
            L_S_L = max(ids_len, ids_sampled_len)
            if ids_len < L_S_L:
                ids = F.pad(ids, (0, L_S_L - ids_len), "constant", self.pad_id)
            if ids_sampled_len < L_S_L:
                ids_sampled = F.pad(ids_sampled, (0, L_S_L - ids_sampled_len), "constant", self.pad_id)
            assert ids.shape[1] == L_S_L and ids_sampled.shape[1] == L_S_L
            L_S_O = L_S_L - 1
            #* prepare embs
            L_ref_in = self.to_L(ids_sampled, 'refiner')
            O_ref_in = self.pos_embed_ref_O[:, :L_S_O].expand(bs, -1, -1)
            O_ref_in = O_ref_in + self.modal_embed[:, 2]
            #* padding mask
            padding_mask_L = (ids_sampled == self.pad_id)
            padding_mask_VLO = F.pad(padding_mask_L, (L_V, L_S_O), "constant", 0)
            #- mask visual embs with probability
            if torch.rand(1).item() < self.ref_V_masking_prob:
                padding_mask_VLO[:, :L_V] = 1
            #* attention mask
            attn_mask_refine = self.attn_mask_refine.to(self._device)
            select_indices = torch.arange(L_V).tolist() + (L_V + torch.arange(L_S_L)).tolist()\
                + (L_V + L_L + torch.arange(L_S_O)).tolist()
            attn_mask_refine_t = attn_mask_refine[select_indices][:, select_indices]
            #* refiner
            V_ref_out, L_ref_out, O_ref_out, agg_ref = self.refine(V, L_ref_in, O_ref_in, attn_mask_refine_t, padding_mask_VLO)
            #- loss
            #* Language Modeling
            logits_ref_char = self.char_head_ref(O_ref_out)
            tgt_out = ids[:, 1:]
            loss_ref_char = nn.CrossEntropyLoss(ignore_index=self.pad_id)(logits_ref_char.moveaxis(-1, 1), tgt_out)
            #* Replaced Token Detection
            logits_ref_rtd = self.rtd_head_ref(O_ref_out)
            rtd_tgt = (ids_sampled[:, 1:] == tgt_out).float()
            loss_ref_rtd = nn.BCEWithLogitsLoss()(logits_ref_rtd.squeeze().float(), rtd_tgt)
            #* refiner loss
            loss_ref = self.ref_char_loss_scale * loss_ref_char + self.ref_rtd_loss_scale * loss_ref_rtd
            #* total loss
            loss = loss_dec + loss_ref
            #- accuracy
            probs_ref = logits_ref_char.softmax(-1)
            preds_ref, probs_ref_trunc = self.tokenizer.decode(probs_ref)
            results_ref = [(a == b) for (a, b) in zip(labels, preds_ref)]
            self.results_ref.extend(results_ref)
            if len(self.results_ref) > 10000:
                train_acc_ref = sum(self.results_ref) / len(self.results_ref) * 100
                self.log('train_acc_ref', train_acc_ref)
                self.results_ref = []
        else:
            loss_ref = 0
            loss = loss_dec
        
        self.log('loss', loss)
        self.log('loss_ref', loss_ref)
        self.log('loss_dec', loss_dec)
        
        
        return loss
