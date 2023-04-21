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

class VLO(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int, ref_depth: int,
                 dropout: float, QK: List[List[str]], ref_char_loss_scale: int, ref_rtd_loss_scale: int, ref_iters: int,
                 dec_sampling_method: str, dec_sampling_temp : float, ref_objective: str,
                 ref_vis_masking_prob: float,
                 debug: bool = False, **kwargs: Any) -> None:
        """
        Args:
            QK : Specifies allowed attention. "VV" stands for self-attention of visual embs.
                "OV" stands for ordinal embs as query and visual embs as key.
                
                QK = [query_V_list, query_L_list, query_O_list].
                query_V_list = [key_V, key_L, key_O]
                
                e.g. QK = [['V', 'L'], [], ['O']] means that "VV", "VL', "OO" attention is allowed.
                
                Language and ordinal embs are always causal, including self.
        """
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay, debug)
        print('Model : VLO')
        self.results_dec = []
        self.results_ref = []
        self.debug = debug
        self.dec_sampling_method = dec_sampling_method
        self.dec_sampling_temp = dec_sampling_temp
        self.ref_vis_masking_prob = ref_vis_masking_prob
        self.max_label_length = max_label_length
        self.save_hyperparameters()

        # Model
        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio)
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))
        self.refiner = Decoder(decoder_layer, num_layers=ref_depth, norm=nn.LayerNorm(embed_dim)) if ref_depth > 0 else None
        self.ref_iters = ref_iters
        
        # Losses
        self.ref_char_loss_scale = ref_char_loss_scale
        self.ref_rtd_loss_scale = ref_rtd_loss_scale
        
        # Character Embeddings / Heads
        self.char_embed_dec = TokenEmbedding(len(self.tokenizer), embed_dim)
        self.char_embed_ref = GradientDisentangledTokenEmbedding(len(self.tokenizer), embed_dim, self.char_embed_dec)
        self.char_head_dec = nn.Linear(embed_dim, len(self.tokenizer))
        self.char_head_dec.weight = self.char_embed_dec.embedding.weight
        self.char_head_ref = nn.Linear(embed_dim, len(self.tokenizer))
        self.char_head_ref.weight = self.char_embed_ref.embedding.weight
        self.rtd_head_ref = nn.Linear(embed_dim, 1)

        # Positional Embeddings
        self.pos_embed_dec_L = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim)) # +1 for <eos>
        self.pos_embed_dec_O = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.pos_embed_ref_L = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.pos_embed_ref_O = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        
        # Modality Embeddings
        self.modal_embed  = nn.Parameter(torch.Tensor(1, 3, embed_dim))
        
        # Initialization
        named_apply(partial(init_weights, exclude=['encoder']), self) # Encoder has its own init.
        nn.init.trunc_normal_(self.pos_embed_dec_L, std=.02)
        nn.init.trunc_normal_(self.pos_embed_dec_O, std=.02)
        nn.init.trunc_normal_(self.pos_embed_ref_L, std=.02)
        nn.init.trunc_normal_(self.pos_embed_ref_O, std=.02)
        nn.init.trunc_normal_(self.modal_embed, std=.02)
        
        # attn_mask
        am = AttentionMask(max_label_length, QK, self.hparams)
        self.attn_mask = am.get_attn_mask(img_size, patch_size)
        self.attn_mask_refine = am.get_attn_mask(img_size, patch_size, refine_layer=True)
        am.visualize_attn_mask(self.attn_mask)
        am.visualize_attn_mask(self.attn_mask_refine, refine_layer=True)
        self.dummy_emb = torch.zeros((1, 1, embed_dim))
        
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
    
    def encode(self, img: torch.Tensor):
        return self.encoder(img)
    
    def to_lan(self, tgt_in, module):
        """Converts tensor of language token ids to character embeddings."""
        bs, L_L = tgt_in.shape
        if module == 'decoder':
            null_ctx = self.char_embed_dec(tgt_in[:, :1]) # tgt_in stats with [B]. No positional encoding added to [B]
            lan = torch.cat([null_ctx, self.char_embed_dec(tgt_in[:, 1:]) + self.pos_embed_dec_L[:, :L_L - 1]], dim=1)
            lan = lan +  self.modal_embed[:, 1]
        elif module == 'refiner':
            null_ctx = self.char_embed_ref(tgt_in[:, :1]) # tgt_in stats with [B]. No positional encoding added to [B]
            lan = torch.cat([null_ctx, self.char_embed_ref(tgt_in[:, 1:]) + self.pos_embed_dec_L[:, :L_L - 1]], dim=1)
            lan = lan + self.modal_embed[:, 1]
        else:
            raise Exception()
            
        return lan

    def decode(self, vis:torch.Tensor, lan:torch.Tensor,  pos:torch.Tensor, dummy_emb:torch.Tensor,
               attn_mask:torch.Tensor, padding_mask:Optional[Tensor]=None, debug=False):
        """
        Used in forward-pass of train.
        Run Decoder.
        
        Args:
            vis : Visual embs. Shape: N, L_V, D
            lan : Language embs. Shape: N, L_L, D
            pos : Ordinal embs. Shape: N, L_O, D
        """
        lan = self.dropout(lan)
        pos = self.dropout(pos)
        dummy_emb = dummy_emb.expand(pos.shape[0], -1, -1)
        return self.decoder(vis, lan, pos, dummy_emb, attn_mask=attn_mask, padding_mask=padding_mask, debug=debug)
    
    def refine(self, vis:torch.Tensor, lan:torch.Tensor,  pos:torch.Tensor, dummy_emb:torch.Tensor,
               attn_mask:torch.Tensor, padding_mask:Optional[Tensor]=None, debug=False):
        """
        Used in forward-pass of train.
        Further refines initial decoder prediction.
        Stop gradient is applied to language and positional embs,
        to prevent information leak from future steps.
        
        Args:
            vis : Visual embs. Shape: N, L_V, D
            lan : Language embs. Shape: N, L_L, D
            pos : Ordinal embs. Shape: N, L_O, D
        """
        lan = self.dropout(lan)
        pos = self.dropout(pos)
        dummy_emb = dummy_emb.expand(pos.shape[0], -1, -1)
        # vis is 
        return self.refiner(vis.detach(), lan.detach(), pos.detach(), dummy_emb, attn_mask=attn_mask, padding_mask=padding_mask, debug=debug)
 
    def forward(self, images:Tensor, validation: bool = False, debug: bool = False, DEC_IDX=0, REF_IDX=0) -> Tensor:
        """
        Forward-pass for test & val.
        Used implicitly in forward-pass of val.
        
        Args:
            images : input image
            return_intermediate_logits : In case of decoder-refiner structure, also return decoder logits.
            validation : Is validation step.
            debug : Is debug mode.
            DEC_IDX : Debugging target decoder index.
            REF_IDX : Debugging target refiner index.
        """
        if debug :
            agg_system = System_Data()
        else:
            agg_system = None
        
        testing = not validation
        bs = images.shape[0]
        L_L = self.max_label_length + 2 # +2 for [B], [E]
        L_O = num_steps = self.max_label_length + 1 # +1 for [E]
        
        #@ decoder
        #* prepare embs
        vis = self.encode(images)
        vis = vis + self.modal_embed[:, 0]
        L_V = vis.shape[1]
        lan_ids = torch.full((bs, L_L), self.pad_id, dtype=torch.long, device=self._device)
        lan_ids[:, 0] = self.bos_id
        ord_dec_in = self.pos_embed_dec_O[:, :L_O].expand(bs, -1, -1)
        ord_dec_in = ord_dec_in + self.modal_embed[:, 2]
        dummy_emb = self.dummy_emb.to(self._device)
        attn_mask = self.attn_mask.to(self._device)
        #* decoding
        logits_dec = []
        agg_dec_ts = []
        for i in range(num_steps):
            j = i + 1 # next token index
            lan_dec_in = self.to_lan(lan_ids, 'decoder')
            select_indices = torch.arange(L_V).tolist() + (L_V + torch.arange(j)).tolist() + [L_V + L_L + i] + [-1]
            attn_mask_t = attn_mask[select_indices][:, select_indices]
            vis_dec_out, lan_dec_out, ord_dec_out, agg_dec_t = self.decode(vis, lan_dec_in[:, :j], ord_dec_in[:, i:j], dummy_emb, attn_mask=attn_mask_t, debug=debug)
            agg_dec_ts.append(agg_dec_t)
            logits_dec_i = self.char_head_dec(ord_dec_out)
            logits_dec.append(logits_dec_i)
            max_time_step = i
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                lan_ids[:, j] = logits_dec_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if testing and (lan_ids == self.eos_id).any(dim=-1).all():
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
            lan_ref_in = self.to_lan(ids_sampled, 'refiner')
            L_S_O = min(ids_sampled.shape[1] + 5, self.max_label_length + 1)
            ord_ref_in = self.pos_embed_ref_O[:, :L_S_O].expand(bs, -1, -1)
            ord_ref_in = ord_ref_in + self.modal_embed[:, 2]
            #* padding mask
            padding_mask_L = (ids_sampled == self.pad_id)
            padding_mask_VLO = F.pad(padding_mask_L, (L_V, L_S_O + 1), "constant", 0)
            #* attention mask
            attn_mask_refine = self.attn_mask_refine.to(self._device)
            select_indices = torch.arange(L_V).tolist() + (L_V + torch.arange(L_S_L)).tolist()\
                +  (L_V +  L_L + torch.arange(L_S_O)).tolist() + [-1]
            attn_mask_refine_t = attn_mask_refine[select_indices][:, select_indices]
            #* refine
            vis_ref_out, lan_ref_out, ord_ref_out, agg_ref = self.refine(vis, lan_ref_in, ord_ref_in, dummy_emb, attn_mask_refine_t, padding_mask_VLO, debug=debug)
            logits_ref = self.char_head_ref(ord_ref_out)
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

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # handle hydra x pytorch lightning bug
        if os.path.exists('./config'):
            shutil.rmtree('./config')
        for log_path in glob.glob('./*.log'):
            if 'ddp' in log_path:
                os.remove(log_path)
        
        images, labels = batch
        bs = images.shape[0]
        
        #* vis embs
        vis = self.encode(images)
        vis = vis + self.modal_embed[:, 0]
        L_V = vis.shape[1]
        #@ decoding stage.
        #* lan embs
        ids = self.tokenizer.encode(labels, self._device)
        L_L = self.max_label_length + 2 # +2 for [B], [E]
        L_O = self.max_label_length + 1 # +1 for [E]
        tgt_in = ids[:, :-1]
        tgt_out = ids[:, 1:]
        lan_dec_in = self.to_lan(tgt_in, 'decoder')
        #* ord embs
        ord_dec_in = self.pos_embed_dec_O[:, :L_O].expand(bs, -1, -1)
        ord_dec_in = ord_dec_in + self.modal_embed[:, 2]
        ord_dec_in = ord_dec_in[:, :tgt_out.shape[1]]
        #* padding mask
        padding_mask = (tgt_in == self.pad_id)
        padding_mask = F.pad(padding_mask, (L_V, ord_dec_in.shape[1] + 1), "constant", 0) # +1 for dummy token
        #* dummy emb
        dummy_emb = self.dummy_emb.to(self._device)
        #* attention mask
        attn_mask = self.attn_mask.to(self._device)
        select_indices = torch.arange(L_V).tolist() + (L_V + torch.arange(tgt_in.shape[1])).tolist()\
            + (L_V + L_L + torch.arange(tgt_out.shape[1])).tolist() + [-1]
        attn_mask_t = attn_mask[select_indices][:, select_indices]
        #* decoding
        vis_dec_out, lan_dec_out, ord_dec_out, agg_dec = self.decode(vis, lan_dec_in, ord_dec_in, dummy_emb, attn_mask_t, padding_mask)
        logits_dec = self.char_head_dec(ord_dec_out)
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
            lan_ref_in = self.to_lan(ids_sampled, 'refiner')
            ord_ref_in = self.pos_embed_ref_O[:, :L_S_O].expand(bs, -1, -1)
            ord_ref_in = ord_ref_in + self.modal_embed[:, 2]
            #* padding mask
            padding_mask_L = (ids_sampled == self.pad_id)
            padding_mask_VLO = F.pad(padding_mask_L, (L_V, L_S_O + 1), "constant", 0)
            #- mask visual embs with probability
            if torch.rand(1).item() < self.ref_vis_masking_prob:
                padding_mask_VLO[:, :L_V] = 1
            #* attention mask
            attn_mask_refine = self.attn_mask_refine.to(self._device)
            select_indices = torch.arange(L_V).tolist() + (L_V + torch.arange(L_S_L)).tolist()\
                + (L_V + L_L + torch.arange(L_S_O)).tolist() + [-1]
            attn_mask_refine_t = attn_mask_refine[select_indices][:, select_indices]
            #* refiner
            vis_ref_out, lan_ref_out, ord_ref_out, agg_ref = self.refine(vis, lan_ref_in, ord_ref_in, dummy_emb, attn_mask_refine_t, padding_mask_VLO)
            #- loss
            #* Language Modeling
            logits_ref_char = self.char_head_ref(ord_ref_out)
            tgt_out = ids[:, 1:]
            loss_ref_char = nn.CrossEntropyLoss(ignore_index=self.pad_id)(logits_ref_char.moveaxis(-1, 1), tgt_out)
            #* Replaced Token Detection
            logits_ref_rtd = self.rtd_head_ref(ord_ref_out)
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
