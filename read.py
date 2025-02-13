#!/usr/bin/env python3
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

import argparse
import os
import glob
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import normalize
from scipy.special import softmax

import torch
from hydra.utils import instantiate
from hydra import compose, initialize
from omegaconf import open_dict

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import parse_model_args, init_dir

import warnings
warnings.filterwarnings('ignore')


@torch.inference_mode()
def main():
    def str2bool(x):
        return x.lower() == 'true'
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--image_dir', default='./demo_images', help='Directory of input images')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--debug', type=str2bool, default=True, help='Debugging mode (visualization)')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')
    ckpt_split = args.checkpoint.split('/')
    exp_dir = '/'.join(ckpt_split[:ckpt_split.index('checkpoints')])
    initialize(config_path=f'{exp_dir}/config', version_base='1.2')
    cfg = compose(config_name='config')
    with open_dict(cfg):
        cfg.model.debug = args.debug
    for k, v in kwargs.items():
        setattr(cfg.model, k, v)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    torch.cuda.set_device(int(args.gpu))
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cuda')['state_dict'])
    model.eval().cuda()
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    hparams = model.hparams
    
    debug_dir = f'{exp_dir}/debug'
    init_dir(f'{debug_dir}/read')
    
    image_paths = sorted(glob.glob(f'{args.image_dir}/*'))
    
    for image_path in image_paths:
        basename = os.path.basename(image_path)
        image_save_path = f'{debug_dir}/read/{basename}' # save input image
        
        # Load image and prepare for input
        image = Image.open(image_path).convert('RGB')
        image.save(image_save_path)
        image_t = img_transform(image).unsqueeze(0).cuda()

        # logits, logits_inter, agg = model(image_t, debug=args.debug, DEC_IDX=0, REF_IDX=0)
        logits, logits_inter, agg = model(image_t, debug=args.debug)
        dist = logits.softmax(-1)
        dist_inter = logits_inter.softmax(-1)
        pred, prob_seq = model.tokenizer.decode(dist)
        pred_inter, prob_seq_inter = model.tokenizer.decode(dist_inter)
        '''
        Uncomment lines for desired visualization.
        '''
        # visualize_char_embed_self_sim(model, image_save_path)
        
        #- decoder
        visualize_char_probs(pred_inter, dist_inter, model, image_save_path, 'dec')
        # visualize_attn_balance(pred, pred_inter, agg.sa_weights_dec, hparams, image_save_path, module='decoder')
        # visualize_self_attn_VLO(pred, pred_inter, agg.sa_weights_dec, hparams, image, image_save_path, Q='VLO', K='VLO', module='decoder')
        # visualize_self_attn_VLO(pred, pred_inter, agg.sa_weights_dec, hparams, image, image_save_path, Q='O', K='V', module='decoder')
        # visualize_self_attn_VLO(pred, pred_inter, agg.sa_weights_dec, hparams, image, image_save_path, Q='O', K='L', module='decoder')
        # visualize_self_attn_VLO(pred, pred_inter, agg.sa_weights_dec, hparams, image, image_save_path, Q='O', K='O', module='decoder')
        # visualize_self_attn_VLO(pred, pred_inter, agg.sa_weights_dec, hparams, image, image_save_path, Q='L', K='L', module='decoder')
        # #- refiner
        # visualize_char_probs(pred, dist, model, image_save_path, 'ref')
        # visualize_attn_balance(pred, pred_inter, agg.sa_weights_ref, hparams, image_save_path, module='refiner')
        # visualize_self_attn_VLO(pred, pred_inter, agg.sa_weights_ref, hparams, image, image_save_path, Q='VLO', K='VLO', module='refiner')
        # visualize_self_attn_VLO(pred, pred_inter, agg.sa_weights_ref, hparams, image, image_save_path, Q='O', K='V', module='refiner')
        # visualize_self_attn_VLO(pred, pred_inter, agg.sa_weights_ref, hparams, image, image_save_path, Q='O', K='L', module='refiner')
        # visualize_self_attn_VLO(pred, pred_inter, agg.sa_weights_ref, hparams, image, image_save_path, Q='O', K='O', module='refiner')
        
        print(f'{basename}: {pred[0]}')


def save_heatmap(data, rows, cols, title, save_path, sim_scale,
                 figsize=(15, 15), dpi=96, vmin=0, vmax=1,
                 annot=False, annot_size=10, tick_size=None,
                 linewidths=0, labelsize=None, x_rot=0, y_rot=0,
                 rects=None):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data *= sim_scale
    df = pd.DataFrame(data, index=rows, columns=cols)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title)
    ax = plt.gca()
    # ax_pos = [0.15, 0.01, 0.84, 0.84]
    # ax.set_position(ax_pos)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    sa = sns.heatmap(df,
                    vmin=vmin,
                    vmax=vmax,
                    annot=annot,
                    fmt='.2f',
                    annot_kws={'size': annot_size},
                    ax=ax,
                    cbar_ax=cax,
                    cbar=True,
                    linewidths=linewidths,
                    )
    cbar = sa.collections[0].colorbar
    if labelsize is not None:
        cbar.ax.tick_params(labelsize=labelsize)
    if rects is not None:
        for rect in rects:
            sa.add_patch(rect)
    sa.xaxis.tick_top()
    if tick_size is not None:
        sa.set_xticklabels(sa.get_xmajorticklabels(), fontsize=tick_size, rotation=x_rot)
        sa.set_yticklabels(sa.get_ymajorticklabels(), fontsize=tick_size, rotation=y_rot)
    else:
        sa.set_xticklabels(sa.get_xmajorticklabels(), rotation=x_rot)
        sa.set_yticklabels(sa.get_ymajorticklabels(), rotation=y_rot)
    plt.savefig(save_path)
    plt.close(fig)
    

def save_blended_heatmap(data, image, save_path, alpha=0.7):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    cm = plt.get_cmap('jet')
    attn = data
    attn = (attn - attn.min()) / (attn.max() - attn.min())
    attn = np.clip(attn, 0.0, 1.0)
    attn = cm(attn)
    attn = Image.fromarray((attn * 255).astype(np.uint8)).convert('RGB')
    attn = attn.resize(image.size)
    blend = Image.blend(image, attn, alpha=alpha)
    blend.save(save_path)


def visualize_attn_balance(pred, pred_inter, sa_weights, hparams, image_save_path, module='refiner', sim_scale=1.0):
    """
    Visualize ratio between V, L, O tokens attention values summed.
    """
    if sa_weights is None: return
    assert module in ['decoder', 'refiner']
    if module == 'decoder':
        tag = 'dec'
        pred = pred_inter
    elif module == 'refiner':
        tag = 'ref'
        pred = pred
    else:
        raise Exception()
    filename_path, ext = os.path.splitext(image_save_path)
    pred = list(pred[0])
    vis_size = [a // b for (a, b) in zip(hparams.img_size, hparams.patch_size)]
    L_V = vis_size[0] * vis_size[1]
    L_L = L_O = hparams.max_label_length + 1
    L_T = L_V + L_L + L_O
    assert sa_weights.shape[-1] == L_T + 1 # +1 for dummy token
    rows = pred + ['[E]']
    cols = ['V attn', 'L attn', 'O attn']
    sa_weights_last = sa_weights[-1]
    V_attn, L_attn, O_attn, _ = sa_weights_last.split([L_V, L_L, L_O, 1], dim=-1)
    V_attn = V_attn.sum(dim=-1, keepdim=True)
    L_attn = L_attn.sum(dim=-1, keepdim=True)
    O_attn = O_attn.sum(dim=-1, keepdim=True)
    attn_balance = torch.cat([V_attn, L_attn, O_attn], dim=-1)
    save_heatmap(attn_balance[L_V + L_L:L_V + L_L + len(rows), :], rows , cols, f'Attention balance', f'{filename_path}_{tag}_attn_balance{ext}',
                 sim_scale=sim_scale, figsize=(7.5, 15), annot=True, annot_size=20, tick_size=15)
    

def visualize_self_attn_VLO(pred, pred_inter, sa_weights, hparams, image, image_save_path, module='refiner', Q='VLO', K='VLO', sim_scale=1.0):
    """
    Self-attn visualization of multi-modal Transformer.
    
    Args:
        Q : Query. e.g. 'V', 'L', 'O'
        K : Key. e.g. 'V', 'L', 'O'
    """
    if sa_weights is None: return
    assert module in ['decoder', 'refiner']
    filename_path, ext = os.path.splitext(image_save_path)
    pred = list(pred[0])
    pred_inter = list(pred_inter[0])
    if module == 'decoder':
        tag = 'dec'
        row_pred = pred_inter
        col_pred = pred_inter
    elif module == 'refiner':
        tag = 'ref'
        row_pred = pred
        col_pred = pred_inter
    else:
        raise Exception()
    vis_size = [a // b for (a, b) in zip(hparams.img_size, hparams.patch_size)]
    L_V = vis_size[0] * vis_size[1]
    L_L = L_O = hparams.max_label_length + 1
    L_T = L_V + L_L + L_O
    assert sa_weights.shape[-1] == L_T + 1 # +1 for dummy token
    rows = list(range(L_T + 1))
    for i in range(L_L + L_O):
        rows[L_V + i] = '[P]'
    for i in range(len(row_pred) + 1):
        rows[L_V + i] = (['[B]'] + row_pred)[i]
    for i in range (len(row_pred) + 1):
        rows[L_V + L_L + i] = (row_pred + ['[E]'])[i]
    rows[-1] = '[D]'
    rows_V = rows[:L_V]
    rows_L = rows[L_V:L_V + L_L]
    rows_O = rows[L_V + L_L:L_V + L_L + L_O]
    cols = list(range(L_T + 1))
    for i in range(L_L + L_O):
        cols[L_V + i] = '[P]'
    for i in range(len(col_pred) + 1):
        cols[L_V + i] = (['[B]'] + col_pred)[i]
    for i in range (len(col_pred) + 1):
        cols[L_V + L_L + i] = (col_pred + ['[E]'])[i]
    cols[-1] = '[D]'
    cols_V = cols[:L_V]
    cols_L = cols[L_V:L_V + L_L]
    cols_O = cols[L_V + L_L:L_V + L_L + L_O]
    V_ind = list(range(L_V))
    L_ind = list(range(L_V, L_V + L_L))
    O_ind = list(range(L_V + L_L, L_V + L_L + L_O))
    rows, cols, row_ind, col_ind = [], [], [], []
    if 'V' in Q:
        rows.extend(rows_V)
        row_ind.extend(V_ind)
    if 'L' in Q:
        rows.extend(rows_L)
        row_ind.extend(L_ind)
    if 'O' in Q:
        rows.extend(rows_O)
        row_ind.extend(O_ind)
    if 'V' in K:
        cols.extend(cols_V)
        col_ind.extend(V_ind)
    if 'L' in K:
        cols.extend(cols_L)
        col_ind.extend(L_ind)
    if 'O' in K:
        cols.extend(cols_O)
        col_ind.extend(O_ind)
    if Q == K == 'VLO':
        rects = []
        for x, y, w, h in [(0, 0, L_V, L_V), (L_V, 0, L_L, L_V), (L_V + L_L, 0, L_O, L_V),
         (0, L_V, L_V, L_L), (L_V, L_V, L_L, L_L), (L_V + L_L, L_V, L_O, L_L),
         (0, L_V + L_L, L_V, L_O), (L_V, L_V + L_L, L_L, L_O), (L_V + L_L, L_V + L_L, L_O, L_O)]:
            rects.append(patches.Rectangle((x, y,), w, h, edgecolor='green', facecolor='none'))
        t = len(pred)
        for x, y, w, h in [(L_V, L_V, t + 1, t + 1), (L_V + L_L, L_V, t + 1, t + 1), (L_V, L_V + L_L, t + 1, t + 1), (L_V + L_L, L_V + L_L, t + 1, t + 1)]:
            rects.append(patches.Rectangle((x, y,), w, h, edgecolor='white', facecolor='none', linewidth=0.7))
        save_heatmap(sa_weights[-1][row_ind, :][:, col_ind], rows, cols, f'{Q}-{K}', f'{filename_path}_{tag}_sa_{Q}_{K}{ext}', sim_scale, rects=rects)
    elif Q + K in ['LL', 'LO', 'OL', 'OO']:
        if 'dec' in tag:
            for t, sa_weights_t in enumerate(sa_weights):
                sa_weights_t = sa_weights_t[row_ind, :][:, col_ind].detach().cpu().numpy()
                rects = [patches.Rectangle((0, 0,), t + 1, t + 1, edgecolor='white', facecolor='none')]
                save_heatmap(sa_weights_t, rows, cols, f'{Q}-{K}', f'{filename_path}_{tag}_sa_{Q}_{K}_{t:02d}{ext}', sim_scale, rects=rects, annot=True)
        elif 'ref' in tag:
            t_h = len(row_pred)
            t_w = len(col_pred)
            sa_weights_t = sa_weights[0][row_ind, :][:, col_ind].detach().cpu().numpy()
            rects = [patches.Rectangle((0, 0,), t_w + 1, t_h + 1, edgecolor='white', facecolor='none')]
            save_heatmap(sa_weights_t, rows, cols, f'{Q}-{K}', f'{filename_path}_{tag}_sa_{Q}_{K}{ext}', sim_scale, rects=rects, annot=True)
        else:
            raise NotImplementedError
    elif Q + K in ['OV', 'LV']:
        if 'dec' in tag:
            for t, sa_weights_t in enumerate(sa_weights):
                save_path = f'{filename_path}_{tag}_sa_{Q}_{K}_{t:02d}{ext}'
                sa_weights_t = sa_weights_t[row_ind, :][:, col_ind]
                sa_weights_t = sa_weights_t[t]
                sa_weights_t = sa_weights_t.view(*vis_size)
                save_blended_heatmap(sa_weights_t, image, save_path)
        elif 'ref' in tag:
            for t, sa_weights_t in enumerate(sa_weights[0][row_ind, :][:, col_ind]):
                if t > len(pred): continue
                save_path = f'{filename_path}_{tag}_sa_{Q}_{K}_{t:02d}{ext}'
                sa_weights_t = sa_weights_t.view(*vis_size)
                save_blended_heatmap(sa_weights_t, image, save_path)
        else:
            raise NotImplementedError
    elif Q + K in ['VV']:
        sa_weights = sa_weights[-1]
        sa_weights = sa_weights[row_ind, :][:, col_ind]
        for pix in range(sa_weights.shape[0]):
            save_path = f'{filename_path}_{tag}_sa_{Q}_{K}_{pix:02d}{ext}'
            sa_weights_t = sa_weights[pix]
            sa_weights_t = sa_weights_t.view(*vis_size)
            attn = sa_weights_t.detach().cpu().numpy()
            attn = (attn - attn.min()) / (attn.max() - attn.min())
            attn = np.clip(attn, 0.0, 1.0)
            cm = plt.get_cmap('jet')
            attn = cm(attn)
            attn[pix // vis_size[1], pix % vis_size[1]] = [1, 0, 1, 1] # query pixel is magenta
            attn = Image.fromarray((attn * 255).astype(np.uint8)).convert('RGB')
            attn = attn.resize(image.size, resample=Image.NEAREST)
            blend = Image.blend(image, attn, alpha=0.5)
            blend.save(save_path)
    elif Q + K in ['VO', 'VL']:
        raise NotImplementedError
    else:
        raise NotImplementedError
        

def visualize_char_probs(pred, dist, model, image_save_path, tag):
    filename_path, ext = os.path.splitext(image_save_path)
    rows = pred = list(pred[0]) + ['[E]']
    dist = dist[0].detach().cpu().numpy()[:len(pred), :] # probs up to [E], [seq_len + 1, len(charset_train) - 2]
    charset_train = model.hparams.charset_train
    try:
        cols = ['[E]'] + list(charset_train) + ['[B]', '[P]']
        save_path = f'{filename_path}_{tag}_char_probs{ext}'
        save_heatmap(dist, rows, cols, '', save_path, 1.0, figsize=(30, len(rows)), annot=True, annot_size=5, tick_size=15, labelsize=15, linewidths=1)
    except:
        cols = ['[E]'] + list(charset_train)
        save_path = f'{filename_path}_{tag}_char_probs{ext}'
        save_heatmap(dist, rows, cols, '', save_path, 1.0, figsize=(30, len(rows)), annot=True, annot_size=5, tick_size=15, labelsize=15, linewidths=1)

def visualize_char_embed_self_sim(model, image_save_path, sim_scale=1.0):
    emb = model.text_embed.embedding.weight.detach().cpu().numpy()
    charset_train = model.hparams.charset_train
    rows = cols = ['[E]'] + list(charset_train) + ['[B]', '[P]']
    visualize_similarity(emb, emb, rows, cols, image_save_path, sim_scale, annot=False, figsize=(30,30))

def visualize_similarity(target, source, rows, cols, image_save_path, sim_scale=1.0, annot=False, tag='', figsize=(15,15)):
    filename_path, ext = os.path.splitext(image_save_path)
    target = normalize(target)
    source = normalize(source)
    similarity_mtx = target @  source.T
    save_heatmap(similarity_mtx, rows, cols, '', f'{filename_path}_sim{tag}{ext}', sim_scale, annot=annot, annot_size=10, tick_size=10, labelsize=10, figsize=figsize)

'''

 

def visualize_head_self_sim(model, image_save_path):
    head = model.head.weight.detach().cpu().numpy()
    charset_train = model.hparams.charset_train
    rows = cols = (['[E]'] + list(charset_train) + ['[B]', '[P]'])[:head.shape[0]]
    visualize_similarity(head, head, rows, cols, image_save_path) 


def visualize_sim_with_head(attr, agg, pred, model, image_save_path, sim_scale=1.0):
    head = model.head.weight.detach().cpu().numpy()
    charset_train = model.hparams.charset_train
    if len(head) == 95:
        cols = ['[E]'] + list(charset_train)
    else:
        cols = ['[E]'] + list(charset_train) + ['[B]', '[P]']
    if attr == 'content':
        rows = ['[B]'] + list(pred[0])
    else:
        rows = list(pred[0]) + ['[E]']
    target = getattr(agg, attr)
    target = target.detach().cpu().numpy()[0]
    visualize_similarity(target, head, rows, cols, image_save_path, sim_scale=sim_scale, tag='_' + attr)


def visualize_pos_embed_self_sim(pred, model, image_save_path, sim_scale=1.0):
    pred = list(pred[0]) + ['[E]']
    emb = model.pos_embed_O.detach().cpu().numpy()[0][:len(pred), :]
    rows = cols = list(range(1, len(pred) + 1))
    visualize_similarity(emb, emb, rows, cols, image_save_path, sim_scale, annot=True)
    
    
def visualize_sim_with_pe(target, pred, model, image_save_path, sim_scale=1.0):
    rows = pred = list(pred[0]) + ['[E]']
    pos_queries = model.pos_queries.detach().cpu().numpy()[0][:len(pred), :]
    target = target.detach().cpu().numpy()[0][:len(pred), :]
    cols = list(range(1, len(pred) + 1))
    visualize_similarity(pos_queries, pos_queries, rows, cols, image_save_path, sim_scale, annot=True)
    
    
def visualize_char_embed_sim_with_head(model, image_save_path): 
    text_embed = model.text_embed.embedding.weight.detach().cpu().numpy() # [charset_size, embed_dim]
    head = model.head.weight.detach().cpu().numpy()
    charset_train = model.hparams.charset_train
    rows = cols = (['[E]'] + list(charset_train) + ['[B]', '[P]'])[:head.shape[0]]
    visualize_similarity(text_embed, head, rows, cols, image_save_path)
            
   
def visualize_cross_attn(ca_weights, hparams, image, image_save_path, tag=''):
    filename_path, ext = os.path.splitext(image_save_path)
    if ca_weights is None: return
    vis_size = [a // b for (a, b) in zip(hparams.img_size, hparams.patch_size)]
    ca_weights = ca_weights.view(-1, vis_size[0], vis_size[1])
    ca_weights = ca_weights.detach().cpu().numpy()
    
    for i, attn in enumerate(ca_weights):
        save_path = f'{filename_path}_ca{tag}_{i:02d}{ext}'
        save_blended_heatmap(attn, image, save_path, alpha=0.8)
    

def visualize_sim_with_memory(target, memory, image, image_save_path):
    filename_path, ext = os.path.splitext(image_save_path)
    cm = plt.get_cmap('jet')
    memory = memory.view(-1, 384).detach().cpu().numpy()
    target = target.view(-1, 384).detach().cpu().numpy()
    seq_sim_mtx = target @ memory.T
    for i, sim_mtx in enumerate(seq_sim_mtx):
        save_path = f'{filename_path}_sm_{i:02d}{ext}'
        attn = softmax(sim_mtx)
        save_blended_heatmap(attn, image, save_path)
'''

if __name__ == '__main__':
    main()
