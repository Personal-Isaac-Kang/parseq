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
import string
import sys
import os
from dataclasses import dataclass
from typing import List
from hydra.utils import instantiate
from hydra import compose, initialize
from omegaconf import open_dict

import torch

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import parse_model_args, init_dir


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


@torch.inference_mode()
def main():
    def str2bool(x):
        return x.lower() == 'true'
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='../data/parseq/english/lmdb')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', type=str2bool, default=False, help='Cased comparison')
    parser.add_argument('--punctuation', type=str2bool, default=False, help='Check punctuation')
    parser.add_argument('--new', type=str2bool, default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--debug', type=str2bool, default=False, help='Run in debugging mode (visualization)')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    torch.cuda.set_device(args.gpu)
    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
        
    print(f'Additional keyword arguments: {kwargs}')
    ckpt_split = args.checkpoint.split('/')
    exp_dir = '/'.join(ckpt_split[:ckpt_split.index('checkpoints')])
    initialize(config_path=f'{exp_dir}/config', version_base='1.2')
    cfg = compose(config_name='config')
    with open_dict(cfg):
        cfg.model.debug = args.debug
    if cfg.model.get('perm_num') is not None:
        if cfg.model.perm_num == 1:
            if kwargs.get('refine_iters') is None:
                cfg.model.refine_iters = 0
            if kwargs.get('perm_mirrored') is None:
                cfg.model.perm_mirrored = False
    for k, v in kwargs.items():
        setattr(cfg.model, k, v)
    model = instantiate(cfg.model)
    hp = model.hparams
    print(model.hparams)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cuda')['state_dict'])
    model.eval().cuda()
    
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation, debug=args.debug)

    # test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    # if args.new:
    #     test_set += SceneTextDataModule.TEST_NEW
    # test_set = sorted(set(test_set))
    
    # #00FFFF : temporary
    test_set = SceneTextDataModule.TEST_TRAIN
    test_set = sorted(set(test_set))

    results = {}
    pred_gts = {}
    max_width = max(map(len, test_set))
    debug_dir = f'{exp_dir}/debug'
    for dname, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        preds = []
        preds_inter = []
        gts = []
        if args.debug:
            init_dir(f'{debug_dir}/images/{dname}')
            for imgs, labels, img_keys, img_origs in tqdm(iter(dataloader), desc=f'{dname:>{max_width}}'):
                result = model.test_step((imgs.to(model.device), labels, img_keys, img_origs), False, debug_dir, dname)
                preds.extend(result['preds'])
                preds_inter.extend(result['preds_inter'])
                res = result['output']
                gts.extend(labels)
                total += res.num_samples
                correct += res.correct
                ned += res.ned
                confidence += res.confidence
                label_length += res.label_length
        else:
            for imgs, labels in tqdm(iter(dataloader), desc=f'{dname:>{max_width}}'):
                res = model.test_step((imgs.to(model.device), labels), False)['output']
                total += res.num_samples
                correct += res.correct
                ned += res.ned
                confidence += res.confidence
                label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[dname] = Result(dname, total, accuracy, mean_ned, mean_conf, mean_label_length)
        if args.debug:
            init_dir(f'{debug_dir}/texts/{dname}')
            with open(f'{debug_dir}/texts/{dname}/results.txt', 'w') as f:
                f.write(f'{"pred_inter":26s}{"pred":26s}{"gt":26s}{"refined":10s}{"correct":10s}\n')
                f.write(f'{"----------":26s}{"----":26s}{"--":26s}{"------":10s}{"------":10s}\n')
                for pred_inter, pred, gt in zip(preds_inter, preds, gts):
                    refined = 'o' if pred != pred_inter else ''
                    correct = 'x' if pred != gt else ' '
                    f.write(f'{pred_inter:26s}{pred:26s}{gt:26s}{refined:10s}{correct:10s}\n')

    # result_groups = {
    #     'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB,
    #     'Benchmark': SceneTextDataModule.TEST_BENCHMARK
    # }
    # if args.new:
    #     result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    
    # #00FFFF : temporary
    result_groups = {
        'Train' : SceneTextDataModule.TEST_TRAIN
    }
    
    log_tag = ''
    for k, v in kwargs.items():
        if k == 'charset_test':
            continue
        else:
            log_tag += '.' + str(k) + '=' + str(v)
            
    with open(args.checkpoint + log_tag + '.table.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)



if __name__ == '__main__':
    main()
