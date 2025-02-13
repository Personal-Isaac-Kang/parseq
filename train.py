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
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    # config
    trainer_strategy = None
    with open_dict(config):
        config.data.root_dir = os.path.abspath(config.data.root_dir)
        gpus = config.trainer.get('gpus', 0)
        if isinstance(gpus, int):
            num_gpus = gpus
        else:
            num_gpus = len(gpus)
        if num_gpus > 1:
            config.trainer.strategy = 'ddp'
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            config.trainer.val_check_interval //= num_gpus
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= num_gpus

    if config.model.get('perm_num') is not None:
        if config.model.perm_num == 1:
            config.model.perm_mirrored = False
            config.model.refine_iters = 0
    
    # model
    model: BaseSystem = hydra.utils.instantiate(config.model)
    print(model.hparams)
    print(summarize(model, max_depth=1))

    # data
    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    # trainer
    checkpoint = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=3, save_last=True,
                                 filename='{epoch}-{step}-{val_accuracy:.4f}')
    swa = StochasticWeightAveraging(swa_epoch_start=0.75)
    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else str(Path(config.ckpt_path).parents[1].absolute())
    lr_monitor = LearningRateMonitor()
    print(f'checkpoint directory : {cwd}')
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(cwd, '', '.'),
                                               strategy=trainer_strategy, enable_model_summary=False,
                                               callbacks=[checkpoint, swa, lr_monitor])
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()