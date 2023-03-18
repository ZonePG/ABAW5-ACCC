import os
import pathlib

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary, StochasticWeightAveraging, LearningRateMonitor

from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger

from model import config, ABAW5Model, ABAW5DataModule
from model.config import cfg
from model.io import pathmgr

from datetime import datetime
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    config.load_cfg_fom_args("ABAW5")
    config.assert_and_infer_cfg()
    cfg.freeze()

    pl.seed_everything(cfg.RNG_SEED)

    pathmgr.mkdirs(cfg.OUT_DIR)
    run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg_file_dir = pathlib.Path(cfg.OUT_DIR, cfg.TASK, run_version)
    pathmgr.mkdirs(cfg_file_dir)
    cfg_file = config.dump_cfg(cfg_file_dir)

    # raise ValueError('Do not implement with {} logger yet.'.format(cfg.LOGGER))
    logger = TensorBoardLogger(cfg.OUT_DIR, name=cfg.TASK, version=run_version)
    output_dir = logger.log_dir

    if cfg.TEST.WEIGHTS != '':
        result_dir = '/'.join(cfg.TEST.WEIGHTS.split('/')[:-1])
    else:
        result_dir = ''

    print('Working on Task: ', cfg.TASK)
    max_epochs = cfg.OPTIM.MAX_EPOCH if cfg.TEST.WEIGHTS == '' else 1

    ABAW5_dataset = ABAW5DataModule()
    ABAW5_model = ABAW5Model()

    fast_dev_run = False
    richProgressBarTheme = RichProgressBarTheme(description="blue", progress_bar="green1",
                                                progress_bar_finished="green1")

    # backbone_finetunne = MultiStageABAW5(unfreeze_temporal_at_epoch=3, temporal_initial_ratio_lr=0.1,
    #                                         should_align=True, initial_denom_lr=10, train_bn=True)
    ckpt_cb = ModelCheckpoint(monitor='val_metric', mode="max", save_top_k=1, save_last=True)
    trainer_callbacks = [ckpt_cb,
                         LearningRateMonitor(logging_interval=None)
                         ]
    if cfg.LOGGER in ['TensorBoard', ] and not cfg.OPTIM.TUNE_LR:
        trainer_callbacks.append(RichProgressBar(refresh_rate_per_second=1, theme=richProgressBarTheme, leave=True))
        trainer_callbacks.append(RichModelSummary())

    if cfg.OPTIM.USE_SWA:
        swa_callbacks = StochasticWeightAveraging(swa_epoch_start=0.8,
                                                  swa_lrs=cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR,
                                                  annealing_epochs=1)
        trainer_callbacks.append(swa_callbacks)


    trainer = Trainer(gpus=[0], fast_dev_run=fast_dev_run, accumulate_grad_batches=cfg.TRAIN.ACCUM_GRAD_BATCHES,
                      max_epochs=max_epochs, deterministic=True, callbacks=trainer_callbacks, enable_model_summary=False,
                      num_sanity_val_steps=0, enable_progress_bar=True, logger=logger,
                      gradient_clip_val=0.,
                      limit_train_batches=cfg.TRAIN.LIMIT_TRAIN_BATCHES, limit_val_batches=1.,
                      precision=32 // (cfg.TRAIN.MIXED_PRECISION + 1),
                      auto_lr_find=True, #auto_scale_batch_size=None,
                      )


    if cfg.TEST_ONLY != 'none':
        header_name = ['image_location', 'valence', 'arousal']
        with open('VA.txt', 'a+') as fd:
            fd.write(','.join(header_name) + '\n')

        print('Testing only. Loading checkpoint: ', cfg.TEST_ONLY)
        if not os.path.isfile(cfg.TEST_ONLY):
            raise ValueError('Could not find {}'.format(cfg.TEST_ONLY))
        # Load pretrained weights
        pretrained_state_dict = torch.load(cfg.TEST_ONLY)['state_dict']
        ABAW5_model.load_state_dict(pretrained_state_dict, strict=True)

        # Prepare test set
        ABAW5_dataset.setup(stage='test')
        # Generate prediction
        print("Evaluate Validation")
        # trainer.test(dataloaders=ABAW5_dataset.test_dataloader(), ckpt_path=None, model=ABAW5_model)
        print('Generate prediction')
        test = trainer.predict(dataloaders=ABAW5_dataset.test_dataloader(), ckpt_path=None, model=ABAW5_model)
        print('Testing finished.')
    else:
        #
        trainer.fit(ABAW5_model, datamodule=ABAW5_dataset)
        print('Pass with best val_metric: {}. Generating the prediction ...'.format(ckpt_cb.best_model_score))
        if cfg.OPTIM.USE_SWA:
            print('Evaluating with SWA')
            trainer.test(dataloaders=ABAW5_dataset.val_dataloader(), ckpt_path=None, model=ABAW5_model)
            trainer.save_checkpoint(ckpt_cb.last_model_path.replace('.ckpt', '_swa.ckpt'))

        trainer.test(dataloaders=ABAW5_dataset.val_dataloader(), ckpt_path='best')
        trainer.predict(dataloaders=ABAW5_dataset.test_dataloader(), ckpt_path='best')
