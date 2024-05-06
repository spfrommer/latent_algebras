from __future__ import annotations
from typing import Dict, Tuple

import lightning as L
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import random
import string

import colored_traceback
from dataclasses import dataclass
import click

from latalg.utils import pretty, dirs, file_utils
from latalg.utils import torch_utils as TU


@dataclass
class CoreParams:
    project: str
    group: str
    run_name: str
    clear_project: bool
    max_epochs: int
    seed: int


click_options = [
    click.option('--project', default='latalg'),
    click.option('--group', default='operator_learning'),
    click.option('--run_name', default='default'),
    click.option('--clear_project/--no_clear_project', default=False),
    click.option('--max_epochs', default=100),
    click.option('--seed', default=0),
]

def init(core_params):
    L.pytorch.seed_everything(core_params.seed, workers=True)

    pretty.init()
    colored_traceback.add_hook()

    import warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")


def handle_directory(params: CoreParams) -> None:
    if params.clear_project:
        file_utils.create_empty_directory(dirs.out_path(params.project))
    file_utils.ensure_created_directory(
        dirs.out_path(params.project, params.group, params.run_name)
    )


def setup_logging(
        params: CoreParams, all_params: Dict
    ) -> Tuple[WandbLogger, ModelCheckpoint]:

    project_path = dirs.out_path(params.project, params.group)
    file_utils.create_empty_directory(dirs.path(project_path, params.run_name))
    logger = WandbLogger(
        save_dir=dirs.path(project_path, params.run_name),
        group=params.group,
        name=params.run_name,
        project=params.project
    )
    logger.experiment.config.update(all_params)
    logger.experiment.log_code(dirs.code_path())
    
    checkpoint = ModelCheckpoint(
        save_top_k=1,
        dirpath=dirs.path(project_path, params.run_name, 'checkpoints'),
        filename='best',
        monitor='monitored',
        save_last=True,
    )

    return logger, checkpoint


def setup_trainer(
        params: CoreParams, logger: WandbLogger, checkpoint: ModelCheckpoint
    ) -> Trainer:

    return Trainer(
        logger=logger,
        callbacks=[checkpoint],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        accelerator='gpu',
        devices=1,
        max_epochs=params.max_epochs,
        check_val_every_n_epoch=1,
        deterministic=True,
        enable_model_summary=True
        # detect_anomaly=True,
    )

def random_run_name() -> str:
    # Useful for wandb hyperparameter sweeps 
    # This should be called before init() to avoid seed setting issue
    return ''.join(random.choices(string.ascii_uppercase, k=10))
