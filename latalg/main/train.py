from __future__ import annotations

from dataclasses import dataclass

from latalg.data.train_latent_model import LatentModule

import click
import wandb

from latalg.utils import log_utils, cmd_utils, pretty, dirs, file_utils

from latalg.main import core as latalg_core
from latalg.main import data_module as latalg_datamodule
from latalg.main import operator_module as latalg_module


@dataclass
class TrainParams:
    batch_size: int


click_options = [
    click.option('--batch_size', default=16),
]


@click.command(context_settings={'show_default': True})
@cmd_utils.add_click_options(latalg_core.click_options)
@cmd_utils.add_click_options(latalg_module.click_options)
@cmd_utils.add_click_options(click_options)
def run(**all_params) -> None:
    load_params = cmd_utils.get_params_loader(all_params)
    train_params: TrainParams = load_params(TrainParams)
    core_params: latalg_core.CoreParams = load_params(latalg_core.CoreParams)

    if core_params.run_name == 'random':
        core_params.run_name = latalg_core.random_run_name()

    latalg_core.init(core_params)
    latalg_core.handle_directory(core_params)
    
    json_path = dirs.out_path(
        core_params.project, core_params.group, core_params.run_name, 'all_params.json'
    )
    file_utils.write_json(json_path, all_params)

    pretty.section_print('Setting up logging and checkpointing')
    logger, checkpoint = latalg_core.setup_logging(core_params, all_params)

    pretty.section_print('Setting up modules')
    latent_module = LatentModule.load_from_checkpoint(
        dirs.data_path('latent_model', 'checkpoints', 'best.ckpt')
    ).cuda()

    module = latalg_module.OperatorModuleWrapper(
        latent_module=latent_module, **latalg_module.preprocess_all_params(all_params)
    )

    data = latalg_datamodule.CompleteINRDataModule(batch_size=train_params.batch_size)

    pretty.section_print('Training')
    trainer = latalg_core.setup_trainer(core_params, logger, checkpoint)
    trainer.fit(module, datamodule=data)

if __name__ == "__main__":
    run()
