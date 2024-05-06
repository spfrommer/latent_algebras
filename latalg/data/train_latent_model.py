from __future__ import annotations
import random
import string
from typing import Dict, List, Tuple, Union
from einops import rearrange
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn.functional as F
from siren_pytorch import SirenNet

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm

import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
# import torchexplorer

from latalg.utils import dirs
from latalg.data.voronoi import pad_points_to_max_len

from dataclasses import dataclass
import click
import wandb
from latalg.data.voronoi import VoronoiShape, remove_padding_from_points, voronoi_target_func
from latalg.utils import cmd_utils, file_utils, log_utils, pretty, dirs, vis_utils
from latalg.utils import torch_utils as TU

from inr2vec.models.encoder import Encoder
from inr2vec.models.idecoder import ImplicitDecoder
from inr2vec.utils import focal_loss

import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

INR_DATA_DIR = dirs.data_path('inr')
DATA_DIR = dirs.data_path('latent_model')


@dataclass
class Params:
    project: str
    train_latent_model: bool
    run_name: str
    group: str
    
    lr: float
    weight_decay: float
    lr_decay_epochs: int
    batch_size: int
    max_epochs: int
    latent_embedding_dim: int

_click_options = [
    click.option('--project', default='latalg'),
    click.option('--train_latent_model', default=True),
    click.option('--run_name', default='latent_model_train'),
    click.option('--group', default='latent_model'),
    
    click.option('--lr', default=0.0001),
    click.option('--weight_decay', default=0.001),
    click.option('--lr_decay_epochs', default=0),
    click.option('--batch_size', default=16),
    click.option('--max_epochs', default=1000),
    click.option('--latent_embedding_dim', default=1024),
]


@click.command(context_settings={'show_default': True})
@cmd_utils.add_click_options(_click_options)
def train(**all_params) -> None:
    if all_params['run_name'] == 'random':
        all_params['run_name'] = ''.join(
            random.choices(string.ascii_uppercase, k=10)
        )

    load_params = cmd_utils.get_params_loader(all_params)
    params = load_params(Params)
    
    project_path = dirs.out_path(params.project)
    
    if params.train_latent_model:
        file_utils.create_empty_directory(dirs.path(project_path, params.run_name))
        logger = WandbLogger(
            save_dir=dirs.path(project_path, params.run_name),
            name=params.run_name,
            project=params.project,
            group=params.group
        )
        logger.experiment.config.update(all_params)
        
        checkpoint = ModelCheckpoint(
            save_top_k=1,
            dirpath=dirs.data_path(project_path, params.run_name, 'checkpoints'),
            filename='best',
            monitor='monitored',
            save_last=True,
        )

        trainer = L.Trainer(
            logger=logger,
            callbacks=[checkpoint],
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            accelerator='gpu',
            devices=1,
            max_epochs=params.max_epochs,
            check_val_every_n_epoch=1,
            deterministic=True,
            # detect_anomaly=True,
        )

        data = INRDataModule(batch_size=params.batch_size)

        module = LatentModule(
            inr_hidden_dim=get_inr_hidden_dim(),
            latent_embedding_dim=params.latent_embedding_dim,
            lr=params.lr,
            weight_decay=params.weight_decay,
            lr_decay_epochs=params.lr_decay_epochs,
        )
        trainer.fit(module, data)
        wandb.finish()
    
    print('Backing up encoder')
    copy_encoder_to_data(params.project, params.run_name)
    
    module = LatentModule.load_from_checkpoint(
        dirs.data_path('latent_model', 'checkpoints', 'best.ckpt')
    ).cuda()
    module.eval()
    
    print('Labelling INR data with latents')
    label_inr_data(module)


def copy_encoder_to_data(project: str, run_name: str) -> None:
    source_path = dirs.out_path(project, run_name)
    target_path = dirs.data_path('latent_model')
    file_utils.create_empty_directory(target_path)
    file_utils.copy_directory(source_path, target_path)
    
def label_inr_data(module: LatentModule) -> None:
    module.eval()
    for i in range(file_utils.num_files(INR_DATA_DIR)):
        dir = dirs.path(INR_DATA_DIR, f'{i}', 'inr_original')
        inr_params_matrix = torch.load(dirs.path(dir, 'inr_params_matrix.pt'))
        inr_params_matrix.requires_grad = False
        inr_params_matrix = inr_params_matrix.to(module.device)
        encoded_vector = module.encoder(inr_params_matrix.unsqueeze(0)).squeeze(0)
        save_dir = dirs.path(INR_DATA_DIR, f'{i}', 'inr_latent')
        file_utils.ensure_created_directory(save_dir)
        torch.save(encoded_vector, dirs.path(save_dir, 'encoded_vector.pt'))

def get_inr_hidden_dim() -> int:
    params_path = dirs.path(INR_DATA_DIR, '0', 'inr_original', 'inr_params_matrix.pt')
    params = torch.load(params_path)
    return params.shape[1]


class INRDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 4, num_workers: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Length is number of subdirectories in INR_DATA_DIR
        self.length = file_utils.num_files(INR_DATA_DIR)
        
    def setup(self, stage: str = None) -> None:
        train_proportion = 0.8 # Must be same as in datamodule.py
        train_index = int(train_proportion * self.length)
        self.train_dataset = INRDataset(
            data_dir=INR_DATA_DIR,
            start_index=0,
            end_index=train_index,
        )
        self.val_dataset = INRDataset(
            data_dir=INR_DATA_DIR,
            start_index=train_index,
            end_index=self.length,
        )
    
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )


class LatentModule(L.LightningModule):
    def __init__(
            self,
            inr_hidden_dim: int,
            latent_embedding_dim: int,
            lr: float,
            weight_decay: float,
            lr_decay_epochs: int,
        ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.encoder = Encoder(
            input_dim=inr_hidden_dim,
            hidden_dims=[512, 512, 1024, 1024],
            # hidden_dims=[512, 512, 512],
            # hidden_dims=[264, 264, 264],
            embed_dim=latent_embedding_dim,
        )
        
        self.decoder = ImplicitDecoder(
            embed_dim=latent_embedding_dim,
            in_dim=2,
            hidden_dim=512,
            # hidden_dim=264,
            num_hidden_layes_before_skip=2,
            num_hidden_layes_after_skip=2,
            out_dim=1
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.latent_embedding_dim = latent_embedding_dim
        
        self.plot_epochs = 10

        self.train_outputs = []
        self.valid_outputs = []
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.lr_decay_epochs > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_decay_epochs, gamma=0.1
            )
            return [optimizer], [scheduler]
        return optimizer

    def forward(self, inr_params_matrix: Tensor, query_points: Tensor) -> Tensor:
        encoded_params = self.encoder(inr_params_matrix)
        pred_labels = self.decoder(encoded_params, query_points)
        return pred_labels

    def training_step(self, batch, batch_idx):
        vars = self._step_variables(batch, batch_idx)
        self.train_outputs.append(vars)
        return vars

    def validation_step(self, batch, batch_idx):
        vars = self._step_variables(batch, batch_idx)
        self.valid_outputs.append(vars)
        return vars
        
    def _step_variables(self, batch, batch_idx):
        # encoded_params = self.encoder(batch['inr_params_matrix'])
        inr_params_matrix = batch['inr_params_matrix']
        query_points = batch['query_points']
        query_labels = rearrange(batch['query_labels'], 'b n 1 -> b n')
        
        pred_labels = self(inr_params_matrix, query_points)
        # loss = focal_loss(pred_labels, query_labels)
        loss = F.binary_cross_entropy_with_logits(pred_labels, query_labels.float())
        
        vars = {
            'loss': loss,
            'target': query_labels,
            'pred': pred_labels.detach(),
        }
        
        if self.is_plotting_epoch and batch_idx == 0:
            vars.update({
                'inshape_points': batch['shape_inshape_points'],
                'outshape_points': batch['shape_outshape_points'],
                'query_points': query_points,
                'pred_labels': pred_labels,
                'accuracy': ((pred_labels >= 0) == query_labels.bool()).float().mean()
            })
        
        return vars

    def on_validation_epoch_end(self) -> None:
        vars = {}
        vars.update(self._collect_losses(self.train_outputs, 'train'))
        vars.update(self._collect_losses(self.valid_outputs, 'valid'))
        vars.update(self._compute_accuracy(self.train_outputs, 'train'))
        vars.update(self._compute_accuracy(self.valid_outputs, 'valid'))

        # Monitor the validation loss for early stopping, best model saving, etc.
        self.log('monitored', vars['valid_loss'])

        # Prefix the metrics so that they are grouped appropriately in wandb
        vars = self._prefix_metrics(vars)

        if self.is_plotting_epoch():
            vars.update(self.create_media(self.train_outputs, 'Train'))
            vars.update(self.create_media(self.valid_outputs, 'Valid'))

        vars['epoch'] = self.current_epoch
        self.logger.experiment.log(vars)

        self.train_outputs.clear()
        self.valid_outputs.clear()

    def _collect_losses(self, outputs: List[Dict], stage: str) -> Dict:
        losses = self._calc_means_containing_key('loss', outputs)
        return {f'{stage}_{k}': v for k, v in losses.items()}

    def _calc_means_containing_key(
            self, in_key: str, outputs: List[Dict], stack=True
        ) -> Dict[str, Float[Tensor, '']]:
        """Collects the outputs for each key containing in_key and averages."""

        def collect(key: str, outputs: List[Dict], stack=False) -> Tensor:
            """Aggregates a specific key across all outputs into one Tensor."""
            collect_func = torch.stack if stack else torch.cat
            return collect_func([r[key] for r in outputs])

        return {
            key: collect(key, outputs, stack=stack).mean()
            for key in outputs[0].keys() if in_key in key
        }

    def _compute_accuracy(self, outputs: List[Dict], stage: str) -> Dict:
        preds, targets = self._collect_preds_targets(outputs)
        accuracy = ((preds >= 0) == targets.bool()).float().mean()
        return {f'{stage}_accuracy': accuracy}

    def _collect_preds_targets(
            self, outputs: List[Dict]
        ) -> Tuple[Float[Tensor, 'b class'], Int[Tensor, 'b']]:

        preds = torch.cat([r['pred'] for r in outputs])
        targets = torch.cat([r['target'] for r in outputs])

        return preds, targets

    def _prefix_metrics(self, metrics: dict) -> Dict:
        prefixed = {}
        for (k, v) in metrics.items():
            prefix = '1) '
            prefix += 'Loss' if 'loss' in k else 'Accuracy'
            prefix += '/'
            prefix += 'Train' if 'train' in k else 'Valid'

            prefixed[f'{prefix}/{k}'] = v
        return prefixed


    def is_plotting_epoch(self) -> bool:
        return (self.current_epoch % self.plot_epochs) == 0

    @torch.no_grad()
    def create_media(
            self, outputs: List[Dict], stage: str
        ) -> Dict[str, Union[wandb.Image, wandb.Video]]:
        
        media = {}
        
        media[f'2) {stage}INR/0'] = self._render_inr(outputs[0], 0)
        media[f'2) {stage}INR/1'] = self._render_inr(outputs[0], 1)
        media[f'2) {stage}INR/2'] = self._render_inr(outputs[0], 2)
        media[f'2) {stage}INR/3'] = self._render_inr(outputs[0], 3)

        return media

    def _render_inr(self, out: Dict, index: int) -> Figure:
        query_points, pred_labels = out['query_points'], out['pred_labels']
        inshape_points, outshape_points = out['inshape_points'], out['outshape_points']
        
        query_points, pred_labels = query_points[index], pred_labels[index]
        inshape_points, outshape_points = inshape_points[index], outshape_points[index]
        
        inshape_points = remove_padding_from_points(inshape_points)
        outshape_points = remove_padding_from_points(outshape_points)
        
        shape = VoronoiShape(inshape_points, outshape_points)
        target_func = voronoi_target_func(shape)

        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        
        tensor_input = torch.tensor(
            np.stack([X, Y], axis=-1), dtype=torch.float32
        ).to(self.device) 
        
        tensor_input = rearrange(tensor_input, 'x y c -> (x y) c')

        Z_func = TU.np(target_func(tensor_input))
        Z_func = rearrange(Z_func, '(x y) 1 -> x y', x=100)

        fig, axs = plt.subplots(1, 2, figsize=(10, 7))
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        
        args = {
            'extent': [-1,1,-1,1], 'origin': 'lower', 'cmap': 'coolwarm', 'norm': norm
        }
        axs[0].set_title('Function Output')
        axs[0].imshow(Z_func, **args)

        axs[1].set_title('Latent Reconstructed Output')
        axs[1].scatter(
            TU.np(query_points[:, 0]),
            TU.np(query_points[:, 1]),
            c=TU.np(pred_labels),
            cmap='coolwarm'
        )
        axs[1].set_aspect('equal')

        return vis_utils.render_and_close_figure(fig)
    

class INRDataset(Dataset):
    """Start inclusive, stop not inclusive."""
    def __init__(
            self,
            data_dir: str,
            start_index: int,
            end_index: int,
        ):
        self.data_dir = data_dir
        self.start_index = start_index
        self.end_index = end_index
    
    def __len__(self):
        return self.end_index - self.start_index
    
    def __getitem__(self, index):
        dir = dirs.path(self.data_dir, f'{index + self.start_index}', 'inr_original')
        
        query_points = torch.load(dirs.path(dir, 'query_points.pt'))
        query_labels = torch.load(dirs.path(dir, 'query_labels.pt'))
        shape = torch.load(dirs.path(dir, 'shape.pt'))
        inr_params_matrix = torch.load(dirs.path(dir, 'inr_params_matrix.pt'))
        inr_params_matrix.requires_grad = False
        
        return {
            'query_points': query_points,
            'query_labels': query_labels,
            'shape_inshape_points': pad_points_to_max_len(shape.inshape_points),
            'shape_outshape_points': pad_points_to_max_len(shape.outshape_points),
            'inr_params_matrix': inr_params_matrix,
        }


if __name__ == "__main__":
    train()