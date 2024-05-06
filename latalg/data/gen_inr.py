from __future__ import annotations
import random
from typing import Callable, Tuple
from einops import pack, rearrange
from jaxtyping import Float, Bool

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset 
import torch.nn.functional as F
from siren_pytorch import SirenNet

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm

import lightning as L
import tqdm
import wandb

from dataclasses import dataclass
import click
from latalg.data.voronoi import make_voronoi_shape, voronoi_target_func
from latalg.utils import cmd_utils, file_utils, log_utils, pretty, dirs
from latalg.utils import torch_utils as TU

import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

DATA_DIR = dirs.data_path('inr')


@dataclass
class Params:
    data_n: int
    voronoi_random_max: int
    query_n: int
    inr_train_epochs: int
    inr_train_batch_size: int

_click_options = [
    click.option('--data_n', default=10000),
    click.option('--voronoi_random_max', default=10),
    click.option('--query_n', default=5000),
    click.option('--inr_train_epochs', default=10),
    click.option('--inr_train_batch_size', default=1000),
]


@click.command(context_settings={'show_default': True})
@cmd_utils.add_click_options(_click_options)
def generate(**all_params) -> None:
    load_params = cmd_utils.get_params_loader(all_params)
    params = load_params(Params)

    file_utils.create_empty_directory(DATA_DIR)
    
    pretty.section_print('Generating data')
    gen_data(params)


def gen_data(params: Params):
    init_dir = dirs.data_path('inr_init')
    file_utils.ensure_created_directory(init_dir)
    reference_inr = INRModule()
    torch.save(reference_inr.state_dict(), dirs.path(init_dir, 'state_dict.pt'))

    for i in tqdm.tqdm(range(params.data_n)):
        dir = dirs.path(DATA_DIR, f'{i}', 'inr_original')
        file_utils.ensure_created_directory(dir)
        
        shape = make_voronoi_shape(params.voronoi_random_max)
        target_func = voronoi_target_func(shape)
        query_points, query_labels = query_points_labels(target_func, params.query_n)
        inr = fit_inr(query_points, query_labels, params)
        
        torch.save(shape, dirs.path(dir, 'shape.pt'))
        
        torch.save(query_points, dirs.path(dir, 'query_points.pt'))
        torch.save(query_labels, dirs.path(dir, 'query_labels.pt'))
        
        params_matrix = extract_parameter_matrix(inr).transpose(0, 1)
        torch.save(params_matrix, dirs.path(dir, 'inr_params_matrix.pt'))
        
        fig = render_inr(target_func, inr)
        fig.savefig(dirs.path(dir, 'inr_reconstruction.png'))
        plt.close(fig)
        
        fig = scatter_query_points(query_points, query_labels)
        fig.savefig(dirs.path(dir, 'scatter.png'))
        plt.close(fig)

def extract_parameter_matrix(inr: INRModule) -> Tensor:
    params = []
    for layer in inr.net.layers:
        params.append(layer.weight)
        params.append(rearrange(layer.bias, 'hid_d -> hid_d 1'))
    params.append(rearrange(inr.net.last_layer.weight, 'out_d hid_d -> hid_d out_d'))
    params_cat, _ = pack(params, 'hidden_d *')
    return params_cat

def query_points_labels(target_func: Callable, query_n: int) -> Tuple[Tensor, Tensor]:
    query_points = torch.rand(query_n, 2) * 2 - 1
    query_labels = target_func(query_points)
    return query_points, query_labels

def fit_inr(
        query_points: Tensor, query_labels: Tensor, params: Params
    ) -> INRModule:
    
    module = INRModule()
    state_dict_path = dirs.data_path('inr_init', 'state_dict.pt')
    module.load_state_dict(torch.load(state_dict_path))

    dataset = TensorDataset(query_points, query_labels)
    dataloader = DataLoader(
        dataset, batch_size=params.inr_train_batch_size, shuffle=True
    )
    trainer = L.Trainer(
        max_epochs=params.inr_train_epochs,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(module, dataloader)
    return module

def render_inr(target_func: Callable, inr: nn.Module) -> Figure:
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    tensor_input = torch.tensor(
        np.stack([X, Y], axis=-1), dtype=torch.float32
    ).to(inr.device) 
    
    tensor_input = rearrange(tensor_input, 'x y c -> (x y) c')

    Z_func = TU.np(target_func(tensor_input))
    Z_func = rearrange(Z_func, '(x y) 1 -> x y', x=100)

    Z_inr = TU.np(inr(tensor_input))
    Z_inr = rearrange(Z_inr, '(x y) 1 -> x y', x=100)

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    args = {
        'extent': [-1, 1, -1, 1], 'origin': 'lower', 'cmap': 'coolwarm', 'norm': norm
    }
    axs[0].set_title('Function Output')
    axs[0].imshow(Z_func, **args)
    axs[1].set_title('INR Output')
    axs[1].imshow(Z_inr, **args)
    return fig

def scatter_query_points(query_points: Tensor, query_labels: Tensor) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(query_points[:, 0], query_points[:, 1], c=query_labels, cmap='coolwarm')
    return fig

class INRModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = SirenNet(
            dim_in=2,
            dim_hidden=128,
            dim_out=1,
            num_layers=3,
            final_activation=torch.nn.Identity(),
            w0_initial=30.0,
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

if __name__ == "__main__":
    generate()