from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker
from einops import rearrange, reduce, repeat, pack

from dataclasses import dataclass


@dataclass
class MLPParams:
    hidden_d: int = 256
    layer_n: int = 3
    initial_norm: str = 'none'
    final_norm: str = 'none' 

    bias: bool = True
    activation: nn.Module = nn.ReLU
    final_op: nn.Module = nn.Identity

    linear_class: nn.Module = nn.Linear


def create_mlp_from_params(
        input_d: int, output_d: int, params: MLPParams
    ) -> nn.Sequential:

    layers = []

    if params.initial_norm == 'batchnorm':
        layers.append(CustomBatchNorm1d(input_d))
    elif params.initial_norm == 'global_batchnorm':
        layers.append(GlobalShiftBatchNorm1d())
    elif params.initial_norm == 'layernorm':
        layers.append(nn.LayerNorm(normalized_shape=(input_d,), elementwise_affine=True))

    for i in range(params.layer_n):
        previous_d = input_d if i == 0 else params.hidden_d
        layers.append(params.linear_class(previous_d, params.hidden_d, bias=params.bias))
        layers.append(params.activation())

    before_last_d = input_d if params.layer_n == 0 else params.hidden_d
    layers.append(params.linear_class(before_last_d, output_d, bias=params.bias))
    layers.append(params.final_op())

    if params.final_norm == 'batchnorm':
        layers.append(CustomBatchNorm1d(output_d))
    elif params.initial_norm == 'global_batchnorm':
        layers.append(GlobalShiftBatchNorm1d())
    elif params.final_norm == 'layernorm':
        layers.append(nn.LayerNorm(normalized_shape=(output_d,), elementwise_affine=True))

    return nn.Sequential(*layers)


def init_weights(module: nn.Module):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    module.apply(init_weights)


class CustomBatchNorm1d(nn.BatchNorm1d):
    """Handles discrepancy where linear layers expect channels last but
    batchnorm1d expects channels second."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() == 3:
            x = rearrange(x, 'n l c -> n c l')
            x = super().forward(x)
            return rearrange(x, 'n c l -> n l c')
        else:
            return super().forward(x)
        
class GlobalShiftBatchNorm1d(nn.BatchNorm1d):
    def __init__(self):
        super().__init__(num_features=1)

    def forward(self, x):
        if x.dim() == 3:
            y = rearrange(x, 'n l c -> n (l c)')
        else:
            y = x
        y = rearrange(y, 'n c -> n () c')
        y = super().forward(y)
        y = rearrange(y, 'n 1 c -> n c')
        if x.dim() == 3:
            y = rearrange(y, 'n (l c) -> n l c', c=x.shape[-1])
        assert y.shape == x.shape
        return y



class PositiveLinear(nn.Module):
    # Adapted from: https://discuss.pytorch.org/t/positive-weights/19701/7

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

    def forward(self, input):
        return F.linear(input, F.softplus(self.weight), self.bias)