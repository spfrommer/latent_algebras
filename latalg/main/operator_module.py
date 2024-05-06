from __future__ import annotations
from collections.abc import MutableMapping

import click
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from torch import Tensor
import lightning as L
import wandb

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm

from dataclasses import dataclass
from jaxtyping import Shaped, Int, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Callable, Iterable, Optional, Union, List, Dict, Tuple
from latalg.data.voronoi import VoronoiShape, voronoi_target_func

from latalg.main import algebra as algebra_module
from latalg.data.train_latent_model import LatentModule

from latalg.utils import torch_utils as TU
from latalg.utils import vis_utils, cmd_utils, model_utils

# import INN
import nice


Batch = Dict


@dataclass
class OperatorModuleParams:
    optimizer: str
    lr: float
    decay_epochs: int

    plot_epochs: int

    algebra: str
    max_literal_n: int

    # If algebra is directparam
    directparam_algebra_symmetric: bool

    # If algebra is transported
    transported_index: int # Which transported in algebra.py
    transported_operations: Tuple[str, str] # No click option for this, add dynamically
    compute_induced_metrics: bool


click_options = [
    click.option('--optimizer', default='adam'),
    click.option('--lr', default=1e-4),
    click.option('--decay_epochs', default=0),

    click.option('--plot_epochs', default=1),

    click.option(
        '--algebra',
        default='transported',
        type=click.Choice(['transported', 'directparam'])
    ),
    click.option('--max_literal_n', default=2),

    click.option('--directparam_algebra_symmetric', default=False),

    click.option('--transported_index', default=0),
    click.option('--compute_induced_metrics', default=False),
]

def preprocess_all_params(all_params: Dict) -> Dict:
    all_params = all_params.copy()
    if all_params['algebra'] == 'transported':
        all_params['transported_operations'] = (
            algebra_module.transported_algebra_ops[all_params['transported_index']]
        )
    else:
        all_params['transported_operations'] = None
    return all_params



@TU.for_all_methods(jaxtyped)
@TU.for_all_methods(typechecker)
class OperatorModule(L.LightningModule):
    def _setup(
            self,
            latent_module: LatentModule,
            params: OperatorModuleParams
        ) -> None:
        # Break out setup for saving hyperparameters

        self.latent_module = latent_module.train(False)
        self.latent_module.requires_grad_(False)
        self.latent_embedding_d = latent_module.latent_embedding_dim
        self.mirrored_d = latent_module.latent_embedding_dim
        self.params = params

        # Enabled during testing
        self.compute_full_metrics = False

        if params.algebra == 'directparam':
            mlp_params = model_utils.MLPParams(
                hidden_d=256, layer_n=2
            )
            self.algebra = algebra_module.DirectParamLatentAlgebra(
                embedding_d=self.latent_embedding_d,
                mlp_params=mlp_params,
                intermediate_d=256,
                permutation_invariant=params.directparam_algebra_symmetric,
            )
        elif params.algebra == 'transported':
            mlp_params = model_utils.MLPParams()
            assert self.latent_embedding_d == self.mirrored_d

            self.phi = nice.NICEModel(
                input_dim=self.mirrored_d,
                num_layers=2, # Should be atleast 2
                nonlin_hidden_dim=256,
                nonlin_num_layers=1
            )
            self.phi_inv = lambda mirror: self.phi.inverse(mirror)

            self.algebra = algebra_module.get_transported_algebra_ops(
                *params.transported_operations,
            )

            # Just a test to make sure the algebra is correct
            index_algebra = algebra_module.get_transported_algebra_index(
                params.transported_index
            )
            assert index_algebra.and_operation == self.algebra.and_operation
            assert index_algebra.or_operation == self.algebra.or_operation
        else:
            raise RuntimeError(f'Unknown algebra: {params.algebra}')

        self.train_outputs = []
        self.valid_outputs = []

    def train(self, mode=True):
        super().train(mode)
        self.latent_module.train(False)
        return self

    def configure_optimizers(self):
        optimizer_classes = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
        optimizer = optimizer_classes[self.params.optimizer](
            self.parameters(), lr=self.params.lr
        )

        if self.params.decay_epochs > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.params.decay_epochs, gamma=0.1
            )
            return [optimizer], [scheduler]
        return optimizer


    # -------------------------------------------------------------------------------- #
    #                                     STEP CODE                                    #
    # -------------------------------------------------------------------------------- #


    def training_step(self, batch: Batch, batch_idx: int) -> Dict:
        assert self.training
        assert not self.latent_module.training

        variables = self._step_variables(batch, batch_idx)
        self.train_outputs.append(variables)
        return variables

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int) -> Dict:
        assert not self.training
        assert not self.latent_module.training

        variables = self._step_variables(batch, batch_idx)
        self.valid_outputs.append(variables)
        return variables


    def _step_variables(self, batch: Batch, batch_idx: int) -> Dict:
        encoded_latent = batch['encoded_latent']
        batch_n = encoded_latent.shape[0]
        if batch_idx == 0:
            self.test_encoder_correct(batch['inr_params_matrix'], encoded_latent)

        is_directparam_algebra = self.params.algebra == 'directparam'

        test_points = torch.rand(1000, 2).to(self.device) * 2 - 1

        # Compute the inset/outset labels for each target func (shape)
        # voronoi_tfs = self.get_voronoi_target_funcs(batch)
        # assert len(voronoi_tfs) == batch_n
        # tf_labels = [rearrange(tf(test_points), 'b 1 -> b') for tf in voronoi_tfs]
        
        with torch.no_grad():
            tf_labels = [
                (
                    self.latent_module.decoder(
                        latent.unsqueeze(0), test_points.unsqueeze(0)
                    ).squeeze(0).detach()
                    >= 0
                ).long()
                for latent in encoded_latent
            ]

        # directparam operations happen in latent space
        mir_or_lat = encoded_latent if is_directparam_algebra else self.phi(encoded_latent)

        vars = {}
        for literal_n in range(2, self.params.max_literal_n + 1):
            # First pick literal_n random "sets"
            random_indices = torch.randperm(batch_n)[:literal_n]
            select_tf_labels = [tf_labels[i] for i in random_indices]
            select_mir_or_lat = mir_or_lat[random_indices]

            # Generate a random expression
            random_expression = algebra_module.random_expression(literal_n)

            # Compute the ground truth correct set for the expression
            gt_labels = algebra_module.evaluate_expression(
                expr=random_expression,
                symbols=algebra_module.symbolize(select_tf_labels),
                ba=algebra_module.SetMembershipAlgebra()
            )

            # Compute the predicted set for the expression
            def compute_pred_logits(expr) -> Tensor:
                mir_or_lat_eval = algebra_module.evaluate_expression(
                    expr=expr,
                    symbols=algebra_module.symbolize(select_mir_or_lat),
                    ba=self.algebra
                )
                mir_or_lat_eval = rearrange(mir_or_lat_eval, 'd -> 1 d')

                if is_directparam_algebra:
                    latent_eval = mir_or_lat_eval
                else:
                    with TU.evaluating(self.phi): # phi_inv in terms of phi
                        latent_eval = self.phi_inv(mir_or_lat_eval)

                return self.latent_module.decoder(
                    latent_eval, test_points.unsqueeze(0)
                ).squeeze(0)

            pred_logits = compute_pred_logits(random_expression)
            pred_labels = (pred_logits >= 0)

            # Compute metrics
            loss = self.compute_loss(pred_logits, gt_labels)
            acc = self.compute_accuracy(pred_logits, gt_labels)
            iou = self.compute_iou(pred_logits, gt_labels)


            var_key = f'literal_n:{literal_n}'
            vars[var_key] = { 'loss': loss, 'accuracy': acc, 'iou': iou }

            # Compute error under equivalent transformations
            if literal_n == self.params.max_literal_n and self.compute_full_metrics:
                equiv_ious = []
                equiv_accs = []
                for law_n in range(9):
                    equiv_expression = algebra_module.random_equivalent_expression(
                        random_expression, law_n=law_n
                    )
                    equiv_logits = compute_pred_logits(equiv_expression)
                    equiv_ious.append({
                        'law_n': law_n,
                        'val': self.compute_iou(equiv_logits, pred_labels)
                    })
                    equiv_accs.append({
                        'law_n': law_n,
                        'val': self.compute_accuracy(equiv_logits, pred_labels)
                    })

                vars[var_key].update({
                    'equivalent_iou_to_pred': equiv_ious,
                    'equivalent_accuracy_to_pred': equiv_accs,
                })

            # Log stuff for visualization
            if batch_idx == 0:
                vars[var_key].update({
                    'expression': random_expression,
                    'test_points': test_points,
                    'tf_set_labels': select_tf_labels,
                    'gt_set_labels': gt_labels,
                    'pred_set_logits': pred_logits,
                })

        # Loss is what is used for backprop
        vars['loss'] = sum([v['loss'] for v in vars.values()])

        return vars

    def get_voronoi_target_funcs(self, batch: Batch) -> List[Callable]:
        inshape_points = batch['shape_inshape_points']
        outshape_points = batch['shape_outshape_points']
        funcs = []
        for ip, op in zip(inshape_points, outshape_points):
            funcs.append(voronoi_target_func(VoronoiShape(ip, op)))
        return funcs

    def test_encoder_correct(self, inr_params_matrix, encoded_latent):
        test_encoded_latent = self.latent_module.encoder(inr_params_matrix)
        assert torch.allclose(test_encoded_latent, encoded_latent, atol=5e-3)

    def compute_loss(self, pred_logits, gt_labels):
        return F.binary_cross_entropy_with_logits(pred_logits, gt_labels.float())

    def compute_accuracy(self, pred_logits, gt_labels):
        return ((pred_logits >= 0) == gt_labels.bool()).float().mean()

    def compute_iou(self, pred_logits, gt_labels) -> Optional[Tensor]:
        pred_labels, gt_labels = (pred_logits >= 0), gt_labels.bool()
        intersection_n = (pred_labels & gt_labels).float().sum()
        union_n = (pred_labels | gt_labels).float().sum()
        if union_n < 0.5:
            return None
        return intersection_n / union_n

    # -------------------------------------------------------------------------------- #
    #                                   LOGGING CODE                                   #
    # -------------------------------------------------------------------------------- #

    def on_validation_epoch_end(self) -> None:
        train_outputs = [self._flatten(o) for o in self.train_outputs]
        valid_outputs = [self._flatten(o) for o in self.valid_outputs]

        vars = {}
        vars.update(self._collect_for_key(train_outputs, 'train', 'loss'))
        vars.update(self._collect_for_key(valid_outputs, 'valid', 'loss'))
        vars.update(self._collect_for_key(train_outputs, 'train', 'acc'))
        vars.update(self._collect_for_key(valid_outputs, 'valid', 'acc'))
        vars.update(self._collect_for_key(train_outputs, 'train', 'iou'))
        vars.update(self._collect_for_key(valid_outputs, 'valid', 'iou'))

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

    def _collect_for_key(self, outputs: List[Dict], stage: str, in_key: str) -> Dict:
        losses = self._calc_means_containing_key(in_key, outputs)
        return {f'{stage}_{k}': v for k, v in losses.items()}

    def _calc_means_containing_key(
            self, in_key: str, outputs: List[Dict], stack=True
        ) -> Dict[str, Float[Tensor, '']]:
        """Collects the outputs for each key containing in_key and averages."""

        def collect(key: str, outputs: List[Dict], stack=False) -> Tensor:
            """Aggregates a specific key across all outputs into one Tensor."""
            collect_func = torch.stack if stack else torch.cat
            # iou metric could be none
            try:
                return collect_func([r[key] for r in outputs if (r[key] != None)])
            except Exception as ex:
                import pdb; pdb.set_trace()

        return {
            key: collect(key, outputs, stack=stack).mean()
            for key in outputs[0].keys() if in_key in key
        }

    def _flatten(self, dictionary: Dict, parent_key='', separator='_'):
        # Adapted from: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + separator + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(self._flatten(value, new_key, separator=separator).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def _prefix_metrics(self, metrics: dict) -> Dict:
        prefixed = {}
        for (k, v) in metrics.items():
            prefix = '1) '
            prefix += 'Train' if 'train' in k else 'Valid'
            prefix += 'Loss' if 'loss' in k else 'Accuracy'

            prefixed[f'{prefix}/{k}'] = v
        return prefixed


    # -------------------------------------------------------------------------------- #
    #                                   PLOTTING CODE                                  #
    # -------------------------------------------------------------------------------- #


    def is_plotting_epoch(self) -> bool:
        return (self.current_epoch % self.params.plot_epochs) == 0

    @torch.no_grad()
    def create_media(
            self, outputs: List[Dict], stage: str
        ) -> Dict[str, Union[wandb.Image, wandb.Video]]:

        media = {}

        out = outputs[0]
        for literal_n in range(2, self.params.max_literal_n + 1):
            literal_n_dict = out[f'literal_n:{literal_n}']
            media[f'2) {stage}Reconstruc/Literal {literal_n}'] = (
                self.create_reconstruction_plot(literal_n_dict)
            )

        return media

    def create_reconstruction_plot(self, literal_n_dict: Dict) -> wandb.Image:
        expression = literal_n_dict['expression']
        points = TU.np(literal_n_dict['test_points'])
        tf_set_labels = literal_n_dict['tf_set_labels']
        gt_set_labels = literal_n_dict['gt_set_labels']
        pred_set_logits = literal_n_dict['pred_set_logits']

        literal_n = len(tf_set_labels)
        def scatter_to_ax(labels, ax):
            norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            ax.scatter(
                points[:, 0], points[:, 1], c=TU.np(labels),
                cmap='coolwarm', norm=norm, s=0.5
            )
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if literal_n in [2, 3]:
            fig, axes = plt.subplots(1, literal_n + 2, figsize=(literal_n + 2, 1))
            for i in range(literal_n):
                scatter_to_ax(tf_set_labels[i].float() * 2 - 1, axes[i])
        else:
            fig, axes = plt.subplots(1, 2, figsize=(2, 1))

        scatter_to_ax(gt_set_labels.float() * 2 - 1, axes[-2])
        scatter_to_ax(pred_set_logits, axes[-1])
        exp_str = str(expression).replace('Symbol', '')
        ax_transform = axes[-2].transAxes
        plt.text(
            0.01, 1.01, exp_str, fontsize=8,
            ha='left', va='bottom', transform=ax_transform
        )

        return vis_utils.render_and_close_figure(fig)


class OperatorModuleWrapper(OperatorModule):
    # This wrapper allows us to save hyperparameters
    def __init__(self, latent_module: LatentModule, **all_params):
        L.LightningModule.__init__(self)

        self.save_hyperparameters(ignore=['latent_module'])

        for key, val in all_params.items():
            self.hparams[key] = val

        load_params = cmd_utils.get_params_loader(all_params)

        params = load_params(OperatorModuleParams)

        self._setup(
            latent_module=latent_module,
            params=params,
        )
