from __future__ import annotations
from collections import defaultdict

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

import tqdm

from latalg.data.train_latent_model import LatentModule
# os.environ["WANDB_SILENT"] = "true"

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from cycler import cycler


import click
import lightning as L

from latalg.utils import log_utils, cmd_utils, pretty, dirs, file_utils, vis_utils

from latalg.main import data_module as latalg_datamodule
from latalg.main.operator_module import OperatorModuleParams, OperatorModuleWrapper
from latalg.main import algebra as algebra_module


MAX_LITERAL_N = 10
results_a_path = dirs.test_path('results_a.pkl')
results_b_path = dirs.test_path('results_b.pkl')
results_c_path = dirs.test_path('results_c.pkl')
results_metadata_path = dirs.test_path('results_metadata.pkl')

@dataclass
class TestParams:
    project: str
    group: str
    compute_statistics: bool


click_options = [
    click.option('--project', default='latalg'),
    click.option('--group', default='maxminaddsub'),
    click.option('--compute_statistics/--no_compute_statistics', default=False),
]

@click.command(context_settings={'show_default': True})
@cmd_utils.add_click_options(click_options)
def run(**all_params) -> None:
    load_params = cmd_utils.get_params_loader(all_params)
    test_params: TestParams = load_params(TestParams)
    
    file_utils.ensure_created_directory(dirs.test_path())
    file_utils.create_empty_directory(dirs.test_path('figs'))

    latent_module = LatentModule.load_from_checkpoint(
        dirs.data_path('latent_model', 'checkpoints', 'best.ckpt')
    ).cuda()
    
    operator_modules = [
        load_transported_operator_module(test_params, i, latent_module)
        for i in range(algebra_module.num_tranported_combinations())
    ]
    
    operator_modules += [
        load_directparam_operator_module(test_params, False, latent_module),
        load_directparam_operator_module(test_params, True, latent_module)
    ]

    
    ignore = []
    operator_modules = [
        m for m in operator_modules if (
            m.params.algebra == 'directparam' or
            all(op not in ignore for op in m.params.transported_operations)
        )
    ]

    data = latalg_datamodule.CompleteINRDataModule(batch_size=MAX_LITERAL_N)
    data.setup()
    test_loader = data.test_dataloader()
    
    # file_utils.create_empty_directory(dirs.test_path('figs'))
    
    if test_params.compute_statistics:
        compute_test_statistics(test_loader, operator_modules)

    make_law_n_line_plot('equivalent_iou_to_pred')
    make_sat_laws_line_plot('iou')
    
    make_scatter_plot('iou', 'literal_n')
    make_scatter_plot('equivalent_iou_to_pred', 'law_n')
    
    all_laws = algebra_module.get_property_types()
    for i, law1 in enumerate(all_laws):
        for j, law2 in enumerate(all_laws[i+1:]):
            make_grouped_plot('iou', law1, law2)


# Shared between plots

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

cbar_key_lookup = {
    'loss': 'Loss (↓)',
    'accuracy': 'Accuracy (↑)',
    'iou': 'Intersection / union (↑)',
    'equivalent_iou_to_pred': 'Self-consistency – intersection / union (↑)',
    'equivalent_accuracy_to_pred': 'Consistency – accuracy (↑)',
}

law_sat_label = 'Satisfied distributive lattice laws (#)'
law_n_label = 'Random law applications'
literal_n_label = 'Random term symbols'

figsize = (4, 4)
dpi = 300

annotation_fontsize = 8.5
tick_fontsize=7.5

def get_annotation(param_key) -> Optional[Tuple[str, str]]:
    # Second is alignment, only for scatter
    algebra, directparam_symmetric, _, comb_operations = param_key
    if comb_operations == ('max', 'min'):
        return ('Riesz (ours)', 'center')
    if algebra == 'directparam':
        label = 'Direct param.'
        label += ' (sym)' if directparam_symmetric else ''
        xalign = 'left' if directparam_symmetric else 'center'
        return (label, xalign)
    return None

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

truncated_plasma = truncate_colormap(plt.get_cmap('plasma'), 0.0, 0.95)

my_colors = ['#fa9e3b', '#c23d80', '#120789']
# my_colors = ['#120789', '#fa9e3b', '#c23d80']
custom_cycler = cycler(color=my_colors)

# Specific plots

def make_scatter_plot(performance_key: str, process: str):
    path = results_a_path if process == 'literal_n' else results_c_path
    results = file_utils.read_pickle(path)
    results_metadata = file_utils.read_pickle(results_metadata_path)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # {num_sat: [{param_key: '', lol_ns: [], results: [{}, ...]}]}
    num_sat_groups = defaultdict(list)
    
    # lol = literal or law
    for param_key, lol_n_results in results.items():
        num_sat = results_metadata[param_key]['num_sat']
        lol_ns, results = zip(*lol_n_results.items())
        num_xticks = 9 # Max number of satisfied laws, want to make plot square
        lol_ns, results = lol_ns[:num_xticks], results[:num_xticks]
        num_sat_groups[num_sat].append({
            'param_key': param_key, 'lol_ns': lol_ns, 'results': results
        })
    
    xs, ys, zs = [], [], []
    for num_sat, group in num_sat_groups.items():
        num_algebras_with_num_sat = len(group)
        
        for i, algebra_res in enumerate(group):
            lol_ns, results = algebra_res['lol_ns'], algebra_res['results']

            hgap = 0.06 # half-gap between algebras on same column
            xpos = num_sat - i * hgap * 2 + (num_algebras_with_num_sat-1) * hgap
            xs += [xpos] * len(lol_ns)
            ys += lol_ns
            zs += [r[performance_key] for r in results]
            
            annotation = get_annotation(algebra_res['param_key'])
            if annotation is not None:
                _scatter_annotate_algebra(ax, xpos, min(ys), max(ys), *annotation)
    
    ax.tick_params(axis=u'x', which=u'both',length=0)
    ax.set_xticks(range(9), minor=False)
    ax.set_xticks([0.5, 1.5, 2.45, 3.55, 4.5, 5.5, 6.5, 7.5], minor=True)
    ax.xaxis.grid(True, which='minor', alpha=0.6, linewidth=0.7)
    
    vert_x, vert_y = 2, 20
    verts = list(zip([-vert_x,vert_x,vert_x,-vert_x],[-vert_y,-vert_y,vert_y,vert_y]))
    ax.scatter(xs, ys, c=zs, cmap=truncated_plasma, marker=verts, s=145)

    ax.set_xlabel(law_sat_label)
    ax.set_ylabel(literal_n_label if (process == 'literal_n') else law_n_label)
    ax.set_aspect('equal', 'box')
    
    
    ax_pos = ax.get_position()
    cax = fig.add_axes([ax_pos.x1 + 0.01, ax_pos.y0, 0.02, ax_pos.height])
    sm = plt.cm.ScalarMappable(cmap=truncated_plasma)
    sm.set_array(zs)
    cbar = fig.colorbar(sm, cax=cax)
    
    cbar.set_label(cbar_key_lookup[performance_key])
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.ax.set_yticklabels(
        cbar.ax.get_yticklabels(), rotation=45, rotation_mode='anchor'
    )
    
    ax.set_axisbelow(True)
    
    ax.set_xticks(range(int(min(xs)), int(max(xs)) + 1))
    ax.set_yticks(range(int(min(ys)), int(max(ys)) + 1))
    
    ax.tick_params(axis='both', which='both', labelsize=tick_fontsize)

    
    path = dirs.test_path('figs', f'{performance_key}_{process}_scatter.png')
    fig.savefig(path, bbox_inches='tight', transparent=True)
    path = dirs.test_path('figs', f'{performance_key}_{process}_scatter.pdf')
    fig.savefig(path, bbox_inches='tight', transparent=True)
    plt.close()
    
def _scatter_annotate_algebra(ax, x, min_y, max_y, algebra_name, xalign='center'):
    rectangle_pad_h, rectangle_pad_v = 0.09, 0.35
    ax.add_patch(Rectangle(
        xy = (x - rectangle_pad_h, min_y - rectangle_pad_v),
        width = 2 * rectangle_pad_h,
        height = (max_y - min_y) + 2 * rectangle_pad_v,
        fill=False,
        # linestyle = (0, (5, 1)),
        linestyle = 'dashed',
        zorder=2,
        alpha=0.7,
        linewidth=0.7
    ))
    relpos = (0.5, 0.5) if xalign == 'center' else (0, 0)
    ax.annotate(
        algebra_name,
        xy=(x, max_y + rectangle_pad_v + 0.0),
        xytext=(x, max_y + 1.0),
        horizontalalignment=xalign,
        verticalalignment='bottom',
        arrowprops=dict(arrowstyle="->", color='black', linewidth=1.0, relpos=relpos),
        bbox=dict(
            pad=0.25, facecolor="white", edgecolor="black", boxstyle='round', alpha=0.7
        ),
        fontsize=annotation_fontsize,
    )
    
    
def make_grouped_plot(performance_key: str, x_property: str, y_property: str):
    results = file_utils.read_pickle(results_b_path)
    results_metadata = file_utils.read_pickle(results_metadata_path)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Each entry in inner list is of the form {param_key: performance_value]
    grid = [[[] for _ in range(3)] for _ in range(3)]
    min_performance = 1e9
    max_performance = -1e9
    
    for param_key, performance in results.items():
        x_sat = results_metadata[param_key]['num_sat_for_property_type'][x_property]
        y_sat = results_metadata[param_key]['num_sat_for_property_type'][y_property]
        
        grid[x_sat][y_sat].append({
            'param_key': param_key, 'performance': performance[performance_key]
        })
        
        min_performance = min(min_performance, performance[performance_key])
        max_performance = max(max_performance, performance[performance_key])
    
    norm = plt.Normalize(min_performance, max_performance)
    
    for i in range(3):
        for j in range(3):
            group = grid[i][j]
            n_rows = math.ceil(math.sqrt(len(group)))
            
            # To make annotations cleaner
            group = sorted(group, key=lambda x: x['param_key'][0] != 'directparam')
            two_directparam_in_group = (
                len(group) >= 2 and
                group[0]['param_key'][0] == 'directparam' and
                group[1]['param_key'][0] == 'directparam'
            )
            if two_directparam_in_group:
                # Move to end so labels don't overlap
                group = group[1:] + group[:1]

            
            # Make a meshgrid of points centered at (i, j)
            x = np.arange(n_rows) * 0.1
            y = -np.arange(n_rows) * 0.1
            x = x - np.mean(x) + i
            y = y - np.mean(y) + j
            X, Y = np.meshgrid(x, y)
            
            X, Y = X.flatten()[:len(group)], Y.flatten()[:len(group)]

            Z = [r['performance'] for r in group]
            
            plt.scatter(X, Y, c=Z, cmap=truncated_plasma, norm=norm, s=20, marker='s')
            
            for k, algebra_data in enumerate(group):
                annotation = get_annotation(group[k]['param_key'])
                if annotation is not None:
                    _group_annotate_algebra(
                        ax, X[k], Y[k], annotation[0], label_above=(k == 0)
                    )
                
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    ax.tick_params(axis=u'both', which=u'both',length=0)

    ax.set_xticks([-0.5, 0.5, 1.5, 2.5], minor=True)
    ax.set_xticks([0, 1, 2], minor=False)
    ax.xaxis.grid(True, which='minor', alpha=0.2, linewidth=1.0)
    ax.set_xticklabels(['✘✘', '✔✘/✘✔', '✔✔'], fontfamily='sans-serif')

    ax.set_yticks([-0.5, 0.5, 1.5, 2.5], minor=True)
    ax.set_yticks([0, 1, 2], minor=False)
    ax.yaxis.grid(True, which='minor', alpha=0.2, linewidth=1.0)
    ax.set_yticklabels(
        ['✘✘', '✔✘/✘✔', '✔✔'], rotation=90, va='center', fontfamily='sans-serif'
    )
    
    ax.set_xlabel(x_property.capitalize())
    ax.set_ylabel(y_property.capitalize())
    ax.set_aspect('equal', 'box')
    
    
    ax_pos = ax.get_position()
    cax = fig.add_axes([ax_pos.x1 + 0.01, ax_pos.y0, 0.02, ax_pos.height])
    cbar = plt.colorbar(cax=cax)
    
    cbar.set_label(cbar_key_lookup[performance_key])
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.ax.set_yticklabels(
        cbar.ax.get_yticklabels(), rotation=45, rotation_mode='anchor'
    )
    cbar.ax.zorder = -1

    path = dirs.test_path('figs', f'{performance_key}_{x_property}_{y_property}_grouped.png')
    fig.savefig(dirs.test_path(path), bbox_inches='tight', transparent=True)
    path = dirs.test_path('figs', f'{performance_key}_{x_property}_{y_property}_grouped.pdf')
    fig.savefig(dirs.test_path(path), bbox_inches='tight', transparent=True)
    plt.close()
    
def _group_annotate_algebra(ax, x, y, algebra_name, label_above = True):
    rectangle_pad = 0.05
    ax.add_patch(Rectangle(
        xy = (x - rectangle_pad, y - rectangle_pad),
        width = 2 * rectangle_pad,
        height = 2 * rectangle_pad,
        fill=False,
        linestyle = '--',
    ))
    xy = (x, y + rectangle_pad) if label_above else (x, y - rectangle_pad)
    xytext = (x, y + 0.24) if label_above else (x, y - 0.24)
    verticalalignment = 'bottom' if label_above else 'top'
    ax.annotate(
        algebra_name,
        xy=xy,
        xytext=xytext,
        horizontalalignment='center',
        verticalalignment=verticalalignment,
        arrowprops=dict(arrowstyle='->', color='black', linewidth=1.0),
        bbox=dict(
            pad=0.25, facecolor='white', edgecolor='black', boxstyle='round', alpha=0.7
        ),
        fontsize=annotation_fontsize
    )


def make_law_n_line_plot(key: str):
    results = file_utils.read_pickle(results_c_path)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax.set_prop_cycle(custom_cycler)
    
    # Sort param keys to have 1st: if the param_key combination is max, min, and 2nd if it is directparam
    has_labelled_other_algeras = False
    for param_key, law_n_results in results.items():
        law_ns, results = zip(*law_n_results.items())
        # law_ns, results = law_ns[:9], results[:9]
        
        ys = [r[key + '_median'] for r in results]
        ys_20 = [r[key + '_25%'] for r in results]
        ys_80 = [r[key + '_75%'] for r in results]
        
        annotation = get_annotation(param_key)
        if annotation is None:
            label = None if has_labelled_other_algeras else 'Other algebras'
            ax.plot(
                law_ns, ys, color='k', alpha=0.1, zorder=1, label=label, linewidth=1.0
            )
            has_labelled_other_algeras = True
        else:
            # linestyle = '--' if param_key[0] == 'transported' else '-'
            linestyle = '-'
            zorder = 3 if param_key[0] == 'transported' else 2
            ax.plot(law_ns, ys, label=annotation[0], linestyle=linestyle, zorder=zorder)
            ax.fill_between(law_ns, ys_20, ys_80, alpha=0.2, zorder=2)
    
    ax.set_xlabel(law_n_label)
    ax.set_ylabel(cbar_key_lookup[key])
    
    ax.tick_params(axis='both', which='both', labelsize=tick_fontsize)
    
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 3, 1]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='lower right',
        fontsize=annotation_fontsize
    )
    
    path = dirs.test_path('figs', f'{key}_law_n_line.png')
    fig.savefig(path, bbox_inches='tight', transparent=True)
    path = dirs.test_path('figs', f'{key}_law_n_line.pdf')
    fig.savefig(path, bbox_inches='tight', transparent=True)
    plt.close()
    

def make_sat_laws_line_plot(performance_key: str, results_file='results_a'):
    if results_file == 'results_a':
        results = file_utils.read_pickle(results_a_path)
    else:
        # For this, literal_ns is actually law_ns...
        results = file_utils.read_pickle(results_c_path)
    results_metadata = file_utils.read_pickle(results_metadata_path)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # {num_sat: [{param_key: , literal_ns: [], results: [{}, ...]}]}
    num_sat_groups = defaultdict(list)
    
    for param_key, lol_n_results in results.items():
        num_sat = results_metadata[param_key]['num_sat']
        lol_ns, results = zip(*lol_n_results.items())
        num_xticks = 9 # Max number of satisfied laws, want to make plot square
        lol_ns, results = lol_ns[:num_xticks], results[:num_xticks]
        num_sat_groups[num_sat].append({
            'param_key': param_key, 'lol_ns': lol_ns, 'results': results
        })
    
    xs, ys = [], []
    for num_sat, group in num_sat_groups.items():
        xs.append(num_sat)
        group_ys = []
        for group_entry in group: # One specific algebra
            annotation = get_annotation(group_entry['param_key'])
            # Get mean performance across all lol_ns
            performances = [r[performance_key] for r in group_entry['results']]

            mean_perf = np.mean(performances)
            
            if 'directparam' in group_entry['param_key']:
                ax.scatter(
                    [num_sat], [mean_perf],
                    s=20, zorder=2, c='k', alpha=0.2, edgecolors='none'
                )
            else:
                group_ys.append(mean_perf)
            
            if annotation is not None:
                _sat_laws_annotate_algebra(ax, num_sat, mean_perf, annotation[0])
        
        ax.scatter(
            num_sat * np.ones(len(group_ys)), group_ys,
            s=20, zorder=2, c=my_colors[0], alpha=0.7, edgecolors='none'
        )
        ys.append(np.mean(group_ys))
    
    # Sort xs and ys
    xs, ys = zip(*sorted(zip(xs, ys)))
    ax.plot(
        xs, ys, color=my_colors[0], alpha=1.0, zorder=1,
        label='Mean (over transported algebras w/ same #)',
    )
    
    ax.set_xlabel(law_sat_label)
    ax.set_ylabel(cbar_key_lookup[performance_key])
    
    ax.tick_params(axis='both', which='both', labelsize=tick_fontsize)
    
    ax.legend(loc='lower right', fontsize=annotation_fontsize-1)
    
    path = dirs.test_path('figs', f'{performance_key}_sat_laws_line.png')
    fig.savefig(path, bbox_inches='tight', transparent=True)
    path = dirs.test_path('figs', f'{performance_key}_sat_laws_line.pdf')
    fig.savefig(path, bbox_inches='tight', transparent=True)
    plt.close()

def _sat_laws_annotate_algebra(ax, x, y, algebra_name):
    rectangle_pad_h, rectangle_pad_v = 0.1, 0.005
    label_above = ('sym' not in algebra_name) and ('ours' not in algebra_name)
    xy = (x, y + rectangle_pad_v) if label_above else (x, y - rectangle_pad_v)
    xytext = (x, y + 0.1) if label_above else (x, y - 0.1)
    verticalalignment = 'bottom' if label_above else 'top'
    if x < 0.1:
        horizontalalignment = 'left'
        relpos = (0, 0) if label_above else (0, 1)
    elif x > 7.9:
        horizontalalignment = 'right'
        relpos = (1, 0) if label_above else (1, 1)
    else:
        horizontalalignment = 'center'
        relpos = (0.5, 0.5)
    
    ax.annotate(
        algebra_name,
        xy=xy,
        xytext=xytext,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        arrowprops=dict(arrowstyle="->", color='black', linewidth=1.0, relpos=relpos),
        bbox=dict(
            pad=0.25, facecolor="white", edgecolor="black", boxstyle='round', alpha=0.7
        ),
        fontsize=annotation_fontsize
    )


def compute_test_statistics(
        test_loader: L.LightningDataLoader,
        operator_modules: List[OperatorModuleWrapper],
    ) -> Dict:

    for operator_module in operator_modules:
        operator_module.compute_full_metrics = True 

    def key_fn(params: OperatorModuleParams):
        return (
            params.algebra, params.directparam_algebra_symmetric,
            params.transported_index, params.transported_operations
        )
    
    # WILL BE FLATTENED
    # {operator_module.params: {literal_n: [{loss: , acc, ..}, ...]} }
    results_a = defaultdict(lambda: defaultdict(list))

    # Average results_a over all literal_ns and within literal_ns
    # {params: {iou: 0.5, ...}}
    results_b = defaultdict(dict)

    # WILL BE FLATTENED
    # {operator_module.params: {law_n: [{loss: , acc, ..}, ...]} }
    results_c = defaultdict(lambda: defaultdict(list))

    # {params: {'num_sat': int, 'num_sat_for_property_type': {commutativity: 1, ...}}
    results_metadata = {}
    
    # Results a & c
    for batch in tqdm.tqdm(test_loader):
        for k, v in batch.items():
            batch[k] = v.cuda()
        
        for operator_module in operator_modules:
            param_key = key_fn(operator_module.params)
            is_directparam_algebra = (operator_module.params.algebra == 'directparam')

            prefix = ''
            # prefix = '' if is_directparam_algebra else 'induced_'
            # operator_module.params.compute_induced_metrics = True

            step_vars = operator_module._step_variables(batch, 1)

            for literal_n in range(2, MAX_LITERAL_N + 1):
                literal_n_vars = step_vars[f'literal_n:{literal_n}']
                
                maybe_iou = literal_n_vars.get(f'{prefix}iou')
                results_a[param_key][literal_n].append({
                    'loss': literal_n_vars[f'{prefix}loss'].item(),
                    'accuracy': literal_n_vars[f'{prefix}accuracy'].item(),
                    'iou': maybe_iou.item() if maybe_iou is not None else None,
                })
                
                if literal_n != 10:
                    continue

                law_n_max = len(literal_n_vars['equivalent_iou_to_pred'])
                for law_n in range(law_n_max):
                    equiv_iou = literal_n_vars['equivalent_iou_to_pred'][law_n]
                    equiv_acc = literal_n_vars['equivalent_accuracy_to_pred'][law_n]
                    assert equiv_iou['law_n'] == law_n
                    assert equiv_acc['law_n'] == law_n
                    
                    iou_val, acc_val = equiv_iou['val'], equiv_acc['val']

                    results_c[param_key][law_n].append({
                        'equivalent_iou_to_pred': (
                            iou_val.item() if iou_val is not None else None
                        ),
                        'equivalent_accuracy_to_pred': acc_val
                    })

    # Results b
    for operator_module in operator_modules:
        param_key = key_fn(operator_module.params)
        def inner_average(l: List[Dict[str, float]]) -> Dict[str, float]:
            return {
                k: np.mean([d[k] for d in l if (d[k] is not None)])
                for k in l[0].keys()
            }
                
        def get_average(k):
            return np.mean([inner_average(r)[k] for r in results_a[param_key].values()])
        
        results_b[param_key] = {
            'loss': get_average('loss'),
            'accuracy': get_average('accuracy'),
            'iou': get_average('iou'),
        }
            
    # Results metadata
    for operator_module in operator_modules:
        param_key = key_fn(operator_module.params)
        num_sat = algebra_module.num_satisfied_properties(operator_module.algebra)
        num_sat_for_property_type = algebra_module.num_satisfied_for_property_type(
            operator_module.algebra
        )
        results_metadata[param_key] = {
            'num_sat': num_sat,
            'num_sat_for_property_type': num_sat_for_property_type
        }
    
    # {operator_module.params: {lol_n: {loss: , acc: , loss_std: , acc_std: }} }
    def do_flatten(results_a_or_c):
        flattened = {}
        
        # lol = literal or law
        for params_key, lol_n_results in results_a_or_c.items():
            flattened[params_key] = {}
            for lol_n, results_a in lol_n_results.items():
                flattened[params_key][lol_n] = update_dict = {}
                for metric in results_a[0].keys():
                    vals = torch.tensor(
                        [r[metric] for r in results_a if r[metric] is not None]
                    )
                    update_dict[metric] = vals.mean().item()
                    update_dict[metric + '_median'] = vals.median().item()
                    update_dict[metric + '_std'] = vals.std().item()
                    update_dict[metric + '_25%'] = vals.kthvalue(
                        int(0.2 * len(vals))
                    ).values.item()
                    update_dict[metric + '_75%'] = vals.kthvalue(
                        int(0.8 * len(vals))
                    ).values.item()

        
        return flattened
    
    file_utils.write_pickle(results_a_path, do_flatten(results_a))
    file_utils.write_pickle(results_b_path, results_b)
    file_utils.write_pickle(results_c_path, do_flatten(results_c))
    file_utils.write_pickle(results_metadata_path, results_metadata)

def load_transported_operator_module(
        test_params: TestParams, transported_index: int, latent: LatentModule
    ) -> OperatorModuleWrapper:
    
    run_name = f'transported{transported_index}'
    
    path = dirs.out_path(
        test_params.project, test_params.group, run_name, 'checkpoints', 'best.ckpt'
    )

    mod = OperatorModuleWrapper.load_from_checkpoint(path, latent_module=latent).cuda()
    mod.eval()
    
    return mod

def load_directparam_operator_module(
        test_params: TestParams, symmetric: bool, latent: LatentModule
    ) -> OperatorModuleWrapper:
    
    run_name = 'directparam_symmetric' if symmetric else 'directparam'
    
    path = dirs.out_path(
        test_params.project, test_params.group, run_name, 'checkpoints', 'best.ckpt'
    )

    mod = OperatorModuleWrapper.load_from_checkpoint(path, latent_module=latent).cuda()
    mod.eval()
    
    return mod
    

if __name__ == "__main__":
    run()

