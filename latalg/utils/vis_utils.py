from __future__ import annotations

from matplotlib.figure import Figure
import torch
import numpy as np

import matplotlib.pyplot as plt
import wandb

from . import torch_utils as TU


def imshow(X):
    if X.shape[0] == 1:
        X = X[0]
    if torch.is_tensor(X):
        X = TU.np(X)
    X = np.moveaxis(X, 0, -1)

    plt.figure()
    plt.imshow(X)
    plt.axis('off')
    plt.show()


def histshow(X):
    X = X.flatten().tolist()
    plt.figure()
    plt.hist(X, 50)
    plt.show()


def render_and_close_figure(fig: Figure) -> wandb.Image:
    image = wandb.Image(fig)
    plt.close(fig)
    return image

def figure_to_numpy(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)