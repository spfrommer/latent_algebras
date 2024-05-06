from __future__ import annotations
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from jaxtyping import Float

import tqdm
from contextlib import contextmanager


def device():
    return torch.device('cuda')


def gpu_n():
    return 1 if torch.cuda.is_available() else 0


def np(tensor):
    return tensor.detach().cpu().numpy()


def detach_all_values(d: Dict):
    return {k: v.detach() for k, v in d.items()}


def random_batch_subset(tensor: Tensor, subset_n: int) -> Tensor:
    indices = torch.randperm(tensor.shape[0])[:subset_n]
    return tensor[indices]


def batch_list(tensor: Tensor, unsqueeze_batch=True) -> List[Tensor]:
    return [(t.unsqueeze(0) if unsqueeze_batch else t) for t in tensor]


def cat_select(tensors: List[tuple[Tensor]], index: int) -> Tensor:
    return torch.cat([t[index] for t in tensors], dim=0)


def shuffle_tensors(tensors: List[Tensor]) -> List[Tensor]:
    shuffle_inds = torch.randperm(tensors[0].shape[0])
    return [t[shuffle_inds] for t in tensors]


def split_tensors_into_datasets(
        tensors: List[Float[Tensor, 'b ...']], train_proportion=0.7, val_proportion=0.2
    ) -> Tuple[Dataset, Dataset, Dataset]:

    data_n = tensors[0].shape[0]
    train_size = int(train_proportion * data_n)
    val_size = int(val_proportion * data_n)

    train_tensors = [t[:train_size] for t in tensors]
    val_tensors = [t[train_size : train_size + val_size] for t in tensors]
    test_tensors = [t[train_size + val_size :] for t in tensors]

    return (
        TensorDataset(*train_tensors),
        TensorDataset(*val_tensors),
        TensorDataset(*test_tensors),
    )


# Adds a decorator to all methods on an object
# Good for automatically enforcing pytorch tensor type checks
# Adapted from: https://stackoverflow.com/questions/6307761/how-to-decorate-all-functions-of-a-class-without-typing-it-over-and-over-for-eac
def for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.
    From https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
    '''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def fetch_dataset(dataset, fetch_n):
    signals, targets = next(iter(DataLoader(dataset, batch_size=fetch_n)))
    signals, targets = signals.to(device()), targets.to(device())
    return signals, targets


def fetch_dataloader(dataloader, fetch_n, do_tqdm=False):
    if do_tqdm:
        pbar = tqdm.tqdm(total=fetch_n)

    i = 0
    for (signals, targets) in dataloader:
        for (signal, target) in zip(signals, targets):
            yield signal.to(device()).unsqueeze(0), target.to(device()).unsqueeze(0)
            if do_tqdm:
                pbar.update(1)
            i += 1

            if i >= fetch_n:
                if do_tqdm:
                    pbar.close()
                return

    if do_tqdm:
        pbar.close()


def fetch_dataloader_batch(dataloader, fetch_n):
    signals, targets = [], []
    for (signal, target) in fetch_dataloader(dataloader, fetch_n):
        signals.append(signal)
        targets.append(target)

    return torch.cat(signals, dim=0), torch.cat(targets, dim=0)


def check_dataloader_deterministic(dataloader):
    s1, t1 = next(iter(fetch_dataloader(dataloader, 1)))
    s2, t2 = next(iter(fetch_dataloader(dataloader, 1)))
    assert (s1 == s2).all() and (t1 == t2)