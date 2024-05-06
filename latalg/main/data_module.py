import lightning as L
import tqdm
import torch
import wandb

from dataclasses import dataclass
import click

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from latalg.utils import dirs, file_utils
from latalg.utils import torch_utils as TU
from latalg.data.voronoi import pad_points_to_max_len


INR_DATA_DIR = dirs.data_path('inr')

@dataclass
class DataParams:
    batch_size: int

click_options = [
    click.option('--batch_size', default=4),
]


class CompleteINRDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 4, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Length is number of subdirectories in INR_DATA_DIR
        self.length = file_utils.num_files(INR_DATA_DIR)
        
    def setup(self, stage: str = None) -> None:
        train_proportion = 0.8 # Must be same as in train_latent_model.py
        val_proportion = 0.1
        train_index = int(train_proportion * self.length)
        val_index = int(val_proportion * self.length)
        self.train_dataset = CompleteINRDataset(
            data_dir=INR_DATA_DIR,
            start_index=0,
            end_index=train_index,
        )
        self.val_dataset = CompleteINRDataset(
            data_dir=INR_DATA_DIR,
            start_index=train_index,
            end_index=train_index + val_index,
        )
        self.test_dataset = CompleteINRDataset(
            data_dir=INR_DATA_DIR,
            start_index=train_index + val_index,
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
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )
    

class CompleteINRDataset(Dataset):
    """Start inclusive, stop not inclusive."""
    # TODO: Needs to have "filter max inshape/outshape points feature"
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
        abs_index = index + self.start_index
        dir_original = dirs.path(self.data_dir, f'{abs_index}', 'inr_original')
        dir_latent = dirs.path(self.data_dir, f'{abs_index}', 'inr_latent')
        
        # query_points = torch.load(dirs.path(dir_original, 'query_points.pt'))
        # query_labels = torch.load(dirs.path(dir_original, 'query_labels.pt'))
        shape = torch.load(dirs.path(dir_original, 'shape.pt'))
        inr_params_matrix = torch.load(dirs.path(dir_original, 'inr_params_matrix.pt'))
        inr_params_matrix.requires_grad = False
        
        encoded_latent = torch.load(dirs.path(dir_latent, 'encoded_vector.pt'))
        
        return {
            # 'query_points': query_points,
            # 'query_labels': query_labels,
            'shape_inshape_points': pad_points_to_max_len(shape.inshape_points),
            'shape_outshape_points': pad_points_to_max_len(shape.outshape_points),
            'inr_params_matrix': inr_params_matrix,
            'encoded_latent': encoded_latent,
        }
