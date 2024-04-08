from torch.utils.data import DataLoader, IterableDataset, Dataset
import torch
import torchvision.io as io

import numpy as np
import cv2

from pathlib import Path
import lightning.pytorch as pl

import random

MEANS = [0.5022, 0.4599, 0.3994]
STDS = [0.2554, 0.2457, 0.2504]

def normalize(im, tensor=False):
    # means = torch.tensor([0.5022, 0.4599, 0.3994])
    # stds = torch.tensor([0.2554, 0.2457, 0.2504])
    f = (lambda x: torch.tensor(x).to(im)) if tensor else np.array
    means = f(MEANS)
    stds = f(STDS)
    im -= means[..., None, None]
    im /= stds[..., None, None]
    return im
    
def unnormalize(im, tensor=False):
    f = (lambda x: torch.tensor(x).to(im)) if tensor else np.array
    means = f(MEANS)[..., None, None]
    stds = f(STDS)[..., None, None]

    if im.ndim == 4:
        means = means.unsqueeze(0)
        stds = stds.unsqueeze(0)

    im = im * stds
    im = im + means
    im = im.clamp(0., 1.)
    return im

class AFHQDataset(Dataset):
    def __init__(self, path, im_size=512, use_normalize=False, dtype=None):
        super().__init__()
        self.image_filenames = list(Path(path).glob('**/*.jpg'))
        self.im_size = im_size
        self.normalize = normalize if use_normalize else (lambda x: x)
        self.dtype = np.float32 if dtype is None else dtype
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        return self.normalize(cv2.resize(cv2.imread(str(self.image_filenames[idx].resolve()), cv2.IMREAD_COLOR), (self.im_size, self.im_size), interpolation=cv2.INTER_AREA)[..., ::-1].transpose(-1, 0, 1) / 255).astype(self.dtype)

class AFHQDataModule(pl.LightningDataModule):
    def __init__(self, base_dataset, batch_size, num_workers=1, shuffle=True):
        super().__init__()        
        self.base_dataset = base_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def train_dataloader(self):
        return DataLoader(
            dataset = self.base_dataset,
            batch_size = self.batch_size | self.hparam.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = self.shuffle
        )