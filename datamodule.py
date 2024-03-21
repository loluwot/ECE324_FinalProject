from torch.utils.data import DataLoader, IterableDataset, Dataset
import torch
import torchvision.io as io

import numpy as np
import cv2

from pathlib import Path
import lightning.pytorch as pl

MEANS = [0.5022, 0.4599, 0.3994]
STDS = [0.2554, 0.2457, 0.2504]

def normalize(im):
    # means = torch.tensor([0.5022, 0.4599, 0.3994])
    # stds = torch.tensor([0.2554, 0.2457, 0.2504])
    
    means = np.array(MEANS)
    stds = np.array(STDS)
    im -= means[..., None, None]
    im /= stds[..., None, None]
    return im
    
def unnormalize(im, tensor=False):
    f = (lambda x: torch.tensor(x).to(im)) if tensor else np.array
    means = f(MEANS)
    stds = f(STDS)
    im *= stds[..., None, None]
    im += means[..., None, None]
    return im

class AFHQDataset(Dataset):
    def __init__(self, path, im_size=512, use_normalize=False):
        super().__init__()
        self.image_filenames = list(Path(path).glob('**/*.jpg'))
        self.im_size = im_size
        self.normalize = normalize if use_normalize else (lambda x: x)
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # return normalize(io.read_image(str(self.image_filenames[idx].resolve())) / 255)
        return self.normalize(cv2.resize(cv2.imread(str(self.image_filenames[idx].resolve()), cv2.IMREAD_COLOR), (self.im_size, self.im_size), interpolation=cv2.INTER_AREA).transpose(-1, 0, 1) / 255)

class AFHQDataModule(pl.LightningDataModule):
    def __init__(self, base_dataset, num_workers=1, batch_size=32):
        super().__init__()
        self.base_dataset = base_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        
    def train_dataloader(self):
        return DataLoader(
            dataset = self.base_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
        )