from typing import Any, Dict, Optional, Type, Union

import torchvision
import torch
import torch.utils.data as data

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class MnistDataset(pl.LightningDataModule):
    def __init__(self, download_to: str, batch_size: int):
        super().__init__()
        self.download_to = download_to
        self.batch_size = batch_size
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # once-time tasks like downloading, generating
        torchvision.datasets.MNIST(self.download_to, train=True, download=True)
        torchvision.datasets.MNIST(self.download_to, train=False, download=True)

    def setup(self, stage=None): 
        if stage is None or stage == "fit":
            self.train =  torchvision.datasets.MNIST(self.download_to, train=True, transform=self.transform)
            self.val   = torchvision.datasets.MNIST(self.download_to, train=False, transform=self.transform)
        if stage is None or stage == "test":
            self.test  = torchvision.datasets.MNIST(self.download_to, train=False, transform=self.transform)

    def train_dataloader(self):
        # TODO: pin_memory=True might be faster
        return data.DataLoader(self.train, batch_size=self.batch_size, pin_memory=False, num_workers=3, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size=self.batch_size, pin_memory=False, num_workers=3)

    def test_dataloader(self):
        return data.DataLoader(self.test, batch_size=self.batch_size, pin_memory=False)