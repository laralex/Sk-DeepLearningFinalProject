from typing import Any, Dict, Optional, Type, Union

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class SplitStepGenerator(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # some one-time stuff like downloading, generating
        pass

    def setup(self, stage: Optional[str] = None):
        # transforming, splitting
        if stage is None or stage == "fit":
            self.train = torch.zeros((1000, 1, 1))
            self.val = torch.zeros((1000, 1, 1))
        if stage is None or stage == "test":
            self.test = torch.zeros((1000, 1, 1))

    def train_dataloader(self):
        # TODO: pin_memory=True might be faster
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, pin_memory=False)