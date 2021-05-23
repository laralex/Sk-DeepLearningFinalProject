import pytorch_lightning as pl
from typing import Optional
import torch
from torch import nn
class DatasetDebuggingModel(pl.LightningModule):
    def __init__(self,
        is_two_dim: bool,
        expected_signal_size: int,
        num_blocks: Optional[int],
        batch_size: int):

        super().__init__()
        self.is_two_dim = is_two_dim
        self.expected_signal_size = expected_signal_size
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.net = nn.parameter.Parameter(torch.tensor([0.0]), requires_grad=True)

    def forward(self, x):
        x = x.squeeze()
        if self.is_two_dim:
            assert x.shape == (self.batch_size, self.expected_signal_size, self.num_blocks), "Expecting [bs, h, w] size"
        else:
            assert x.shape == (self.batch_size, self.expected_signal_size), "Expecting [bs, seq] size"
        return x * self.net

    def training_step(self, batch, batch_idx):
        distorted, target = batch
        print(f'train batch {distorted.sum():.3f}')
        assert distorted.shape == target.shape
        loss = self.forward(distorted).sum()
        return loss

    def validation_step(self, batch, batch_idx):
        distorted, target = batch
        print(f'val batch {distorted.sum():.3f}')
        assert distorted.shape == target.shape
        loss = self.forward(distorted).sum()
        return loss

    def test_step(self, batch, batch_idx):
        distorted, target = batch
        print(f'test batch {distorted.sum():.3f}')
        assert distorted.shape == target.shape
        loss = self.forward(distorted).sum()
        return loss

    def configure_optimizers(self):
        return [torch.optim.SGD(self.parameters(), lr=.1)], []