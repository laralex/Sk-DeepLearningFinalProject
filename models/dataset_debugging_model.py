import pytorch_lightning as pl
from typing import Optional
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

    def forward(self, x):
        x = x.squeeze()
        if self.is_two_dim:
            assert x.shape == (self.batch_size, self.expected_signal_size, self.num_blocks), "Expecting [bs, h, w] size"
        else:
            assert x.shape == (self.batch_size, self.expected_signal_size), "Expecting [bs, seq] size"
        return x

    def training_step(self, batch, batch_idx):
        print(f'training batch {batch.sum():.3f}')
        distorted, target = batch
        assert distorted.shape == target.shape
        self.forward(distorted)
        loss = 0.0
        return loss

    def validation_step(self, batch, batch_idx):
        print(f'val batch {batch.sum():.3f}')
        distorted, target = batch
        assert distorted.shape == target.shape
        self.forward(distorted)
        loss = 0.0
        return loss

    def test_step(self, batch, batch_idx):
        print(f'test batch {batch_idx}')
        distorted, target = batch
        assert distorted.shape == target.shape
        self.forward(distorted)
        loss = 0.0
        return loss

    def configure_optimizers(self):
        return [], []