import pytorch_lightning as pl

class DatasetDebuggingModel(pl.LightningModule):
    def __init__(self, is_two_dim: bool):
        super().__init__()
        self.is_two_dim = is_two_dim

    def forward(self, x):
        if self.is_two_dim:
            assert len(x.shape) == 4, "Expecting [bs, ch, h, w] size"
        else:
            assert len(x.shape) == 3, "Expecting [bs, ch, seq] size"
        assert x.shape[1] == 1, "Number of channels should be 1"
        return x

    def training_step(self, batch, batch_idx):
        distorted, target = batch
        assert distorted.shape == target.shape
        self.forward(distorted)
        loss = 0.0
        return loss

    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return [], []