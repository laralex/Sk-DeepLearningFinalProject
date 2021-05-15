from typing import Any, Dict, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
import torchmetrics

import pytorch_lightning as pl

class MnistClassifier(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            optimizer: str = 'SGD',
            optimizer_kwargs: Dict[str, Any] = { 'lr': 0.001 },
            scheduler: str = 'StepLR',
            scheduler_kwargs: Dict[str, Any] = {'step_size': 10 },
            ):

        super().__init__()

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.net = SimpleConvNet(in_channels, num_classes)
        self.criterion = nn.NLLLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        preds_proba = torch.softmax(preds, dim=1)
        loss = self.criterion(preds, target)
        self.log("loss/train", loss, prog_bar=True, logger=False)

        return {'loss' : loss, 'preds_proba' : preds_proba, 'target' : target}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        preds_proba = torch.softmax(preds, dim=1)

        loss = self.criterion(preds, target)
        self.log("loss/val", loss, prog_bar=True, logger=False)
        return {'loss' : loss, 'preds_proba' : preds_proba, 'target' : target}

    def training_step_end(self, outputs):
        self.log("loss/train", outputs["loss"].mean(dim=0), prog_bar=True)

        self.train_accuracy(outputs['preds_proba'], outputs['target'])

        self.log('accuracy/train', self.train_accuracy, prog_bar=True)

    def validation_step_end(self, outputs):
        self.log("loss/val", outputs["loss"].mean(dim=0), prog_bar=True)

        self.val_accuracy(outputs['preds_proba'], outputs['target'])
        self.log('accuracy/val', self.train_accuracy, prog_bar=True)

    def configure_optimizers(self):
        OptimizerClass = getattr(torch.optim, self.optimizer)
        SchedulerClass = getattr(torch.optim.lr_scheduler, self.scheduler)
        opt = OptimizerClass(self.parameters(), **self.optimizer_kwargs)
        sch = SchedulerClass(opt, **self.scheduler_kwargs)
        return [opt], [sch]

# Simple example for testing from https://github.com/pytorch/examples/blob/master/mnist/main.py
class SimpleConvNet(nn.Module):
    def __init__(self, c_in=1, out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        outputs = F.log_softmax(x)
        return outputs