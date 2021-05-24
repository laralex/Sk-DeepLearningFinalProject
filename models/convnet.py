import pytorch_lightning as pl
from torch import nn, mean, optim
from copy import deepcopy

from typing import Any, Dict
from torch.optim.optimizer import Optimizer
import torchmetrics
import models
from models.loss_functions import EVM
from data.transform_1d_2d import *

class ConvNet_regressor(pl.LightningModule):
    def __init__(
        self, 
        in_features: int,
        bias = False, 

        optimizer:str = 'Adam',
        optimizer_kwargs: Dict[str, Any] = {'lr':1e-4},
        scheduler: str = 'StepLR',
        scheduler_kwargs: Dict[str, Any] = {'step_size':10},
        criterion: str = 'EVM'
        ):
        '''
        in_features (int) - number of input features in model
        bias (bool) - whether to use bias in linear layers or not.
        
        optimizer (str) - name of optimizer (ex. "Adam", "SGD")
        optimizer_kwargs (dict) - parameters of optimizer (ex. {'lr':1e-4})
        scheduler (str) - name of scheduler that will be used
        scheduler_kwargs (dict) - parameters of scheduler
        criterion (str) - Loss function that will be used for training. "MSE" or "EVM"
        '''
        

        super().__init__()

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.net = ConvNet(in_features, bias)

        if criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif criterion == 'EVM':
            self.criterion = EVM()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()


    def forward(self, x):
        x = x.permute(2, 0, 1)
        real = x.real
        imag = x.imag
        real, imag = self.net(real, imag)
        return (real + 1j*imag).unsqueeze(1).permute(1, 2, 0)


    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        
        loss = self.criterion(transform_to_1d(preds.unsqueeze(1)), transform_to_1d(target.unsqueeze(1)))
        self.log("loss_train", loss, prog_bar = False, logger = True)
        return loss


    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)

        loss = self.criterion(transform_to_1d(preds.unsqueeze(1)), transform_to_1d(target.unsqueeze(1)))
        self.log("loss_val", loss, prog_bar = False, logger = True)


    def configure_optimizers(self):
        OptimizerClass = getattr(optim, self.optimizer)
        SchedulerClass = getattr(optim.lr_scheduler, self.scheduler)
        opt = OptimizerClass(self.parameters(), **self.optimizer_kwargs)
        sch = SchedulerClass(opt, **self.scheduler_kwargs)

        return [opt], [sch]


    # def training_epoch_end(self, outputs):
    #     preds = torch.cat([r['preds'] for r in outputs], dim=0)
    #     targets = torch.cat([r['target'] for r in outputs], dim=0)
    #     self.train_accuracy(preds, targets)
    #     self.log('accuracy_train', self.train_accuracy, prog_bar = True, logger = True)
    

    # def validation_epoch_end(self, outputs):
    #     preds = torch.cat([r['preds'] for r in outputs], dim=0)
    #     targets = torch.cat([r['target'] for r in outputs], dim=0)

    #     self.val_accuracy(preds, targets)
    #     self.log('accuracy_val', self.val_accuracy, prog_bar = True, logger = True)

    def get_configuration(self):
        '''
        Returns dict of str with current configuration of the model.
        Can be used for Logger.
        '''
        configuration = {
            'activation': self.net.activation_name, 
            'criterion': str(self.criterion.__repr__())[:-2],
            'optimizer': self.optimizer,
            'optimizer_param': str(self.optimizer_kwargs)[1:-1], 
            'scheduler': self.scheduler,
            'scheduler_param': str(self.scheduler_kwargs)[1:-1]
        }
        return configuration


class ConvNet(nn.Module):
    '''
    Model with 1D-CNN architecture
    Inspired from https://discuss.pytorch.org/t/cnn-architecture-for-short-time-series-data/99814
    '''
    
    def __init__(self, in_features: int, bias=False, activation='ReLU'):
        '''
        @num_classes - number of features in input vector
        @bias - whether to use bias for convolutional layers or not
        @activation - the activation function used after each conv layer
        '''
        super(ConvNet, self).__init__()
        
        # Activation function
        self.activation_name = activation
        ActivationClass = getattr(nn, activation)
        
        self.real_model = nn.Sequential(
                    nn.Conv1d(1, 3, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.BatchNorm1d(3),
                    ActivationClass(),
                    nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                    nn.Dropout(0.3),
            
                    nn.Conv1d(3, 5, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.BatchNorm1d(5),
                    ActivationClass(),
                    nn.Dropout(0.3),
            
                    nn.Conv1d(5, 5, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.BatchNorm1d(5),
                    ActivationClass(),
            
                    nn.Dropout(0.3)
                    )
        
        self.flatten = nn.Flatten()
        self.imag_model = deepcopy(self.real_model)
        self.real_fc = nn.Linear(in_features, in_features)
        self.imag_fc = nn.Linear(in_features, in_features)
        
        
    def forward(self, real, imag):
        real = self.real_model(real)
        real = mean(real, dim = 1)
        # real = self.flatten(real)
        real = self.real_fc(real)
        
        imag = self.imag_model(imag)
        imag = mean(imag, dim = 1)
        # imag = self.flatten(imag)
        imag = self.imag_fc(imag)
        
        return real, imag