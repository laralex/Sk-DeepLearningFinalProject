import torch
import pytorch_lightning as pl
from torch import nn, mean, optim
from copy import deepcopy

from typing import Any, Dict
from torch.optim.optimizer import Optimizer
import torchmetrics
import models
from models.loss_functions import EVM
from models.metrics import BERMetric, QFactor
from data.transform_1d_2d import transform_to_1d
from math import sqrt

# import plotly.express as px
# from plotly.offline import plot
import matplotlib.pyplot as plt

class ConvNet_regressor(pl.LightningModule):
    def __init__(
        self, 
        seq_len: int,
        pulse_width: float,
        z_end: float,
        dim_t: int,
        decision_level: float,
        
        in_features: int,
        bias = False,

        optimizer:str = 'Adam',
        optimizer_kwargs: Dict[str, Any] = {'lr':1e-4},
        scheduler: str = 'StepLR',
        scheduler_kwargs: Dict[str, Any] = {'step_size':10},
        criterion: str = 'EVM',
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
        
        self.loss_type = criterion
        
        if criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif criterion == 'EVM':
            self.criterion = EVM()
        
        self.seq_len = seq_len
        self.pulse_width = pulse_width
        self.z_end = z_end
        self.dim_t = dim_t
        
        self.t_end = ((seq_len - 1)//2 + 1) * pulse_width      
        tMax = self.t_end + 4*sqrt(2*(1 + z_end**2))
        tMin = -tMax
        dt = (tMax - tMin) / dim_t
        self.t = torch.linspace(tMin, tMax-dt, dim_t)
        self.t_window = [torch.abs(self.t + self.t_end).argmin(),
                         torch.abs(self.t - self.t_end).argmin()]
        
        self.ber = BERMetric(decision_level=decision_level,
                     pulse_number=seq_len,
                     pulse_width=pulse_width,
                     t=self.t+self.t_end,
                     t_window=self.t_window)


    def forward(self, x):
        if len(x.shape) == 4 and x.shape[0] == 1:
            x = x.squeeze(dim=0)
        x = x.permute(2, 0, 1)
        real = x.real
        imag = x.imag
        real, imag = self.net(real, imag)
        return (real + 1j*imag).unsqueeze(1).permute(1, 2, 0)


    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        
        preds = transform_to_1d(preds)
        target = transform_to_1d(target)
        
        if self.loss_type == 'MSE':
            loss_real = self.criterion(preds.real, target.real)
            loss_imag = self.criterion(preds.imag, target.imag)
            loss = loss_real + loss_imag
        elif self.loss_type == 'EVM':
            loss = self.criterion(preds, target)
            
        self.log("loss_train", loss, prog_bar = False, logger = True)
        return {"loss": loss, "preds": preds, "target": target}


    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)

        preds = transform_to_1d(preds)
        target = transform_to_1d(target)
        
        if self.loss_type == 'MSE':
            loss_real = self.criterion(preds.real, target.real)
            loss_imag = self.criterion(preds.imag, target.imag)
            loss = loss_real + loss_imag
        elif self.loss_type == 'EVM':
            loss = self.criterion(preds, target)
        
        self.log("loss_val", loss, prog_bar = False, logger = True)
        return {'preds':preds , 'target': target}


    def configure_optimizers(self):
        OptimizerClass = getattr(optim, self.optimizer)
        SchedulerClass = getattr(optim.lr_scheduler, self.scheduler)
        opt = OptimizerClass(self.parameters(), **self.optimizer_kwargs)
        sch = SchedulerClass(opt, **self.scheduler_kwargs)

        return [opt], [sch]


    def training_epoch_end(self, outputs):
        preds = torch.cat([r['preds'] for r in outputs], dim=0)
        target = torch.cat([r['target'] for r in outputs], dim=0)

        self.ber.update(preds.squeeze(1), target.squeeze(1))
        ber_value = self.ber.compute()
        q_factor = QFactor(ber_value)
        self.log('Q_factor_train', q_factor, prog_bar = True, logger = True)
        
        # nt1 = torch.abs(self.t - 0.5 * self.pulse_width).argmin()
        # nt2 = torch.abs(self.t - 8.5 * self.pulse_width).argmin()
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(self.t[nt1:nt2], target[0,0,nt1:nt2].cpu().real, label='Target')
        # ax.plot(self.t[nt1:nt2], preds[0,0,nt1:nt2].detach().cpu().real, label='Predicted')
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Re(E)')
        # self.logger.experiment.add_figure('prediction_train', fig, global_step=self.current_epoch)

    def validation_epoch_end(self, outputs):
        preds = torch.cat([r['preds'] for r in outputs], dim=0)
        target = torch.cat([r['target'] for r in outputs], dim=0)

        self.ber.update(preds.squeeze(1), target.squeeze(1))
        ber_value = self.ber.compute()
        q_factor = QFactor(ber_value)
        self.log('Q_factor_val', q_factor, prog_bar = True, logger = True)
        
        nt1 = torch.abs(self.t - 0.5 * self.pulse_width).argmin()
        nt2 = torch.abs(self.t - 8.5 * self.pulse_width).argmin()
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.t[nt1:nt2], target[0,0,nt1:nt2].cpu().real, label='Target')
        ax.plot(self.t[nt1:nt2], preds[0,0,nt1:nt2].detach().cpu().real, label='Predicted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Re(E)')
        self.logger.experiment.add_figure('prediction_val', fig, global_step=self.current_epoch)
        # fig = px.line(x=self.t[nt1:nt2], y=target[0,0,nt1:nt2].real, title='Target', labels={'x':'Time', 'y':'real E'})
        # plot(fig)
        # fig = px.line(x=self.t[nt1:nt2], y=preds[0,0,nt1:nt2].detach().real, title='Predicted', labels={'x':'Time', 'y':'real E'})
        # plot(fig)

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
        
        self.imag_model = deepcopy(self.real_model)
        self.real_fc = nn.Linear(in_features, in_features)
        self.imag_fc = nn.Linear(in_features, in_features)
        
        
    def forward(self, real, imag):
        real = self.real_model(real)
        real = mean(real, dim = 1)
        real = self.real_fc(real)
        
        imag = self.imag_model(imag)
        imag = mean(imag, dim = 1)
        imag = self.imag_fc(imag)
        
        return real, imag