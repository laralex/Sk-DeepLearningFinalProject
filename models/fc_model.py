from sys import prefix
from typing import Any, Dict
import torch
from torch import nn, per_tensor_affine
import pytorch_lightning as pl
from torch.nn.modules.activation import ReLU
from torch.optim.optimizer import Optimizer
import torchmetrics
from typing import Any, Dict, Optional, Type, Union
import models
from models import loss_functions
from copy import deepcopy
from math import sqrt
from models.metrics import BERMetric, QFactor
from data.transform_1d_2d import transform_to_1d
import matplotlib.pyplot as plt



class FC_regressor(pl.LightningModule):
    def __init__(
        self, 

        # params for BER
        seq_len: int,
        pulse_width: float,
        z_end: float,
        dim_t: int,
        decision_level: float,

        in_features: int,
        layers: int, 
        sizes:list = None, 
        bias = False, 

        optimizer:str = 'Adam',
        optimizer_kwargs: Dict[str, Any] = {'lr':1e-4},
        scheduler: str = 'StepLR',
        scheduler_kwargs: Dict[str, Any] = {'step_size':10},
        criterion: str = 'MSE',
        activation: str = 'ReLU',
        use_batchnorm: bool = False,
        dropout: float = 0.0
        # TODO activation kwargs

        ):
        '''
        in_features (int) - number of input features in model
        layers (int) - number of layers in model
        sizes (list) - number of output features for each layer. 
            If sizes == None : in_features will be used instead.
        bias (bool) - whether to use bias in linear layers or not.
        
        optimizer (str) - name of optimizer (ex. "Adam", "SGD")
        optimizer_kwargs (dict) - parameters of optimizer (ex. {'lr':1e-4})
        scheduler (str) - name of scheduler that will be used
        scheduler_kwargs (dict) - parameters of scheduler
        criterion (str) - Loss function that will be used for training. "MSE" or "EVM"
        activation (str) - class of activation function from torch.nn.
        use_batchnorm: (bool) - if True - adds batchnorm1D after each linear layer
        dropout: (float) - if not 0.0 - adds dropout, after each activation
        '''
        

        super().__init__()

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.net = FC_model(in_features, layers, sizes, bias, activation, use_batchnorm, dropout)


        ############################
        if criterion == 'MSE':
            self.loss_type = "MSE"
            self.criterion = nn.MSELoss()
        ############################
        else:
            self.loss_type = criterion
            self.criterion = getattr(loss_functions ,criterion)()

        self.__ber_param_init(seq_len, pulse_width, z_end, dim_t, decision_level)


        #self.train_accuracy = torchmetrics.Accuracy()
        #self.val_accuracy = torchmetrics.Accuracy()


    def __ber_param_init(self, seq_len, pulse_width, z_end, dim_t, decision_level):
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
                     t=self.t + self.t_end + 0.5*pulse_width,
                     t_window=self.t_window)
        


    def forward(self, x):
        # example input shape [1,1,4096,32]
        if len(x.shape) == 4 and x.shape[0] ==1:
            # output shape [1,4096,32]
            x = x.squeeze(dim =0)
        # output shape [32,1, 4096]
        x = x.permute(2,0,1)

        # both shapes: [32,1, 4096]
        real = x.real
        imag = x.imag
        
        # both shapes: [32,1, 4096]
        real, imag= self.net(real,imag)

        #return [1,4096,32] shape vector
        return (real + 1j*imag).permute(1,2,0)


    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)

        preds = transform_to_1d(preds)
        target = transform_to_1d(target)

        #loss = self.criterion(preds, target)

        if self.loss_type == 'MSE':
            loss_real = self.criterion(preds.real, target.real)
            loss_imag = self.criterion(preds.imag, target.imag)
            loss = loss_real + loss_imag
        else: 
            loss  = self.criterion(preds, target)

        self.log("loss_train", loss, prog_bar = False, logger = True)
        return {"loss":loss, "preds": preds, "target": target}


    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)

        preds = transform_to_1d(preds)
        target = transform_to_1d(target)

        #loss = self.criterion(preds, target)
        if self.loss_type == 'MSE':
            loss_real = self.criterion(preds.real, target.real)
            loss_imag = self.criterion(preds.imag, target.imag)
            loss = loss_real + loss_imag
        else: 
            loss = self.criterion(preds, target)

        self.log("loss_val", loss, prog_bar = False, logger= True)

        return {'preds':preds , 'target': target}


    def configure_optimizers(self):
        OptimizerClass = getattr(torch.optim, self.optimizer)
        SchedulerClass = getattr(torch.optim.lr_scheduler, self.scheduler)
        opt = OptimizerClass(self.parameters(), **self.optimizer_kwargs)
        sch = SchedulerClass(opt, **self.scheduler_kwargs)
        
        return [opt], [sch]


    def training_epoch_end(self, outputs):
        preds = torch.cat([r['preds'] for r in outputs], dim=0)
        targets = torch.cat([r['target'] for r in outputs], dim=0)

        self.ber.update(preds.squeeze(1), targets.squeeze(1))
        ber_value = self.ber.compute()
        q_factor = QFactor(ber_value)

        #self.train_accuracy(preds, targets)
        self.log('Q_factor_train', q_factor, prog_bar = True, logger = True)
        self.log('BER_train', ber_value, prog_bar = True, logger = True)
    

    def validation_epoch_end(self, outputs):
        preds = torch.cat([r['preds'] for r in outputs], dim=0)
        targets = torch.cat([r['target'] for r in outputs], dim=0)

        self.ber.update(preds.squeeze(1),targets.squeeze(1))
        ber_value = self.ber.compute()
        q_factor = QFactor(ber_value)

        #self.val_accuracy(preds, targets)
        self.log('Q_factor_val', q_factor, prog_bar = True, logger = True)
        self.log('BER_val', ber_value, prog_bar = True, logger = True)

        nt1 = torch.abs(self.t - 0.5 * self.pulse_width).argmin()
        nt2 = torch.abs(self.t - 8.5 * self.pulse_width).argmin()
        fig, ax = plt.subplots(1, 1, dpi=150)
        ax.plot(self.t[nt1:nt2], targets[0,0,nt1:nt2].cpu().real, label='Target')
        ax.plot(self.t[nt1:nt2], preds[0,0,nt1:nt2].detach().cpu().real, label='Predicted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Re(E)')
        self.logger.experiment.add_figure('prediction_val', fig, global_step=self.current_epoch)

    def get_configuration(self):
        '''
        Returns dict of str with current configuration of the model.
        Can be used for Logger.
        '''
        configuration = {
            'n_layers': self.net.n_layers, 
            'sizes': self.net.sizes,
            'activation': self.net.activation_name, 
            'criterion': str(self.criterion.__repr__())[:-2],
            'optimizer': self.optimizer,
            'optimizer_param': str(self.optimizer_kwargs)[1:-1], 
            'scheduler': self.scheduler,
            'scheduler_param': str(self.scheduler_kwargs)[1:-1],
            'use_batchnorm': self.net.use_batchnorm,
            'dropout': self.net.dropout
        }
        return configuration


class FC_model(torch.nn.Module):
    '''
    Model with linear (fully connected) layers only.
    Number of layers and sizes can be tuned.
    '''
    
    def __init__(self, 
            in_features: int, 
            layers: int, 
            sizes:list = None, 
            bias: bool = False, 
            activation: str = 'ReLU',
            use_batchnorm: bool = False,
            dropout: float = 0.0 
            ):
        '''
        @in_features - number of features in input vector
        @layers - number of linear layers in model
        @sizes - list of output features for linear layers. 
            if @sizes == None : uses @in_features for all layers instead
        @bias - whether to use bias for linear layers or not
        @activation - which type of activation use (torch.nn)
        @use_batchnorm - use Batchnorm1D after linear layer or not
        @dropout - if not 0.0 - then uses dropout after activation
        '''
        super(FC_model, self).__init__()
        # check if @sizes was defined 
        if sizes != None:
            # check if the @sizes has number of elements equal to number of layers
            if len(sizes) != layers:
                raise Exception('Number of sizes do not match to number of layers. Define sizes for all layers.')
            self.sizes = sizes        
        else:
            # if @sizes == None: use @in_features for all layers
            self.sizes = [in_features for _ in range(layers)]

        # Add input size to the begining of @sizes
        self.sizes.insert(0, in_features)

        # Number of layers and list of layers
        self.n_layers = layers
        self.layers_real = nn.ModuleList([])

        # Activation function
        ActivationClass = getattr(torch.nn, activation)

        #parameters for saving configuaration
        self.bias = bias
        self.activation_name = activation
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        # Adding linear layers and activations to list
        for idx in range(layers):
            # if not last layer
            if idx < layers-1:
                # Add linear layer
                self.layers_real.append(nn.Linear(self.sizes[idx], self.sizes[idx+1], bias = bias))
                # Check and add batchnorm
                if self.use_batchnorm:
                    self.layers_real.append(nn.BatchNorm1d(1))
                # Activation
                self.layers_real.append(ActivationClass())
                # if dropout is not 0 -> also add it 
                if self.dropout != 0.0:
                    self.layers_real.append(nn.Dropout(self.dropout))
            # if last layer
            else:
                self.layers_real.append(nn.Linear(self.sizes[idx], self.sizes[idx+1], bias = bias))
                #self.layers_real.append(ActivationClass())

        
        # layers for imaginary values
        self.layers_imag = deepcopy(self.layers_real)


    def forward(self,real, imag):
        # forward prop for all layers 
        for layer_real, layer_imag in zip(self.layers_real, self.layers_imag):
            real = layer_real(real)
            imag = layer_imag(imag)
        
        return real, imag 



