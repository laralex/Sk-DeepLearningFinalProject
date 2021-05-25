import torch
from torch import nn, Tensor

class EVM(nn.Module):
    '''
    Error vector magnetude loss
    Mean value of |E - E_true|^2 / |E_true|^2
    '''
    def __init__(self):
        super(EVM, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''
        Error vector magnetude loss
        Mean value of |input - target|^2 / |target|^2
        '''
        return torch.mean((torch.abs(input - target))**2/(torch.abs(target)+1e-8)**2)
        

class MSE(nn.Module):
    '''
    Mean squared error
    Universal function for both Real and Complex values
    MSE = E[ |target - input|^2 ]
    '''
    def __init__(self):
        super(MSE,self).__init__()
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''Mean squared error loss
        mean value of |target - input|^2'''
        return torch.mean(torch.abs(target - input))