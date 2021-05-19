import torch
from torch import nn, Tensor

class MSE(nn.MSELoss):
    '''
    Usual Mean squared error from pytorch
    '''
    def __init__(self):
        super(MSE, self).__init__()
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)


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
        return torch.mean((torch.abs(input - target))**2/(torch.abs(target))**2)
        