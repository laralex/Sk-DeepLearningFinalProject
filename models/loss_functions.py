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
        