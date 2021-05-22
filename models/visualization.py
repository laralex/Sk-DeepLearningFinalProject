import torch
torch.Tensor.ndim = property(lambda self: len(self.shape))
import matplotlib.pyplot as plt

def plot_BER_Q(BER, Q):
    plt.plot(BER, Q, color='b')
    plt.xlabel('Non-linearity Coefficient')
    plt.ylabel('Q Factor')
    plt.show()