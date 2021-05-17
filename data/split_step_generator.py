from typing import Any, Dict, Optional, Type, Union

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from math import pi, sqrt
from torch.fft import fft, ifft

class SplitStepGenerator(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 seq_len: int,
                 dispersion: float,
                 nonlinearity: float,
                 pulse_width: float,
                 z_end: float,
                 dz: float,
                 z_stride: int,
                 dim_t: int,
                 dispersion_compensate: bool,
                 num_blocks: int):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dispersion = dispersion
        self.nonlinearity = nonlinearity
        self.pulse_width = pulse_width
        self.z_end = z_end
        self.dz = dz
        self.z_stride = z_stride
        self.dim_z = int(z_end / (dz * z_stride)) + 1
        self.dim_t = dim_t
        self.dispersion_compensate = dispersion_compensate
        self.num_blocks = num_blocks
        self.t_end = (seq_len + 1) * pulse_width
        
        self.t_window = None
    
    def Etanal(self, t, z, a, T):
        '''
        Parameters
        ----------
        t : TYPE: torch.float32 tensor of shape [dim_t]
            DESCRIPTION: Time points. See prepare_data description.
        z : TYPE: torch.int64
            DESCRIPTION: In this implementation here z = 0, which corresponds
            to transmitting point.
        a : TYPE: torch.int64 tensor of shape [batch_size, seq_len]
            DESCRIPTION: Pulse amplitude set.
        T : TYPE: int
            DESCRIPTION: Pulse width.

        Returns
        -------
        E0 : TYPE: torch.complex128 tensor of shape [batch_size, dim_t]
            DESCRIPTION: Initial data at transmitting point (target).
        '''
        E0 = torch.zeros(a.shape[0], self.dim_t, dtype=torch.complex128)
        for k in range(1, self.seq_len + 1):
            x = (a[:,k-1]/(pi**(1/4)*torch.sqrt(1 + 1j*z))).view(a.shape[0], 1)
            y = torch.exp(-(t-k*T)**2/(2*(1 + 1j*z))).view(1, self.dim_t)
            
            torch.add(E0, x.multiply(y), out=E0)
        return E0
    
    def split_step_solver(self, a, T, d, c, L):
        '''
        Parameters
        ----------
        a : TYPE: torch.int64 tensor of shape [batch_size, seq_len], optional
            DESCRIPTION: Pulse amplitude set.
        T : TYPE: float
            DESCRIPTION: Pulse width.
        d : TYPE: float
            DESCRIPTION: dispersion coefficient.
        c : TYPE: float
            DESCRIPTION: nonlinearity coefficient.
        L : TYPE: float
            DESCRIPTION: End of transmission line (z_end).

        Returns
        -------
        t : TYPE: torch.float32 tensor of shape [dim_t]
            DESCRIPTION: Time points. See prepare_data description.
        z : TYPE: torch.float32 tensor of shape [dim_z]
            DESCRIPTION: Points in space. See prepare_data description.
        u : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, dim_t].
            DESCRIPTION: Output of the split-step solution.
        '''
        z = torch.linspace(0, L, self.dim_z)
        
        tMax = self.t_end + 5*sqrt(2*(1 + L**2))
        tMin = -tMax
        
        dt = (tMax - tMin) / self.dim_t
        t = torch.linspace(tMin, tMax-dt, self.dim_t)
        
        # prepare frequencies
        dw = 2*pi/(tMax - tMin)
        w = dw*torch.cat((torch.arange(0, self.dim_t/2+1),
                          torch.arange(-self.dim_t/2+1, 0)))
        
        # prepare linear propagator
        LP = torch.exp(-1j*d*self.dz/2*w**2)
        
        # Set initial condition
        u = torch.zeros(self.batch_size, self.dim_z, self.dim_t, dtype=torch.complex128)
        
        buf = self.Etanal(t, torch.tensor(0), a, T)
        u[:,0,:] = buf
        
        n = 0
        # Numerical integration (split-step)
        for i in range(1, int(L / self.dz) + 1):
            buf = ifft(LP * fft(buf))
            buf = buf*torch.exp(1j*c*self.dz*buf.abs()**2)
            buf = ifft(LP*fft(buf))

            if i % self.z_stride == 0:
                n += 1
                u[:, n, :] = buf
        
        # Dispersion compensation procedure (back propagation D**(-1))
        if self.dispersion_compensate:
            zw = torch.mm(z.view(z.shape[-1], 1), w.view(1, w.shape[-1])**2)
            u = ifft(torch.exp(1j*d*zw)*fft(u))
        
        return t, z, u

    def prepare_data(self, two_dim_data=False, a=None):
        '''
        Direct signal propagation is performed through the split-step method.
        Optional: dispersion compensation and data transformation in 2d.

        Parameters
        ----------
        two_dim_data : TYPE: bool, optional
            DESCRIPTION: Determine whether or not to make a transformation
            in 2d. The default is False.
        a : TYPE: torch.int64 tensor of shape [batch_size, seq_len], optional
            DESCRIPTION: Pulse amplitude set. The default is None.

        Returns
        -------
        t : TYPE: torch.float32 tensor of shape [dim_t]
            DESCRIPTION: Time points. The boundaries of this vector are taken
            in such a way that the signal broadened as it propagates does not
            go beyond the calculation boundaries.
        z : TYPE: torch.float32 tensor of shape [dim_z]
            DESCRIPTION: Points in space taken from 0 to z_end with a
            periodicity z_stride.
        E : TYPE: torch.complex128 tensor
            Shape in case of 1d-time: [batch_size, dim_z, dim_t].
            Shape in case of 1d-time: [batch_size, dim_z, 2*dim_t_per_blok, num_bloks].
            DESCRIPTION: Output transmission line data.
        '''
        if a is None:
            a = 2*torch.randint(1,3, size=(self.batch_size, self.seq_len)) - 3
        else:
            self.seq_len = a.shape[-1]
            self.t_end = (self.seq_len + 1) * self.pulse_width
        
        t, z, E = self.split_step_solver(a = a,
                                         T = self.pulse_width,
                                         d = self.dispersion,
                                         c = self.nonlinearity,
                                         L = self.z_end)
        
        self.t_window = [torch.abs(t).argmin(), torch.abs(t-self.t_end).argmin()]
        
        if two_dim_data:
            E = transform_to_2d(E, self.num_blocks)
        
        return t, z, E

    def setup(self, stage: Optional[str] = None):
        # transforming, splitting
        if stage is None or stage == "fit":
            self.train = torch.zeros(1000, 1, 1)
            self.val = torch.zeros(1000, 1, 1)
        if stage is None or stage == "test":
            self.test = torch.zeros(1000, 1, 1)

    def train_dataloader(self):
        # TODO: pin_memory=True might be faster
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, pin_memory=False)


def transform_to_2d(data, num_blocks):
    '''
    Parameters
    ----------
    data : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, dim_t]
        DESCRIPTION: Output transmission line data with 1d-time.
    num_blocks : TYPE: int
        DESCRIPTION: The number of blocks into which the data will be split
        during transformation into 2d-time data. Time dimension is splitted
        on blocks and reshaped to 2d dimensions with overlaps.

    Returns
    -------
    data : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, 2*dim_t_per_blok, num_bloks]
        DESCRIPTION: Output transmission line data with padded 2d-time.
    '''
    bs, dim_z, dim_t = data.shape
    dim_t_per_block = dim_t//num_blocks
    data = data.view(bs, dim_z, num_blocks, dim_t_per_block).transpose(-2, -1)
    
    data_up = data[:, :, :dim_t_per_block//2, 1:]
    data_down = data[:, :, dim_t_per_block//2:, :-1]
    
    padd_zeros = torch.zeros(bs, dim_z, dim_t_per_block//2, 1)
    
    data_up = torch.cat((data_up, padd_zeros), dim=-1)
    data_down = torch.cat((padd_zeros, data_down), dim=-1)
    
    data = torch.cat((data_down, data), dim=-2)
    data = torch.cat((data, data_up), dim=-2)
    
    return data

def transform_to_1d(data):
    '''
    Reverse operation to transform_to_2d
    
    Parameters
    ----------
    data : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, 2*dim_t_per_blok, num_bloks]
        DESCRIPTION: Data with 2d-time.

    Returns
    -------
    data : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, dim_t]
        DESCRIPTION: Data with 1d-time.
    '''
    dim_padded_t = data.shape[-2]
    data = data[:, :, dim_padded_t//4:dim_padded_t*3//4, :]
    data = data.transpose(-2, -1).flatten(-2, -1)
    return data