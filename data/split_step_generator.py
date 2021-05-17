from typing import Any, Dict, Optional, Type, Union

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import numpy as np
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
                 num_t_points: int,
                 dispersion_compensate: bool):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dispersion = dispersion
        self.nonlinearity = nonlinearity
        self.pulse_width = pulse_width
        self.z_end = z_end
        self.dz = dz
        self.z_stride = z_stride
        self.num_z_points = int(z_end / (dz * z_stride)) + 1
        self.num_t_points = num_t_points
        self.dispersion_compensate = dispersion_compensate
    
    def Etanal(self, t, z, a, T):
        E0 = torch.zeros(t.shape, dtype=torch.complex128)
        for k in range(1, len(a)+1):
            E0 = E0 + a[k-1]/(np.pi**(1/4)*torch.sqrt(1 + 1j*z))*torch.exp(-(t-k*T)**2/(2*(1 + 1j*z)))
        return E0

    def prepare_data(self, a=None):
        if a is None:
            a = 2*np.random.randint(1,3, size=self.seq_len) - 3
        else:
            self.seq_len = len(a)
        
        z = torch.linspace(0, self.z_end, self.num_z_points)
        
        tmax = (self.seq_len + 1) * self.pulse_width
        tMax = tmax + 5*np.sqrt(2*(1 + self.z_end**2))
        tMin = -tMax
        
        dt = (tMax - tMin) / self.num_t_points
        t = torch.linspace(tMin, tMax-dt, self.num_t_points)
        
        # prepare frequencies
        dw = 2*np.pi/(tMax - tMin)
        w = dw*torch.cat((torch.arange(0, self.num_t_points/2+1),
                          torch.arange(-self.num_t_points/2+1, 0)))
        
        # prepare linear propagator
        LP = torch.exp(-1j*self.dispersion*self.dz/2*w**2)
        
        # Set initial condition
        u = torch.zeros(self.num_z_points, self.num_t_points, dtype=torch.complex128)
        
        buf = self.Etanal(t, torch.tensor(0), a, self.pulse_width)
        u[0,:] = buf
        
        n = 0
        # Numerical integration (split-step)
        for i in range(1, int(self.z_end / self.dz) + 1):
            buf = ifft(LP * fft(buf))
            buf = buf*torch.exp(1j*self.nonlinearity*self.dz*buf.abs()**2)
            buf = ifft(LP*fft(buf))

            if i % self.z_stride == 0:
                n += 1
                u[n,:] = buf
        
        # Dispersion compensation procedure (back propagation D**(-1))
        if self.dispersion_compensate:
            zw = torch.mm(z.view(z.shape[-1], 1), w.view(1, w.shape[-1])**2)
            u = ifft(torch.exp(1j*self.dispersion*zw)*fft(u))
        
        return t, z, u

    def setup(self, stage: Optional[str] = None):
        # transforming, splitting
        if stage is None or stage == "fit":
            self.train = torch.zeros((1000, 1, 1))
            self.val = torch.zeros((1000, 1, 1))
        if stage is None or stage == "test":
            self.test = torch.zeros((1000, 1, 1))

    def train_dataloader(self):
        # TODO: pin_memory=True might be faster
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, pin_memory=False)
