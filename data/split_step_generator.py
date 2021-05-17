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
                 num_z_points: int,
                 num_t_points: int,
                 dispersion_compensate: bool):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dispersion = dispersion
        self.nonlinearity = nonlinearity
        self.pulse_width = pulse_width
        self.z_end = z_end
        self.num_z_points = num_z_points
        self.num_t_points = num_t_points
        self.dispersion_compensate = dispersion_compensate
    
    def Etanal(self, t, z, a, T):
        E0 = torch.zeros(t.shape, dtype=torch.complex128)
        for k in range(1, len(a)+1):
            E0 = E0 + a[k-1]/(np.pi**(1/4)*torch.sqrt(1 + 1j*z))*torch.exp(-(t-k*T)**2/(2*(1 + 1j*z)))
        return E0

    def prepare_data(self):
        # some one-time stuff like downloading, generating
        a = 2*np.random.randint(1,3, size=self.seq_len) - 3
        z = torch.linspace(0, self.z_end, self.num_z_points+1);
        
        tmax = (self.seq_len + 1) * self.pulse_width
        tMin = -5*tmax
        tMax = 5*tmax
        
        # nMin = int(-self.num_t_points/2)
        # nMax = int(-nMin-1)
        # nPoints = int(nMax-nMin+1)
        
        nPoints = self.num_t_points
        
        dt = (tMax - tMin)/nPoints
        t = torch.linspace(tMin, tMax-dt, nPoints)
        
        # prepare Fourier multipliers
        dw = 2*np.pi/(tMax-tMin)
        w = dw*torch.cat((torch.arange(0, nPoints/2+1), torch.arange(-nPoints/2+1, 0)))
        
        dz = z[2]-z[1];
        # prepare linear propagators (diagonal in Fourier space)
        LP2 = torch.exp(-1j * self.dispersion * w**2 * dz/2)
        
        # Set initial condition
        u = torch.zeros(len(z), nPoints, dtype=torch.complex128)
        
        u[0,:] = self.Etanal(t, torch.tensor(0), a, self.pulse_width)
        
        # Numerical integration (split-step)
        for i in range(len(z)-1):
            u[i+1,:] = ifft(LP2 * fft(u[i,:]))
            u[i+1,:] = u[i+1,:] * torch.exp(1j * self.nonlinearity * dz * u[i+1,:].abs()**2)
            u[i+1,:] = ifft(LP2 * fft(u[i+1,:]))
            
            # Dispersion compensation procedure (back propagation D**(-1))
            if self.dispersion_compensate and (i != 1):
                u[i,:] = ifft(torch.exp(1j * self.dispersion * w**2 * z[i]) * fft(u[i,:]));
        
        # Dispersion compensation, last step
        if self.dispersion_compensate:
            u[-1,:] = ifft(torch.exp(1j * 1/2 * w**2 * z[-1]) * fft(u[-1,:]))
        
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
