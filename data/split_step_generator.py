from typing import Any, Dict, Optional, Type, Union
import time
import contextlib

import torch
from torch.utils.data import DataLoader, Dataset, random_split
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
                 num_blocks: int,
                 n_train_batches: int,
                 n_val_batches: int,
                 n_test_batches: int,
                 two_dim_data: bool,
                 device: Union[torch.device, str],
                 pulse_amplitudes: Optional[torch.Tensor] = None,
                 pulse_amplitudes_seed: Optional[int] = None,
                 ):
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
        self.n_train_batches = n_train_batches
        self.n_val_batches = n_val_batches
        self.n_test_batches = n_test_batches
        self.two_dim_data = two_dim_data

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            if device.startswith(('cuda', 'cpu')):
                self.device = torch.device(device)
            elif device == 'available':
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                raise ValueError("Incorrect value of device string")

        self.pulse_amplitudes = pulse_amplitudes
        if isinstance(self.pulse_amplitudes, torch.Tensor):
            self.pulse_amplitudes = self.pulse_amplitudes.to(device)

        self.pulse_amplitudes_seed = pulse_amplitudes_seed
        self.t_end = ((seq_len - 1)//2 + 1) * pulse_width
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
        E0 = torch.zeros(a.shape[0], self.dim_t, dtype=torch.complex128, device=self.device)
        complex_z = torch.as_tensor(1 + 1j*z, dtype=torch.complex128, device=self.device)
        half_seq_len = (self.seq_len - 1)//2
        sequence_indices = torch.arange(-half_seq_len, half_seq_len + 1, device=self.device).view(-1, 1)
        # [1, dim_t] minus [seq_len, 1] makes them broadcasted to [seq_len, dim_t]
        y_whole = torch.exp(-(t.view(1, -1) - T * sequence_indices)**2 / (2*complex_z)) # [seq_len, dim_t]
        x_whole = a/(pi**(1/4)*torch.sqrt(complex_z)) # [bs, seq_len]
        torch.matmul(x_whole, y_whole, out=E0) # [bs, dim_t]
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
        z = torch.linspace(0, L, self.dim_z, device=self.device)
        
        tMax = self.t_end + 4*sqrt(2*(1 + L**2))
        tMin = -tMax
        
        dt = (tMax - tMin) / self.dim_t
        t = torch.linspace(tMin, tMax-dt, self.dim_t, device=self.device)
        
        # prepare frequencies
        dw = 2*pi/(tMax - tMin)
        w = dw*torch.cat((torch.arange(0, self.dim_t/2+1, device=self.device),
                          torch.arange(-self.dim_t/2+1, 0, device=self.device)))
        
        # prepare linear propagator
        LP = torch.exp(-1j*d*self.dz/2*w**2)
        
        # Set initial condition
        u = torch.zeros(a.shape[0], self.dim_z, self.dim_t, dtype=torch.complex128, device=self.device)
        
        buf = self.Etanal(t, torch.tensor(0, device=self.device), a, T)
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

    def prepare_data(self):
        '''
        Direct signal propagation is performed through the split-step method.
        Optional: dispersion compensation and data transformation in 2d.

        Assigns self. attributes
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
            Shape in case of 2d-time: [batch_size, dim_z, 2*dim_t_per_blok, num_bloks].
            DESCRIPTION: Output transmission line data.
        '''
        print('Generating the dataset using Split-Step')
        start_time = time.time()

        a = self.pulse_amplitudes

        if a is None:
            # fix seed for this particular generation
            if self.pulse_amplitudes_seed is None:
                context = contextlib.nullcontext()
            else:
                context = torch.random.fork_rng()
                torch.random.manual_seed(self.pulse_amplitudes_seed)
            with context:
                dataset_size = self.batch_size * (self.n_train_batches + self.n_val_batches + self.n_test_batches)
                a = 2*torch.randint(1,3, size=(dataset_size, self.seq_len), device=self.device) - 3
        else:
            a = a.to(self.device)
            self.seq_len = a.shape[-1]
            self.t_end = ((self.seq_len - 1)//2 + 1) * self.pulse_width
        
        # TODO(laralex): consider dropping intermediate z (keep only z=0 and z=z_end)
        self.t, self.z, self.E = self.split_step_solver(
                                        a = a,
                                        T = self.pulse_width,
                                        d = self.dispersion,
                                        c = self.nonlinearity,
                                        L = self.z_end)
        
        self.t_window = [torch.abs(self.t+self.t_end).argmin(), torch.abs(self.t-self.t_end).argmin()]
        
        if self.two_dim_data:
            self.E = transform_to_2d(self.E, self.num_blocks)
        end_time = time.time()
        print(f'Dataset was generated in {int(end_time - start_time)} sec')

    def setup(self, stage: Optional[str] = None):
        # transforming, splitting
        # TODO(laralex): avoid moving data back to CPU (but otherwise CUDA
        # crashes in SplitStepDataset)
        if stage is None or stage == "fit":
            train_end = self.batch_size*self.n_test_batches
            self.train = self.E[:train_end, ...].transpose(0, 1).to('cpu')
            val_end = train_end + self.batch_size*self.n_val_batches
            self.val = self.E[train_end:val_end, ...].transpose(0, 1).to('cpu')
        if stage is None or stage == "test":
            self.test = self.E[val_end:, ...].transpose(0, 1).to('cpu')

    def train_dataloader(self):
        # TODO: pin_memory=True might be faster
        return DataLoader(SplitStepDataset(self.train),
            batch_size=self.batch_size, pin_memory=False, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(SplitStepDataset(self.val),
            batch_size=self.batch_size, pin_memory=False, num_workers=3)

    def test_dataloader(self):
        return DataLoader(SplitStepDataset(self.test),
            batch_size=self.batch_size, pin_memory=False, num_workers=3)


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
    
    padd_zeros = torch.zeros(bs, dim_z, dim_t_per_block//2, 1, device=data.device)
    
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

# TODO(laralex): consider IterableDataset
class SplitStepDataset(Dataset):
    def __init__(self, dataset_content):
        super().__init__()
        self.dataset_content = dataset_content

    # get sample
    def __getitem__(self, idx):
        dataset_part = self.dataset_content[:, idx, ...]
        # TODO(laralex): some models require size [bs, ch, seq], not [bs, seq]
        # (or for 2d [bs, ch, h, w])
        input_ = dataset_part[0, ...].unsqueeze(0)
        target_ = dataset_part[-1, ...].unsqueeze(0)
        return input_, target_

    def __len__(self):
        return self.dataset_content.shape[1]