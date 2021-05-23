import auxiliary
from data.transform_1d_2d import *

from typing import Any, Dict, Optional, Type, Union, Tuple
import time
import contextlib
from torch import tensor

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
                 two_dim_data: bool,
                 complex_type_size: int = 128,
                 device: Union[torch.device, str] = 'available',
                 data_source_type: str = 'generation',
                 generation_seed: Optional[int] = None,
                 generation_nonlinearity_limits: Optional[Tuple[float, float]] = None,
                 generate_n_train_batches: Optional[int] = 0,
                 generate_n_val_batches: Optional[int] = 0,
                 generate_n_test_batches: Optional[int] = 0,
                 load_dataset_root_path: Optional[str] = None,
                 ):
        super().__init__()

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

        if complex_type_size == 128:
            self.complex_type = torch.complex128
        elif complex_type_size == 64:
            self.complex_type = torch.complex64
        elif complex_type_size == 32:
            self.complex_type = torch.complex32
        else:
            raise ValueError("complex_type_size has to be 128 or 64 or 32")

        # generation / loading
        assert data_source_type in ['filesystem', 'generation']
        self.data_source_type = data_source_type
        self.generate_n_train_batches = generate_n_train_batches
        self.generate_n_val_batches = generate_n_val_batches
        self.generate_n_test_batches = generate_n_test_batches
        self.generation_seed = generation_seed
        torch.random.manual_seed(self.generation_seed)
        self.train_seed, self.val_seed, self.test_seed = torch.randint(2**31, size=(3,))
        self.generation_nonlinearity_limits = generation_nonlinearity_limits
        self.load_dataset_root_path = load_dataset_root_path

        # usage
        self.batch_size = batch_size
        self.loader_batch_size = self.batch_size if self.data_source_type == 'filesystem' else 1

        self.signal_hparams = {
            'seq_len': self.seq_len,
            'dispersion': self.dispersion,
            'nonlinearity': self.nonlinearity,
            'pulse_width': self.pulse_width,
            'z_end': self.z_end,
            'dz': self.dz,
            'z_stride': self.z_stride,
            'dim_z': self.dim_z,
            'dim_t': self.dim_t,
            'dispersion_compensate': self.dispersion_compensate,
            'num_blocks': self.num_blocks,
            'two_dim_data': self.two_dim_data,
        }

    def prepare_data(self):
        if self.data_source_type == 'filesystem':
            assert self.load_dataset_root_path is not None, "Path to load the dataset isn't specified through config or command line args"
            subdir = auxiliary.files.find_dataset_subdir(self.signal_hparams, self.load_dataset_root_path)
            assert subdir is not None, "Can't find a dataset with signal_hparams.yaml matching parameters"
            self.train, self.val, self.test = auxiliary.files.load_from_subdir(subdir, self.complex_type)
            self.train = SplitStepDataset(data=self.train)
            self.val = SplitStepDataset(data=self.val)
            self.test = SplitStepDataset(data=self.test)
        elif self.data_source_type == 'generation':
            if self.generation_nonlinearity_limits is None:
                nonlinearity_limits = (self.nonlinearity, self.nonlinearity)
            else:
                nonlinearity_limits = self.generation_nonlinearity_limits
            make_dataset = lambda n_batches, seed, pregenerate=False: SplitStepDataset(
                n_batches=n_batches,
                seed=seed,
                pregenerate=pregenerate,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                dispersion=self.dispersion,
                nonlinearity=self.nonlinearity,
                pulse_width=self.pulse_width,
                z_end=self.z_end,
                dz=self.dz,
                z_stride=self.z_stride,
                dim_z=self.dim_z,
                dim_t=self.dim_t,
                dispersion_compensate=self.dispersion_compensate,
                num_blocks=self.num_blocks,
                two_dim_data=self.two_dim_data,
                complex_type=self.complex_type,
                device=self.device,
                nonlinearity_limits=nonlinearity_limits,)
            self.train = make_dataset(self.generate_n_train_batches, self.train_seed)
            self.val = make_dataset(self.generate_n_val_batches, self.val_seed, pregenerate=True)
            self.test = make_dataset(self.generate_n_test_batches, self.test_seed, pregenerate=True)

    def train_dataloader(self):
        # TODO: pin_memory=True might be faster
        return DataLoader(self.train,
            batch_size=self.loader_batch_size, pin_memory=False, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val,
            batch_size=self.loader_batch_size, pin_memory=False, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test,
            batch_size=self.loader_batch_size, pin_memory=False, num_workers=3)

# TODO(laralex): consider IterableDataset
class SplitStepDataset(Dataset):
    def __init__(self,
        batch_size: int,
        seq_len: int,
        dispersion: float,
        nonlinearity: float,
        pulse_width: float,
        z_end: float,
        dz: float,
        z_stride: int,
        dim_z: int,
        dim_t: int,
        dispersion_compensate: bool,
        num_blocks: int,
        two_dim_data: bool,
        complex_type: type,
        n_batches: int,
        device: torch.device,
        nonlinearity_limits: Optional[Tuple[float, float]] = None,
        data: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        pregenerate: bool = False):

        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dispersion = dispersion
        self.nonlinearity = nonlinearity
        self.pulse_width = pulse_width
        self.z_end = z_end
        self.dz = dz
        self.z_stride = z_stride
        self.dim_z = dim_z
        self.dim_t = dim_t
        self.dispersion_compensate = dispersion_compensate
        self.num_blocks = num_blocks
        self.two_dim_data = two_dim_data
        self.device = device
        self.complex_type = complex_type
        self.t_end = ((seq_len - 1)//2 + 1) * pulse_width
        self.t_window = None

        self.data = data
        self.n_batches = n_batches
        self.seed = seed
        if self.seed is not None:
            torch.random.manual_seed(self.seed)
            self.rng_state = torch.random.get_rng_state()
        if nonlinearity_limits is not None:
            self.nonlin_min, self.nonlin_max = nonlinearity_limits
        else:
            self.nonlin_min, self.nonlin_max = self.nonlinearity, self.nonlinearity
        self.batches_list = None
        if pregenerate:
            self.batches_list = []
            for idx in range(self.n_batches):
                _, _, batch = self.generate_batch(self.get_nonlinearity_coef(idx))
                self.batches_list.append(batch)

    # get sample
    def __getitem__(self, idx):
        if self.data is not None:
            dataset_part = self.data[:, idx, ...]
        elif self.batches_list is not None:
            dataset_part = self.batches_list[idx]
        else:
            if self.seed is not None:
                torch.random.set_rng_state(self.rng_state)
            _, _, dataset_part = self.generate_batch(self.get_nonlinearity_coef(idx))
            if self.seed is not None:
                self.rng_state = torch.random.get_rng_state()
        input_ = dataset_part[0, ...].squeeze()
        target_ = dataset_part[-1, ...].squeeze()
        return input_, target_

    def __len__(self):
        if self.data is not None:
            return self.data.shape[1]
        else:
            return self.n_batches

    def get_nonlinearity_coef(self, batch_idx):
        return self.nonlin_min + (batch_idx / self.n_batches)*(self.nonlin_max-self.nonlin_min)

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
        E0 = torch.zeros(a.shape[0], self.dim_t, dtype=self.complex_type, device=self.device)
        complex_z = torch.as_tensor(1 + 1j*z, dtype=self.complex_type, device=self.device)
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
        u = torch.zeros(a.shape[0], self.dim_z, self.dim_t, dtype=self.complex_type, device=self.device)

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

    def generate_batch(self, nonlinearity=None):
        '''
        Direct signal propagation is performed through the split-step method.
        Optional: dispersion compensation and data transformation in 2d.

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
            Shape in case of 2d-time: [batch_size, dim_z, 2*dim_t_per_blok, num_bloks].
            DESCRIPTION: Output transmission line data.
        '''
        a = 2*torch.randint(1,3, size=(self.batch_size, self.seq_len), device=self.device) - 3
        if nonlinearity is None:
            nonlinearity = self.nonlinearity
        # TODO(laralex): consider dropping intermediate z (keep only z=0 and z=z_end)
        t, z, E = self.split_step_solver(
                                        a = a,
                                        T = self.pulse_width,
                                        d = self.dispersion,
                                        c = nonlinearity,
                                        L = self.z_end)

        self.t_window = [torch.abs(t+self.t_end).argmin(), torch.abs(t-self.t_end).argmin()]

        if self.two_dim_data:
            E = transform_to_2d(E, self.num_blocks)

        E = E.transpose(0, 1) # input/output first, then batch size
        return t, z, E