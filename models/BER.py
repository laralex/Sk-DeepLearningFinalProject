import torch
from torchmetrics import Metric

#BER estimation realised as a nn.Module
class BEREstimater(torch.nn.Module):
  def __init__(self, decision_level=None, pulse_number=None, pulse_width=None,
               t=None, t_window=None ):
        '''

        Class to do BER Estimation
        Take as input simmetric pulses (negative and positive time),
        but work only with pulses in POSITIVE time!!

        Parameters
        ----------
        decision_level : TYPE: float
            DESCRIPTION: threshold for bit = 1
            value <  threshold ---> 0
            value >  threshold ---> 1
        pulse_width : TYPE: int
            DESCRIPTION: Pulse width in time
        pulse_number : TYPE: int
            DESCRIPTION: number of pulses to decoding (only in positive time)
            pulse_number = seq_len//2 (seq_lin form /data/split_step_generator.py)
        pulse_width : TYPE: int
            DESCRIPTION: Pulse width in time
        t : TYPE: torch.float32 tensor of shape [dim_t]
            DESCRIPTION: Time points. The boundaries of this vector are taken
            in such a way that the signal broadened as it propagates does not
            go beyond the calculation boundaries
        t_window : TYPE: list [torch.int64,torch.int64]
            DESCRIPTION: Contain t_start and t_end to select
            time (negative and positive) with pulses from t. 
l
        '''
               
        super().__init__()
        self.setup(decision_level, pulse_number, pulse_width, t, t_window)

  def setup(self, decision_level=None, pulse_number=None, pulse_width=None,
            t=None, t_window=None):
        if decision_level is not None:
          self.decision_level = decision_level
        if pulse_number is not None:
          self.pulse_number = pulse_number
        if pulse_width is not None:
          self.pulse_width = pulse_width
        if t is not None:
          self.register_parameter('t', torch.nn.Parameter(t,
                                                     requires_grad=False))
        if t_window is not None:
          self.t_window = t_window


  ### Component  funtions   
  def decod_signal(self, signal, pulse_width, t, t_window):
        '''
        Takes as input symmetric pulses (negative and positive time),
        but work only with pulses in POSITIVE time (without symmetric at zero pulse )

        Parameters
        ----------
        signal : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, dim_t].
            DESCRIPTION: Output of the split-step solution.
        pulse_width : TYPE: int
            DESCRIPTION: Pulse width.
        t : TYPE: torch.float32 tensor of shape [dim_t]
            DESCRIPTION: Time points. The boundaries of this vector are taken
            in such a way that the signal broadened as it propagates does not
            go beyond the calculation boundaries
        t_window : TYPE: torch.int64 tensor of shap [2] or (int, int )
            DESCRIPTION: Contain t_start and t_end to select positive
            time with pulses from t. 
        Returns
        -------
        t_dec : TYPE: torch.float32 tensor of shape [dim_t_dec]
            DESCRIPTION: positive time when there are pulses
        signal_decoded : TYPE: torch.float64 tensor of shape [batch_size,dim_z,dim_t_dec]
            DESCRIPTION: decoded signal
        '''

        # saving divice
        device = signal.device

        #cutting the time (we work only with positive time,
        # without symmetric pulse at zero)
        T = pulse_width
        t_start, t_end = t_window
        t_start = torch.argmin(torch.abs(t - 0))
        t_dec = t[t_start:t_end]
        signal = signal[:,:,t_start:t_end] 

        #preparation
        start_pulse = torch.argmin(torch.abs(t_dec - 0.5*T))
        end_pulse = torch.argmin(torch.abs(t_dec - 1.5*T))
        w_pulse = end_pulse - start_pulse

        #take date without symmetric at zero  pulse
        u = torch.zeros_like(signal).to(device)
        u[:,:,start_pulse:t_end] = signal[:,:,start_pulse:t_end]

        u_shifted = torch.zeros_like(u).to(device)
        u_shifted[:,:,start_pulse:-w_pulse] = u[:,:,end_pulse:]       

        #decoding
        signal_decoded = (u + u_shifted)
        # signal_decoded = (u + u_shifted)/2
        signal_decoded = signal_decoded*torch.conj(signal_decoded)

        return u, u_shifted, signal_decoded.real, t_dec


  def decod_bit_seq(self, signal_decoded, t_dec, pulse_width,
                    pulse_number, decision_level):
        '''
        Parameters
        ----------
        signal_decoded : TYPE: torch.float64 tensor of shape [batch_size,dim_z,dim_t_dec]
            DESCRIPTION: decoded signal
        t_dec : TYPE: torch.float32 tensor of shape [dim_t_dec]
            DESCRIPTION: positive time when there are pulses
        pulse_width : TYPE: int
            DESCRIPTION: Pulse width.
        pulse_number : TYPE: int
            DESCRIPTION: number of pulses
        decision_level : TYPE: float
            DESCRIPTION: threshold for bit = 1
            value <  threshold ---> 0
            value >  threshold ---> 1
        Returns
        -------
        decoded_numbers : TYPE: torch.long tensor of shape [batch_size,dim_z, pulse_number-1]
            DESCRIPTION: decoded bit sequence in bits (1 or 0)
        '''

        device = signal_decoded.device
        #prepare
        T = pulse_width
        n = pulse_number-1
        tt = t_dec.expand(n,-1)

        #Getting indexes for regrouping on bits slots
        #Indaxes for t_starts
        t_starts = (torch.arange(0,n)*T + 0.5*T).reshape(n,1).to(device)
        idx_starts = torch.argmin(torch.abs(tt - t_starts), dim=-1, keepdim=True)

        #Indaxes for pulses and time
        w_pulse = idx_starts[1] - idx_starts[0]
        idx_pulses = torch.arange(0,w_pulse.item()).expand(n,-1).to(device) + idx_starts
        idx_pulses_line = idx_pulses.reshape(-1)

        #regrouping on bits slots 
        pulses = torch.index_select(signal_decoded, dim=-1, index=idx_pulses_line)
        pulses = pulses.view(*signal_decoded.shape[:-1], n,w_pulse)

        t_pulses = torch.index_select(t_dec, dim=-1, index=idx_pulses[0,:])

        # integrating by bits slots
        decoded_bitSeq = torch.trapz(pulses,t_pulses, dim=-1)
        # return decoded_bitSeq

        #decoding
        decoded_bitSeq = (decoded_bitSeq > decision_level).type(torch.long)

        return decoded_bitSeq


  def do_BER(self, decoded_bitSeq):
        '''
        DESCRIPTION
        ----------
        Expected, that 
        target = decoded_numbers[:,0,:]
        pred = decoded_numbers[:,-1,:]

        Parameters
        ----------
        decoded_numbers : TYPE: torch.long tensor of shape [batch_size,dim_z, pulse_number-1]
            DESCRIPTION: decoded bit sequence in bits (1 or 0)

        Returns
        -------
        BER_value : TYPE: float
            DESCRIPTION: estimated BER value
        '''

        target = decoded_bitSeq[:,0,:]
        pred = decoded_bitSeq[:,-1,:]
        n = target.size(-1)
        ber_value = torch.sum(target != pred)/(target.numel())
        return ber_value


  def signal_to_bitSeq(self, signal: torch.Tensor):
        '''
        Parameters
        ----------
        signal : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, dim_t].
            DESCRIPTION: Output of the split-step solution.

        Returns
        -------
        decoded_bitSeq : TYPE: torch.long tensor of shape [batch_size, dim_z, dim_t].
            DESCRIPTION: decoded bit sequence 
        '''

        # Decoding
        _, _, decoded_signal, t_decoder = self.decod_signal(signal, self.pulse_width,
                                                       self.t, self.t_window)
        
        # Getting bit sequence
        decoded_bitSeq = self.decod_bit_seq(decoded_signal, t_decoder,
                                            self.pulse_width, self.pulse_number,
                                            self.decision_level)
        return decoded_bitSeq



  ### Assembled  functions
  def forward_signal(self, signal, t=None, decision_level=None, pulse_number=None,
              pulse_width=None, t_window=None):
        '''
        Parameters
        ----------
        signal : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, dim_t].
            DESCRIPTION: Output of the split-step solution.

        t : TYPE: torch.float32 tensor of shape [dim_t]
            DESCRIPTION: Time points. The boundaries of this vector are taken
            in such a way that the signal broadened as it propagates does not
            go beyond the calculation boundaries

        others: see "setup" function 

        Returns
        -------
        BER_value : TYPE: float
            DESCRIPTION: estimated BER value
        '''

        #setup values
        self.setup(decision_level, pulse_number, pulse_width, t, t_window)

        # Full Decoding
        decoded_bitSeq = self.signal_to_bitSeq(signal)

        # Getting BER value
        ber_value = self.do_BER(decoded_bitSeq)

        return ber_value


  def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        '''
        Parameters
        ----------
        preds : TYPE: tensor of shape  [batch_size, dim_t].
            DESCRIPTION: predictions with 1d_tim
        targets : TYPE: tensor of shape  [batch_size, dim_t].
            DESCRIPTION: targets with 1d_tim

        Returns
        -------
        BER_value : TYPE: float
            DESCRIPTION: estimated BER value
            
        '''

        #concatenation
        z_dim = 1
        signal = torch.cat([targets.unsqueeze(z_dim), preds.unsqueeze(z_dim)],
                           dim=z_dim)
        
        # Getting BER value
        ber_value = self.forward_signal(signal)

        return ber_value

  def to_bitSeq(self, preds: torch.Tensor, target: torch.Tensor):
        '''
        Parameters
        ----------
        preds : TYPE: torch.complex128 tensor of shape [batch_size, dim_t].
            DESCRIPTION: preds with 1_d time
        target : TYPE: torch.complex128 tensor of shape [batch_size, dim_t].
            DESCRIPTION: targets with 1_d time

        Returns
        -------
        preds : TYPE: torch.long tensor of shape [batch_size, dim_t].
            DESCRIPTION: prediction of bit sequence
        target : TYPE: torch.long tensor of shape [batch_size, dim_t].
            DESCRIPTION: target bit sequence
        '''

        #concatenation
        z_dim = 1
        signal = torch.cat([target.unsqueeze(z_dim), preds.unsqueeze(z_dim)],
                           dim=z_dim)
        #full decoding
        decoded_numbers = self.signal_to_bitSeq(signal)

        #deconcatenation
        preds, target  = decoded_numbers[:,-1,:], decoded_numbers[:,0,:]

        return preds, target



class BERMetric(Metric):
    def __init__(self, decision_level=None, pulse_number=None, pulse_width=None,
               t=None, t_window=None, dist_sync_on_step=False):
        '''
        Class to use pytorch_lightning workflow

        Parameters
        ----------
        See BEREstimator

        dist_sync_on_step:  TYPE: bool
            DESCRIPTION: predefined parameter for torchmetrics.Metric 
-
        '''
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.BEREstimater = BEREstimater(decision_level, pulse_number,
                                       pulse_width, t, t_window)
        
    def setup(self, decision_level=None, pulse_number=None, pulse_width=None,
            t=None, t_window=None):
        self.BEREstimater.setup(decision_level, pulse_number, pulse_width,
                               t, t_window)
        
    def _input_format(self, preds, target):
        preds, target = self.BEREstimater.to_bitSeq(preds, target)
        return preds, target
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds != target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

