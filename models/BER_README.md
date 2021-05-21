# How to use BER

BER is Bit error rate. The less the better! 
For example value: 1e-3 - is very bad, we want less value 

There are two ways:
* use BEREstimater like odinary loss in pytorch
* use BERMetric (pytorch_lightning workflow) 

Main workflow:
1.  Creat class instanse with parameters which you inisially have 
```bash
ber = BEREstimater(decision_level=decision_level,pulse_number=pulse_number, pulse_width=pulse_width,t=None,t_window=None)
```
2.  Setup class when you have the rest parameters and setup device:
```bash
ber.setup(t=t, t_window=data_gen.t_window)
ber.to(device)
```
3.  Culculate :
* From probs and target
* From unit tensor:

```bash
#Data from data/test_spst_gen.py

# Parameters:
#seq_len=2049
#pulse_width=10

# Variables:
#u[batch_size, z_dim, t_dim], t[t_dim] 

#Using BEREstimater (torch.nn.module)
seq_len=2049 #parameters in data/test_spst_gen.py
pulse_width=10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
u_ = u.to(device)
preds, target = u_[:,-1,:], u_[:,0,:] # preds and target:[batch_size, dim_t]


#parameters for BER
pulse_width=pulse_width
pulse_number = seq_len//2
decision_level=2

#imputs for BER

u_ = u.to(device)
preds, target = u_[:,-1,:], u_[:,0,:]

#creat BERMetric
ber = BEREstimater(decision_level=decision_level,pulse_number=pulse_number, pulse_width=pulse_width,t=None,t_window=None)

# setup the rest parameters ans setup device
ber.setup(t=t, t_window=data_gen.t_window)
ber.to(u_.device)

#find values from probs and target
preds_bitSeq, target_bitSeq = ber.to_bitSeq(preds, target) 
print(preds_bitSeq.shape) #[batch_size, n_bits], n_bits = pulse_number-1
ber_value = ber(preds, target) #forward method for BEREstimater 
print(ber_value) #tensor(0.8970, device='cuda:0')

print()
#find values from signal u (united matix)[batch_size, dim_z, dim_t]
#target = u[:,0,:] - for z=0, target or good signal
#preds = u[:,-1,:] - for last z, preds or distorted signal
signal_bitSeq = ber.signal_to_bitSeq(u_) 
print(signal_bitSeq.shape) #[batch_size,dim_z,n_bits], n_bits = pulse_number-1
ber_value = ber.forward_signal(u_)
print(ber_value) #tensor(0.8970, device='cuda:0')
```
4. You also can use BERMetric
```bash
#Data from data/test_spst_gen.py
# Parameters:
#seq_len=2049
#pulse_width=10
# Variables:
#u[batch_size, z_dim, t_dim], t[t_dim] 

#Using torchmetric.Matric (pytorch_lightning workflow)
seq_len=2049 #parameters in data/test_spst_gen.py
pulse_width=10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
u_ = u.to(device)
preds, target = u_[:,-1,:], u_[:,0,:] # preds and target:[batch_size, dim_t]


#parameters for BER
pulse_width=pulse_width
pulse_number = seq_len//2
decision_level=2

#imputs for BER

u_ = u.to(device)
preds, target = u_[:,-1,:], u_[:,0,:]

#creat BERMetric
ber = BERMetric(decision_level=decision_level,pulse_number=pulse_number, pulse_width=pulse_width,t=None,t_window=None)

# setup the rest parameters ans setup device
ber.setup(t=t, t_window=data_gen.t_window)
ber.to(u_.device)

#necessery functions
ber.update(preds, target)
ber_value = ber.compute()
print(ber_value) #tensor(0.1030, device='cuda:0') 
```


