import torch
import plotly.graph_objs as go
import plotly.express as px

from split_step_generator import SplitStepGenerator, transform_to_1d
from plotly.offline import plot

import time as time

data_gen = SplitStepGenerator(batch_size=10,
                          seq_len=32,
                          dispersion=0.5,
                          nonlinearity=0.02,
                          pulse_width=10,
                          z_end=100,
                          dz=0.1,
                          z_stride=1000,
                          dim_t=2**12,
                          dispersion_compensate=True,
                          num_blocks = 16,
                          n_train_batches = 1,
                          n_val_batches = 0,
                          n_test_batches = 0,
                          two_dim_data=True,
                          device='cpu',
                          pulse_amplitudes=None,
                          pulse_amplitudes_seed=42,
                          )

t0 = time.time()
data_gen.prepare_data()
t, z, u = data_gen.t, data_gen.z, data_gen.E
print(time.time() - t0)

t_start, t_end = data_gen.t_window

u = transform_to_1d(u)

I = abs(u * torch.conj(u)).type(torch.float32)

# Plot first(0) batch
fig = px.line(x=t[t_start:t_end], y=I[0,-1,t_start:t_end], title='Output intensity', labels={'x':'Time', 'y':'Intensity'})
plot(fig)

fig = px.line(x=t[t_start:t_end], y=u[0,0,t_start:t_end].real, title='Input', labels={'x':'Time', 'y':'real E'})
plot(fig)

fig = px.line(x=t[t_start:t_end], y=u[0,-1,t_start:t_end].real, title='Output', labels={'x':'Time', 'y':'real E'})
plot(fig)

# fig = go.Figure(data=[go.Surface(x=t, y=z, z=u.abs())])
# plot(fig)