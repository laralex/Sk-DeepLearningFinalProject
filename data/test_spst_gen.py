import sys
sys.path.insert(0, '../')

import torch
import plotly.graph_objs as go
import plotly.express as px

from split_step_generator import SplitStepDataset, transform_to_1d
from plotly.offline import plot

data_gen = SplitStepDataset(batch_size=2,
                          seq_len=2049,
                          dispersion=0.5,
                          nonlinearity=0.05,
                          pulse_width=10,
                          z_end=100,
                          dz=0.1,
                          z_stride=1000,
                          dim_z=2,
                          dim_t=2**16,
                          dispersion_compensate=True,
                          num_blocks = 32,
                          n_batches = 1,
                          two_dim_data=True,
                          seed=42,
                          device=torch.device('cpu'),
                          complex_type=torch.complex64,
                          )

t, z, u2d = data_gen.generate_batch()

t_start, t_end = data_gen.t_window

u = transform_to_1d(u2d)

I = abs(u * torch.conj(u)).type(torch.float32)

# Plot first(0) batch
# fig = px.line(x=t[t_start:t_end], y=I[-1,0,t_start:t_end], title='Output intensity', labels={'x':'Time', 'y':'Intensity'})
# plot(fig)

# fig = px.line(x=t[t_start:t_end], y=u[0,0,t_start:t_end].real, title='Input', labels={'x':'Time', 'y':'real E'})
# plot(fig)

# fig = px.line(x=t[t_start:t_end], y=u[-1,0,t_start:t_end].real, title='Output', labels={'x':'Time', 'y':'real E'})
# plot(fig)

fig = px.line(x=t, y=I[-1,0,:], title='Output intensity', labels={'x':'Time', 'y':'Intensity'})
plot(fig)

fig = px.line(x=t, y=u[0,0,:].real, title='Input', labels={'x':'Time', 'y':'real E'})
plot(fig)

fig = px.line(x=t, y=u[-1,0,:].real, title='Output', labels={'x':'Time', 'y':'real E'})
plot(fig)

# fig = go.Figure(data=[go.Surface(x=t, y=z, z=u.abs())])
# plot(fig)