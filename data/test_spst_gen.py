import torch
import plotly.graph_objs as go
import plotly.express as px

from split_step_generator import SplitStepGenerator, transform_to_1d
from plotly.offline import plot

data_gen = SplitStepGenerator(batch_size=10,
                          seq_len=2049,
                          dispersion=0.5,
                          nonlinearity=0.5,
                          pulse_width=10,
                          z_end=100,
                          dz=0.1,
                          z_stride=1000,
                          dim_t=2**16,
                          dispersion_compensate=True,
                          num_blocks = 32,
                          generate_n_train_batches = 1,
                          generate_n_val_batches = 0,
                          generate_n_test_batches = 0,
                          two_dim_data=True,
                          device='cpu',
                          pulse_amplitudes=None,
                          pulse_amplitudes_seed=42,
                          )

data_gen.prepare_data()
t, z, u2d = data_gen.t, data_gen.z, data_gen.E

t_start, t_end = data_gen.t_window

u = transform_to_1d(u2d)

I = abs(u * torch.conj(u)).type(torch.float32)

# Plot first(0) batch
# fig = px.line(x=t[t_start:t_end], y=I[0,-1,t_start:t_end], title='Output intensity', labels={'x':'Time', 'y':'Intensity'})
# plot(fig)

# fig = px.line(x=t[t_start:t_end], y=u[0,0,t_start:t_end].real, title='Input', labels={'x':'Time', 'y':'real E'})
# plot(fig)

# fig = px.line(x=t[t_start:t_end], y=u[0,-1,t_start:t_end].real, title='Output', labels={'x':'Time', 'y':'real E'})
# plot(fig)

fig = px.line(x=t, y=I[0,-1,:], title='Output intensity', labels={'x':'Time', 'y':'Intensity'})
plot(fig)

fig = px.line(x=t, y=u[0,0,:].real, title='Input', labels={'x':'Time', 'y':'real E'})
plot(fig)

fig = px.line(x=t, y=u[0,-1,:].real, title='Output', labels={'x':'Time', 'y':'real E'})
plot(fig)

# fig = go.Figure(data=[go.Surface(x=t, y=z, z=u.abs())])
# plot(fig)