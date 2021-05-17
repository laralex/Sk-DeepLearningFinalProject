import torch
import plotly.graph_objs as go
import plotly.express as px

from split_step_generator import SplitStepGenerator
from plotly.offline import plot

import time as time

data = SplitStepGenerator(batch_size=5,
                          seq_len=10,
                          dispersion=0.5,
                          nonlinearity=0.1,
                          pulse_width=10,
                          z_end=100,
                          dz=0.1,
                          z_stride=10,
                          num_t_points=2**12,
                          dispersion_compensate=True)

# a = torch.tensor([1, -1, 1, -1, -1])

t0 = time.time()
t, z, u = data.prepare_data()
print(time.time() - t0)


I = (u * torch.conj(u)).type(torch.float32)

fig = px.line(x=t, y=I[-1,:], labels={'x':'Time', 'y':'Intensity'})
plot(fig)

fig = px.line(x=t, y=u[0,:].real, title='Input', labels={'x':'Time', 'y':'real E'})
plot(fig)

fig = px.line(x=t, y=u[-1,:].real, title='Output', labels={'x':'Time', 'y':'real E'})
plot(fig)

# fig = go.Figure(data=[go.Surface(x=t, y=z, z=u.abs())])
# plot(fig)