import torch
import numpy as np
from statistics import mode
import plotly.graph_objects as go
import signal_envelope as se
from scipy import interpolate
import sys

np.random.seed(1)
torch.manual_seed(1)



def get_pcs(Xpc, W):
  Xpc = Xpc.astype(int)
  # amp = np.max(np.abs(W))
  # max_T = int(np.max(np.abs(Xpc[1:] - Xpc[:-1])))
  Xlocal = np.linspace(0, 1, mode(Xpc[1:] - Xpc[:-1]))

  orig_pcs = []
  norm_pcs = []
  for i in range(2, Xpc.size):
    x0 = Xpc[i - 1]
    x1 = Xpc[i] + 1
    orig_pcs.append(W[x0 : x1])
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1-x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      # Ylocal = Ylocal / np.max(np.abs(Ylocal)) * amp
      norm_pcs.append(Ylocal)
  return np.average(np.array(norm_pcs), 0), orig_pcs, norm_pcs

def get_loss_plot(Lx, Ly):
  FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )
  fig = go.Figure()
  fig.layout.template ="plotly_white" 
  fig.update_layout(
    xaxis_title="<b>Step</b>",
    yaxis_title="<b>Loss</b>",
    legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
    margin=dict(l=0, r=0, b=0, t=0),
    font=FONT,
    # titlefont=FONT
  )
  # fig.layout.xaxis.title.font=FONT
  # fig.layout.yaxis.title.font=FONT

  fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')
  fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')

  fig.add_trace(
    go.Scatter(
      name="Loss",
      x = Lx,
      y = Ly
    )
  )
  return fig

class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, H) 
    self.linear3 = torch.nn.Linear(H, D_out)

    self.last_y_pred = None
    self.criterion = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adagrad(self.parameters())
  
  def forward(self, x):
    h_relu = torch.tanh(2 * self.linear1(x))
    h_relu = torch.tanh(2 * self.linear2(h_relu))
    y_pred = torch.tanh(2 * self.linear3(h_relu))
    self.last_y_pred = y_pred
    return y_pred

  def backward(self, t):
    loss = self.criterion(t, self.last_y_pred)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.data




name = "mezzosoprano"

'''###### Read wav file ######'''
W, fps = se.read_wav(f"./original_samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
W = W / amp
n = W.size

'''###### Read Pseudo-cycles info ######'''
Xpc = np.genfromtxt(f"./csvs/{name}.csv", delimiter=",")

average_waveform, orig_waveforms, norm_waveforms = get_pcs(Xpc, W)

norm_waveforms = np.array(norm_waveforms)

print(norm_waveforms.shape) # (number of pseudo cycles, length of each pseudo cycle)

net = TwoLayerNet(1, 200, norm_waveforms.shape[1])


T = torch.from_numpy(norm_waveforms.astype(float)).type(torch.float)
X = torch.linspace(0, 1, norm_waveforms.shape[0]).reshape(-1,1)
I = np.arange(norm_waveforms.shape[0])

Ly = []
Lx = []

net.train()
for step in range(100000):
  i = np.random.choice(I)
  y = net.forward(X[i])
  loss = net.backward(T[i])
  if step % 1000 == 0:
    print(i, 'loss: ', loss.data)
    Ly.append(loss.data)
    Lx.append(step)


fig = get_loss_plot(Lx, Ly)
fig.show(config=dict({'scrollZoom': True}))

'''============================================================================'''
'''                               PLOT WAVEFORM                                '''
'''============================================================================'''
FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )
fig = go.Figure()
fig.layout.template ="plotly_white" 
fig.update_layout(
  # xaxis_title="<b><i>i</i></b>",
  # yaxis_title="<b>Amplitude</b>",
  legend=dict(orientation='h', yanchor='top', xanchor='left', y=1.1),
  margin=dict(l=0, r=0, b=0, t=0),
  font=FONT,
  # titlefont=FONT
)
# fig.layout.xaxis.title.font=FONT
# fig.layout.yaxis.title.font=FONT

fig.update_xaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')
fig.update_yaxes(showline=False, showgrid=False, zerolinewidth=1, zerolinecolor='black')

fig.add_trace(
  go.Surface(
    name="Original Waveforms",
    z=norm_waveforms,
    showlegend=True
  )
)

net.eval()
nnresult=[]
for i in I:
  nnresult.append(net.forward(X[i]).detach().numpy())

fig.add_trace(
  go.Surface(
    name="Inferred Waveforms", # <|<|<|<|<|<|<|<|<|<|<|<|
    z=nnresult,
    showlegend=True
  )
)

fig.add_trace(
  go.Surface(
    name="errors", # <|<|<|<|<|<|<|<|<|<|<|<|
    z=norm_waveforms - nnresult,
    showlegend=True
  )
)

fig.update_layout(
  scene_camera = dict(
    center=dict(x=0, y=0, z=-0.2),
    eye=dict(x=-1.2,y=-1.2,z=1.55)
  ),
  # scene_camera_eye=dict(x=-1.2,y=-1.2,z=1.55),
  scene = dict(
    xaxis = dict(title="<b>Frame</b>"),
    yaxis = dict(title="<b>Pseudo cycle number</b>"),
    zaxis = dict(title="<b>Amplitude</b>"),
  )
)

torch.save(net.state_dict(), "test.zip")

se.save_wav(amp * np.array(nnresult).flatten())

fig.show(config=dict({'scrollZoom': True}))
# save_name = sys.argv[0].split('/')[-1].replace(".py", "")
# wid = 650
# hei = 400
# fig.write_image("./paper/images/" + save_name + ".svg", width=wid, height=hei, engine="kaleido", format="svg")
# fig.write_image("./site/public/images/" + save_name + ".webp", width=int(1.7*wid), height=int(1.5*hei), format="webp")
# fig.write_html("./site/public/images/" + save_name + ".html", include_plotlyjs="cdn", include_mathjax="cdn")
# print("saved:", save_name)
