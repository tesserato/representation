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

  lengths = []
  amplitudes = []
  norm_pcs = []
  for i in range(2, Xpc.size):
    x0 = Xpc[i - 1]
    x1 = Xpc[i] + 1
    lengths.append(x1 - x0)
    amplitudes.append(np.max(np.abs(W[x0 : x1])))
    if x1 - x0 >= 4:
      yx = interpolate.interp1d(np.linspace(0, 1, x1-x0), W[x0 : x1], "cubic")
      Ylocal = yx(Xlocal)
      Ylocal = Ylocal / np.max(np.abs(Ylocal))
      norm_pcs.append(Ylocal)
  return np.array(norm_pcs).astype(float), np.array(lengths).astype(float), np.array(amplitudes).astype(float)

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
    self.linear3 = torch.nn.Linear(H, H)
    self.linear4 = torch.nn.Linear(H, D_out)
  
  def forward(self, x):
    h_relu = torch.tanh(2 * self.linear1(x))
    h_relu = torch.tanh(2 * self.linear2(h_relu))
    h_relu = torch.tanh(2 * self.linear3(h_relu))
    y_pred = torch.tanh(2 * self.linear4(h_relu))
    return y_pred



name = "mezzosoprano"

'''###### Read wav file ######'''
W, fps = se.read_wav(f"./original_samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
W = W / amp
n = W.size

'''###### Read Pseudo-cycles info ######'''
Xpc = np.genfromtxt(f"./csvs/{name}.csv", delimiter=",")

norm_waveforms, lengths, amplitudes = get_pcs(Xpc, W)




max_len = np.max(lengths)
max_amp = np.max(amplitudes)
T = np.column_stack((norm_waveforms, lengths / max_len))
T = np.column_stack((T, amplitudes / max_amp))

net = TwoLayerNet(1, 1000, T.shape[1])


T = torch.from_numpy(T).type(torch.float)
X = torch.linspace(0, 1, T.shape[0]).reshape(-1,1)
dataset = torch.utils.data.TensorDataset(X, T)

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=200, shuffle=True)

loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adagrad(net.parameters(), lr=1e-3)



Ly = []
Lx = [0]

net.train()

for epoch in range(1000):
  for x, t in dataloader:
    y = net(x)

    l1 = loss_fn(t[:-2], y[:-2])
    l2 = loss_fn(t[-2], y[-2])
    l3 = loss_fn(t[-1], y[-1])
    loss = l1 + l2 + l3
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(epoch, 'loss: ', loss.data)
    Ly.append(loss.data)
    Lx.append(Lx[-1] + 1)




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
    z=norm_waveforms[:,:-2],
    showlegend=True
  )
)

net.eval()
nnresult=[]
for i in np.arange(T.shape[0]):
  res = net.forward(X[i]).detach().numpy()
  nnresult.append(res[:-2] * res[-1] * max_amp)
nnresult = np.array(nnresult)

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
    z=norm_waveforms[:,:-2] - nnresult[:,:-2],
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
