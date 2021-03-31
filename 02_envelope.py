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
  def __init__(self, layers):
    super(TwoLayerNet, self).__init__()
    self.layers = []
    for i in range(1, len(layers)):
      self.layers.append(torch.nn.Linear(layers[i - 1], layers[i]))
    self.layers = torch.nn.ModuleList(self.layers)
  
  def forward(self, x):
    y_pred = torch.sigmoid(2 * self.layers[0](x))
    for l in self.layers[1:]:
      y_pred = torch.sigmoid(l(y_pred))
    return y_pred



name = "cello"

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
T = amplitudes / max_amp

net = TwoLayerNet([1, 5, 1])


T = torch.from_numpy(T).type(torch.float)
X = torch.linspace(0, 1, T.shape[0]).reshape(-1,1)
dataset = torch.utils.data.TensorDataset(X, T)

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)

loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adagrad(net.parameters(), lr=1e-1)



Ly = []
Lx = []

net.train()

for epoch in range(10000):
  epoch_losses = []
  for x, t in dataloader:
    y = net(x)
    loss = loss_fn(t, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    epoch_losses.append(loss.data.item())
  
  average_loss = np.average(epoch_losses)
  print(epoch, 'loss: ', average_loss)
  Ly.append(average_loss)
  Lx.append(epoch)

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
  go.Scatter(
    name="Envelope",
    # x = Lx,
    y = amplitudes
  )
)

net.eval()
envelopesresult = []
for i in np.arange(T.shape[0]):
  res = net.forward(X[i]).detach().numpy()
  env = res[-1] * max_amp
  envelopesresult.append(env)


envelopesresult = np.array(envelopesresult)


fig.add_trace(
  go.Scatter(
    name="Estimated Envelope",
    # x = Lx,
    y = envelopesresult
  )
)




fig.show(config=dict({'scrollZoom': True}))
# save_name = sys.argv[0].split('/')[-1].replace(".py", "")
# wid = 650
# hei = 400
# fig.write_image("./paper/images/" + save_name + ".svg", width=wid, height=hei, engine="kaleido", format="svg")
# fig.write_image("./site/public/images/" + save_name + ".webp", width=int(1.7*wid), height=int(1.5*hei), format="webp")
# fig.write_html("./site/public/images/" + save_name + ".html", include_plotlyjs="cdn", include_mathjax="cdn")
# print("saved:", save_name)

