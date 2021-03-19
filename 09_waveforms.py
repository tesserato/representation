# from os import linesep
from statistics import mode
import numpy as np
import plotly.graph_objects as go
import signal_envelope as se
from scipy import interpolate
import sys


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

def to_plot(Matrix):
  X = []
  Y = []
  for line in Matrix:
    for x, y in enumerate(line):
      X.append(x)
      Y.append(y)
    X.append(None)
    Y.append(None)
  return X, Y

name = "cello"

'''###### Read wav file ######'''
W, fps = se.read_wav(f"./original_samples/{name}.wav")
W = W - np.average(W)
amp = np.max(np.abs(W))
W = W / amp
n = W.size

'''###### Read Pseudo-cycles info ######'''
Xpc = np.genfromtxt(f"./csvs/{name}.csv", delimiter=",")

X_xpcs, Y_xpcs = [], []
for x in Xpc:
  X_xpcs.append(x)
  X_xpcs.append(x)
  X_xpcs.append(None)
  Y_xpcs.append(-1)
  Y_xpcs.append(1)
  Y_xpcs.append(None)


average_waveform, orig_waveforms, norm_waveforms = get_pcs(Xpc, W)

# norm_waveforms = norm_waveforms - average_waveform



'''============================================================================'''

FONT = dict(
    family="Latin Modern Roman",
    color="black",
    size=13.3333
  )


'''============================================================================'''
'''                               PLOT WAVEFORM                                '''
'''============================================================================'''
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


X, Y = to_plot(norm_waveforms)
fig.add_trace(
  go.Surface(
    name="Normalized Waveforms", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    z=norm_waveforms,
    # fill="toself",
    # mode="lines",
    # line=dict(
    #     width=1,
    #     color="rgba(38, 12, 12, 0.1)",
    #     # showscale=False
    # ),
    # visible = "legendonly"
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

# fig.layout.font = FONT

fig.data[0].showscale=False
fig.data[0].coloraxis=None
# fig.data[0].colorscale='gray'

fig.show(config=dict({'scrollZoom': True}))
save_name = sys.argv[0].split('/')[-1].replace(".py", "")
wid = 650
hei = 400
fig.write_image("./paper/images/" + save_name + ".svg", width=wid, height=hei, engine="kaleido", format="svg")
fig.write_image("./site/public/images/" + save_name + ".webp", width=int(1.7*wid), height=int(1.5*hei), format="webp")
fig.write_html("./site/public/images/" + save_name + ".html", include_plotlyjs="cdn", include_mathjax="cdn")
print("saved:", save_name)

