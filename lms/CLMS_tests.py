import numpy as np
import plotly.graph_objects as go


def direct(Q, V, Y, s):
  '''solving inverse'''
  QTQinv = np.linalg.inv(Q.T @ Q)
  tau = np.linalg.inv(V @ QTQinv @ V.T)
  QTY = Q.T @ Y

  A = QTQinv @ (QTY - V.T @ tau @ V @ QTQinv @ QTY)
  return np.reshape(A, (s, -1))

def sparse(Q, V, Y, k, s, m):
  '''solving by systems (sparse)'''
  a1 = np.hstack([                         # cols           lines
        np.zeros(((k * s, k * s))),        # k * s          k*s
        Q.T,                               # m
        V.T                                # 2 * (s-1)
      ])
  a2 = np.hstack([
      Q,                                   # k * s          m
      -1/2 * np.identity(m),               # m
      np.zeros((m, 2 * (s-1)))             # 2 * (s-1)
    ])
  a3 = np.hstack([
      V,                                   # k * s          2 * (s-1)
      np.zeros((2 * (s-1), m + 2 * (s-1))) # m + 2 * (s-1)
    ])

  x = np.vstack([a1, a2, a3])
  y = np.hstack([np.zeros(2 * (s-1)), Y, np.zeros(k * s)]).T

  A = np.linalg.solve(x, y)[:s*k]
  return np.reshape(A, (s, -1))

def kkt(Q, V, Y, k, s):
  a1 = np.hstack([ 2 * Q.T @ Q , V.T                              ])
  a2 = np.hstack([ V           , np.zeros((2 * (s-1), 2 * (s-1))) ])
  x = np.vstack([a1, a2])
  # print((2 * Q.T @ Y).shape)
  y = np.hstack([2 * Q.T @ Y, np.zeros(2 * (s-1))]).T
  A = np.linalg.solve(x, y)[:s*k]
  print(A)
  return np.reshape(A, (s, -1))

def eval(A, l, s):
  '''evaluates y E Y from a coefficient matrix A'''
  Y_clms = []
  for i in range(s):
    for x in range(i * l, (i + 1) * l):
      y = 0
      for j in range(k):
        y += A[i, j] * X[x]**j
      Y_clms.append(y)
    # Y_clms.append(None)

  x = len(Y_clms)
  while x < m:
    y = 0
    for i in range(k):
      y += A[s-1, i] * X[x]**i
    Y_clms.append(y)
    x += 1
  return Y_clms

np.random.seed(1)
m = 50000
k = 3
X = np.arange(m, dtype=np.float64)
Y = (np.cos(X/100) + np.random.randn(m)/5).astype(np.float64) # b
'''Constrained Least Squares (with q intervals and k-1 polynomial)'''
s = 10
l = m // s
Q = np.zeros((m, k * s), dtype=np.float64) # A
V = np.zeros((2 * (s-1), k * s), dtype=np.float64) # C

'''populating Q'''
for i1 in range(s):
  for i2 in range(l * i1, l * (i1 + 1)):
    for i3 in range(k):
      Q[i2, i1 * k + i3] = X[i2]**i3
'''populating V (constraints)'''
for i in range(s - 1):
  for j in range(k):
    V[2 * i, i * k + j] = X[(i + 1) * l]**j
    V[2 * i, i * k + k + j] = -X[(i + 1) * l]**j
for i in range(s - 1):
  for j in range(1, k):
    V[2 * i + 1, i * k + j] = j * X[(i + 1) * l] ** (j - 1)
    V[2 * i + 1, i * k + k + j] = -j * X[(i + 1) * l] ** (j - 1)


A = direct(Q, V, Y, s)
# A = kkt(Q, V, Y, k, s)

# exit()
'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white"

fig.add_trace(
  go.Scattergl(
    name="Data", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y,
    mode="markers",
    # line=dict(
    #     # size=8,
    #     color="silver",
    #     # showscale=False
    # ),
    visible = "legendonly"
  )
)

Y = eval(A, l, s)
fig.add_trace(
  go.Scatter(
    name="clms", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=Y,
    mode="markers+lines",
    # mode="lines",
    # line=dict(
    #     # size=8,
    #     color="silver",
    #     # showscale=False
    # ),
    # visible = "legendonly"
  )
)

fig.show(config=dict({'scrollZoom': True}))