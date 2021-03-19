import numpy as np
import plotly.graph_objects as go

m = 5000
k = 3
X = np.arange(m)
Y = np.cos(X/100) + np.random.randn(m)/5 # b


'''Least Squares'''

Q = np.zeros((m, k + 1))

for i in range(m):
  for j in range(k + 1):
    Q[i, j] = i**j

A = np.linalg.inv(Q.T @ Q) @ Q.T @ Y

A_ = np.linalg.lstsq(Q, Y)[0]

# print(A)
# print(A_)

print(np.allclose(A, A_))
# exit()

Y_lms = np.zeros((m))

for i in range(m):
  for j in range(k + 1):
    Y_lms[i] += A[j] * i**j



'''Constrained Least Squares (with q intervals and k-1 polynomial)'''
s = 10
l = m // s
Q = np.zeros((m, k * s)) # A
V = np.zeros((2 * (s-1), k * s)) # C

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


'''solving systems'''
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


a = np.vstack([a1, a2, a3])

# print(Y.shape)
y = np.hstack([np.zeros(2 * (s-1)), Y, np.zeros(k * s)]).T

sol = np.linalg.solve(a, y)[:s*k]

print(sol, "\n")

'''solving inverse'''

QTQinv = np.linalg.inv(Q.T @ Q)
tau = np.linalg.inv(V @ QTQinv @ V.T)
QTY = Q.T @ Y

A = QTQinv @ (QTY - V.T @ tau @ V @ QTQinv @ QTY)
print(A)
A = np.reshape(A, (s, -1))

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


# exit()
'''============================================================================'''
'''                                    PLOT                                    '''
'''============================================================================'''
fig = go.Figure()
fig.layout.template ="plotly_white"

fig.add_trace(
  go.Scatter(
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

fig.add_trace(
  go.Scatter(
    name="lms", # <|<|<|<|<|<|<|<|<|<|<|<|
    x=X,
    y=Y_lms,
    mode="markers+lines",
    # mode="lines",
    # line=dict(
    #     # size=8,
    #     color="silver",
    #     # showscale=False
    # ),
    visible = "legendonly"
  )
)

fig.add_trace(
  go.Scatter(
    name="clms", # <|<|<|<|<|<|<|<|<|<|<|<|
    # x=X,
    y=Y_clms,
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