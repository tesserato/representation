import torch
import numpy as np


m = 10    # input
n = 5     # output
lr = 0.001 # learning rate

x = np.ones(m)
t = np.ones(n)

x = torch.Tensor(x).view(1,-1)
t = torch.Tensor(t).view(1,-1)
w = torch.autograd.Variable(torch.randn(m, n), requires_grad = True)
b = torch.autograd.Variable(torch.randn(1, n), requires_grad = True)

for step in range(1000):
  y = x @ w + b
  loss = (y - t).pow(2).sum()
  loss.backward()
  print(loss)

  w.data -= lr * w.grad.data
  b.data -= lr * b.grad.data

  w.grad.data.zero_()
  b.grad.data.zero_()



