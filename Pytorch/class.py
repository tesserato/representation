import torch
import numpy as np


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
    h_relu = torch.sigmoid(self.linear1(x))
    h_relu = torch.sigmoid(self.linear2(h_relu))
    y_pred = torch.sigmoid(self.linear3(h_relu))
    self.last_y_pred = y_pred
    return y_pred

  def backward(self, t):
    loss = self.criterion(t, self.last_y_pred)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.data


net = TwoLayerNet(2, 100, 1)



for step in range(10000):
  x = torch.rand(100,2)
  y = net.forward(x)
  t = (x[:,0]).view(-1,1)
  loss = net.backward(t)
  
  if step % 1000 == 0:
    print('loss: ', loss.data)

net.forward(torch.tensor([.5,0]))


