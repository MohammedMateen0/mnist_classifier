import torch

import torch.nn as nn

class MNISTNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(784,256)
    self.bn1=nn.BatchNorm1d(256)
    self.fc2=nn.Linear(256,128)
    self.bn2=nn.BatchNorm1d(128)
    self.dropout=nn.Dropout(0.3)
    self.fc3=nn.Linear(128,10)
    nn.init.kaiming_normal_(
        self.fc1.weight,
        nonlinearity='relu'
    )
    nn.init.kaiming_normal_(
        self.fc2.weight,
        nonlinearity='relu'
    )
  def forward(self,x):
    x=x.view(
        x.size(0),
        -1
    )
    x=self.fc1(x)
    x=self.bn1(x)
    x=torch.relu(x)
    x=self.dropout(x)
    x=self.fc2(x)
    x=self.bn2(x)
    x=torch.relu(x)
    x=self.dropout(x)
    x=self.fc3(x)
    return x