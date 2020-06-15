import torch
from torch.distributions import Categorical, normal
import numpy as np
import math
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


t = torch.tensor([1.0, 2.0, 3.0, 4.0])
n = Net()
op = optim.Adam(params=n.parameters(), lr=0.01)

while True:
    loss = -n(t)
    print(loss)
    op.zero_grad()
    loss.backward()
    op.step()
