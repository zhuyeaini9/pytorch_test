import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.distributions import Categorical, normal, MultivariateNormal
import math

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc3(x))
        var = F.softplus(self.fc4(x))
        return mu,var


net = Net()
rd = torch.tensor([[0.8, 0.2, 0.3, 0.4]])
optimizer = optim.Adam(net.parameters(), lr=0.01)


while True:
    means,stds = net(rd)
    action_distribution = normal.Normal(means.squeeze(0), stds.squeeze(0))
    sample = action_distribution.sample()
    act_log = action_distribution.log_prob(sample)
    # print(outputs,act_log)

    reward = torch.tensor(1.5)

    loss = -act_log * reward
    print(sample,math.pow(math.e,act_log.data.numpy()),[means.data,stds.data], -act_log)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


