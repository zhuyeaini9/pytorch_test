import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

# 当reward为正：输出的概率会变大，无限接近1
# 当reward为负：输出的概率会变小知道nan，但实际的强化学习中，概率变小到一定程度就不会被选到了，也就不会变成nan

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
rd = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
optimizer = optim.Adam(net.parameters(), lr=0.01)
while True:
    optimizer.zero_grad()

    outputs = net(rd)
    sm = nn.Softmax(dim=1)
    smOut = sm(outputs)
    print(smOut)

    reward = torch.tensor(-1.5)
    loss = -torch.log(smOut[0][1]) * reward
    loss.backward()
    optimizer.step()




