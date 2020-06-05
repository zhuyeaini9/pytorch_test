import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim


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
while True:
    outputs = net(rd)
    sm = nn.Softmax(dim=1)
    smOut = sm(outputs)
    print(smOut)

    loss = -torch.log(smOut[0][0])

    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # loss = criterion(outputs, torch.tensor([[0, 1]]))
    net.zero_grad()
    loss.backward()
    optimizer.step()
