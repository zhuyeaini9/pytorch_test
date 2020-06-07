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
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.01)
optimizer2 = optim.Adam(net.parameters(), lr=0.01)
do_other = False
while True:
    optimizer.zero_grad()

    outputs = net(rd)
    sm = nn.Softmax(dim=1)
    smOut = sm(outputs)
    print(smOut)

    reward = torch.tensor(-1.5)

    if smOut[0][0]<0.2:
        do_other = True
        print('do other')

    if do_other:
        loss = -torch.log(smOut[0][1]) * reward
        loss.backward()
        optimizer2.step()
    else:
        loss = -torch.log(smOut[0][0]) * reward
        loss.backward()
        optimizer.step()





    # if loss.item() < -torch.log(torch.tensor(0.1)) * reward:
    #     loss = torch.max(loss, -torch.log(torch.tensor(0.1)) * reward)
    # loss = torch.min(loss, torch.tensor(-100.0))

    # criterion = nn.CrossEntropyLoss()

    # loss = criterion(outputs, torch.tensor([[0, 1]]))

