import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(100, 200)

    def forward(self, x):
        return self.fc(self.bn(x))


net = Net()

params = []
for k, v in net.named_parameters():
    if v.requires_grad:
        print(k)
        if "bn" in k:
            params += [{"params": [v], "lr": 0.1}]
        elif "bias" in k:
            params += [{"params": [v], "lr": 0.2}]
        else:
            params += [{"params": [v], "lr": 0.3}]
optimizer = torch.optim.SGD(params)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for i in range(100):
    output = net(torch.rand((10, 3, 100, 100)))
    loss = output.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lr_scheduler.step()

lr_group = []
for group in optimizer.param_groups:
    lr_group.append(group["lr"])
print(lr_group)

print(lr_scheduler.state_dict())
