import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from cpu import HookBase, Trainer, set_random_seed


class _EvalHook(HookBase):

    def __init__(self, eval_func):
        self.eval_func = eval_func
        self.all_losses = []
        self.all_test_losses = []
        self.all_accuracy = []

    def after_iter(self):
        self.all_losses.append(self.metric_storage["total_loss"].latest)

    def after_epoch(self):
        test_loss, accuracy = self.eval_func()
        self.all_test_losses.append(test_loss)
        self.all_accuracy.append(accuracy)


class Net(nn.Module):

    def __init__(self, device):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.device = device
        self.to(device)

    def forward(self, data):
        img, target = data
        img = img.to(self.device)
        target = target.to(self.device)

        x = self.conv1(img)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        if self.training:
            loss = F.nll_loss(output, target)
            return loss
        return output


def _test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, target in test_loader:
            output = model((img, target)).cpu()
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def _plain_train_loop(model, train_loader, test_loader, optimizer, lr_scheduler, max_epochs):
    all_losses = []
    all_test_losses = []
    all_accuracy = []
    for _ in range(max_epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
        lr_scheduler.step()

        test_loss, accuracy = _test(model, test_loader)
        all_test_losses.append(test_loss)
        all_accuracy.append(accuracy)
    return all_losses, all_test_losses, all_accuracy


def _setup(dir, device, train_batch_size=64, test_batch_size=1000):
    set_random_seed(seed=1, deterministic=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)

    model = Net(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    return model, optimizer, lr_scheduler, train_loader, test_loader


def test_minist_training(device="cpu", max_epochs=1):
    with tempfile.TemporaryDirectory() as dir:
        model, optimizer, lr_scheduler, train_loader, test_loader = _setup(dir, device)
        all_losses, all_test_losses, all_accuracy = _plain_train_loop(model, train_loader,
                                                                      test_loader, optimizer,
                                                                      lr_scheduler, max_epochs)

        model, optimizer, lr_scheduler, train_loader, test_loader = _setup(dir, device)
        trainer = Trainer(model, optimizer, lr_scheduler, train_loader, max_epochs, dir)
        hook = _EvalHook(lambda: _test(model, test_loader))
        trainer.register_hook(hook)
        trainer.train()
        assert np.allclose(all_losses, hook.all_losses)
        assert np.allclose(all_test_losses, hook.all_test_losses)
        assert np.allclose(all_accuracy, hook.all_accuracy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The test needs cuda")
def test_minist_training_cuda():
    test_minist_training(device="cuda")
