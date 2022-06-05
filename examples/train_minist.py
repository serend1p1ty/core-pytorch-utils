"""PyTorch MNIST example.

The code is modified from: https://github.com/pytorch/examples/blob/main/mnist/main.py
It only supports single-gpu training.
"""
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from inference_hook import InferenceHook
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from cpu import ConfigArgumentParser, EvalHook, Trainer, save_args, set_random_seed, setup_logger

logger = logging.getLogger(__name__)


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
        # CPU has the following assumptions about the input and output of the model:
        # 1. In training phase: the model takes the whole batch as input,
        # and outputs training loss.
        # 2. In test phase: the model still takes the whole batch as input,
        # but the output is unlimited.
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


def test(model, test_loader):
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

    logger.info("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def parse_args():
    parser = ConfigArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--work-dir", type=str, default="work_dir", metavar="DIR",
                        help="Directory to save checkpoints and logs (default: 'work_dir').")
    parser.add_argument("--dataset-dir", type=str, default="../data", metavar="DIR",
                        help="Directory to save dataset (default: './data').")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="Input batch size for training (default: 64).")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="Input batch size for test (default: 1000).")
    parser.add_argument("--epochs", type=int, default=14, metavar="N",
                        help="Number of epochs to train (default: 14).")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR",
                        help="Learning rate (default: 1.0).")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M",
                        help="Learning rate step gamma (default: 0.7).")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disables CUDA training.")
    parser.add_argument("--seed", type=int, default=-1, metavar="S",
                        help="Random seed, set to negative to randomize everything (default: -1).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Turn on the CUDNN deterministic setting.")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="Interval for logging to console and tensorboard (default: 10).")
    return parser.parse_args()


def build_dataset(dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(dir, train=False, transform=transform)
    return train_dataset, test_dataset


def build_dataloader(args):
    train_dataset, test_dataset = build_dataset(args.dataset_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    return train_loader, test_loader


def main():
    # 1. Create an argument parser supporting loading YAML configuration file
    args = parse_args()

    # 2. Basic setup
    setup_logger(output_dir=args.work_dir)
    save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"))
    set_random_seed(args.seed, args.deterministic)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 3. Create data_loader, model, optimizer, lr_scheduler
    train_loader, test_loader = build_dataloader(args)
    model = Net(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 4. Create Trainer
    trainer = Trainer(model, optimizer, lr_scheduler, train_loader, args.epochs,
                      work_dir=args.work_dir, log_period=args.log_interval)
    trainer.register_hooks([
        EvalHook(1, lambda: test(model, test_loader)),
        # Refer to inference_hook.py
        InferenceHook(test_loader.dataset)
    ])
    trainer.train()


if __name__ == "__main__":
    main()
