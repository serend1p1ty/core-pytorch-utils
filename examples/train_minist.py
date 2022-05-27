"""The code is modified from: https://github.com/pytorch/examples/blob/main/mnist/main.py"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cpu import ConfigArgumentParser, EvalHook, Trainer, save_args, set_random_seed, setup_logger
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from inference_hook import InferenceHook


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
        # CPU makes some assumptions about the input and output of the model:
        # 1. In the training phase, the model takes the whole batch as input, and output training loss.
        # 2. In the test phase, the model still takes the whole batch as input, but the output is unlimited.
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


def test(model, test_loader, logger):
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
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # 1. Create an argument parser supporting loading YAML configuration file.
    parser = ConfigArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--work-dir", type=str, default="work_dir",
                        help="working directory to save checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=14, metavar="N",
                        help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR",
                        help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M",
                        help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--deterministic", action="store_true",
                        help=("turn on the CUDNN deterministic setting, which "
                              "can slow down your training considerably."))
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    args = parser.parse_args()

    # 2. Perform some basic setup
    save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"))
    logger = setup_logger("train_minist", args.work_dir)
    set_random_seed(args.seed, deterministic=args.deterministic)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 3. Create model, optimizer, lr_scheduler, data_loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    model = Net(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 4. Create Trainer
    trainer = Trainer(model, optimizer, lr_scheduler, train_loader, args.epochs,
                      work_dir=args.work_dir, log_period=args.log_interval)
    trainer.register_hooks([
        EvalHook(1, lambda: test(model, test_loader, logger)),
        # Refer to inference_hook.py
        InferenceHook(test_dataset)
    ])
    trainer.train()


if __name__ == "__main__":
    main()
