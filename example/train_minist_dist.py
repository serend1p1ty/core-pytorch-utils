import os
import torch
import torch.optim as optim
from cpu import (
    ConfigArgumentParser,
    EvalHook,
    Trainer,
    init_distributed,
    save_args,
    set_random_seed,
    setup_logger,
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from inference_hook import InferenceHook
from train_minist import Net, test


def main():
    # 1. Create an argument parser supporting loading YAML configuration file.
    parser = ConfigArgumentParser(description="Distributed training Example")
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
    rank, local_rank, world_size = init_distributed()  # [Step 1]
    is_dist = world_size > 1

    save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"), rank=rank)
    logger = setup_logger("train_minist", args.work_dir, rank=rank)
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

    if is_dist:  # [Step 2]
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    model = Net(device)
    if is_dist:
        model = DistributedDataParallel(model, device_ids=[local_rank])  # [Step 3]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 4. Create Trainer
    trainer = Trainer(model, optimizer, lr_scheduler, train_loader, args.epochs,
                      work_dir=args.work_dir, log_period=args.log_interval)
    trainer.register_hooks([
        EvalHook(1, lambda: test(model, test_loader, logger)),
        # Refer to inference_hook.py
        InferenceHook(test_dataset)
    ] if rank == 0 else [])  # [Step 4]
    trainer.train()


if __name__ == "__main__":
    main()
