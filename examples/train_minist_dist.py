"""PyTorch MNIST distributed training example.

The code supports both single-gpu and multi-gpu training.
It can be used as your template to start a new project.
"""
import logging
import os

import torch
import torch.optim as optim
from cpu import EvalHook, Trainer, init_distributed, save_args, set_random_seed, setup_logger
from inference_hook import InferenceHook
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from train_minist import Net, build_dataset, parse_args, test

logger = logging.getLogger(__name__)


def main():
    # 1. Create an argument parser supporting loading YAML configuration file
    args = parse_args()

    # 2. Basic setup
    rank, local_rank, world_size = init_distributed()  # [Step 1]
    is_distributed = world_size > 1

    setup_logger(output_dir=args.work_dir, rank=rank)
    save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"), rank=rank)
    # Make sure each worker has a different, yet deterministic seed
    # See: https://github.com/open-mmlab/mmdetection/pull/7432
    set_random_seed(None if args.seed < 0 else args.seed + rank, args.deterministic)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 3. Create data_loader, model, optimizer, lr_scheduler
    train_dataset, test_dataset = build_dataset(args.dataset_dir)
    if is_distributed:  # [Step 2]
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    # DistributedSampler will do shuffle, so we set `shuffle=False` when using DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               shuffle=(train_sampler is None))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    model = Net(device)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])  # [Step 3]

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 4. Create Trainer
    trainer = Trainer(model, optimizer, lr_scheduler, train_loader, args.epochs,
                      work_dir=args.work_dir, log_period=args.log_interval)
    trainer.register_hooks([
        EvalHook(1, lambda: test(model, test_loader)),
        # Refer to inference_hook.py
        InferenceHook(test_dataset)
    ] if rank == 0 else [])  # [Step 4]
    trainer.train()


if __name__ == "__main__":
    main()
