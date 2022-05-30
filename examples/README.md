# Train a CNN on MINIST

Take the official PyTorch MINIST [demo](https://github.com/pytorch/examples/edit/main/mnist/main.py) as an entry point, we implement single-gpu / multi-gpu training using our Trainer.

## Prerequisites

1. Install the latest `core-pytorch-utils`.
2. `pip install -r requirements.txt`

## Single-gpu training

In `train_minist.py`, we demonstrates the usage of some common apis provided by `core-pytorch-utils`, e.g., `ConfigArgumentParser`, `save_args`, `setup_logger`, `set_random_seed`.

To show how to customize hooks, we implement a hook (`inference_hook.py`) to visualize some images in training phase.

Run the script by the following command.

```
python example/train_minist.py --config example/config.yaml
```

## Multi-gpu training

Also known as distributed training. We need four steps to make the code support distributed training.

- Step 1: Call `init_distributed()` to initialize the process group. Otherwise, we cannot use `DistributedDataParallel` and those functions provided by `torch.distributed` module.
- Step 2: Create `DistributedSampler` to ensure that there is no overlap between data used by different processes.
- Step 3: Wrap model with `DistributedDataParallel`.
- Step 4: Make some code run only in specific processes. Here are some operations that we want to do only in the master process:
   - Save checkpoint.
   - Console and tensorboard logging.
   - Model evaluation (If your code does not support distributed evaluation).
   - ......

   `core-pytorch-utils` has implemented the first two (checkpointing and logging only for master process), and the rest needs to be implemented by the user.

Run the script by the following command.

```bash
# Assume you use 2 GPUs. Actually batch size = batch size per GPU (32) * number of GPUs (2) = 64
python -m torch.distributed.launch --nproc_per_node $GPU example/train_minist_dist.py --config example/config.yaml --batch-size 32
# OR
torchrun --nproc_per_node $GPU example/train_minist_dist.py --config example/config.yaml --batch-size 32
```