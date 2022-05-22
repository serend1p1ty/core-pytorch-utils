The tutorial will show you how to install and use the CPU library to train a network.

## Prerequisites

PyTorch 1.6+ to use `torch.cuda.amp`.

## Installation

```
pip install -r requirements.txt
pip install -v -e .
```

## Train a CNN on MINIST

We slightly modify the official PyTorch MINIST [demo](https://github.com/pytorch/examples/edit/main/mnist/main.py) to demonstrate how to use our Trainer. The code is placed in the [example/](https://github.com/serend1p1ty/core-pytorch-utils/tree/main/example) folder.
