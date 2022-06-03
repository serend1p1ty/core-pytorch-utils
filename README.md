<img src="docs/_static/logo.png" alt="drawing" width="200"/>

[![Docs](https://readthedocs.org/projects/core-pytorch-utils/badge/?version=latest)](https://core-pytorch-utils.readthedocs.io/en/latest/?badge=latest)
[![GithubAction](https://github.com/serend1p1ty/core-pytorch-utils/actions/workflows/ci.yml/badge.svg)](https://github.com/serend1p1ty/core-pytorch-utils/actions)
[![Codecov](https://codecov.io/gh/serend1p1ty/core-pytorch-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/serend1p1ty/core-pytorch-utils)
[![License](https://img.shields.io/github/license/serend1p1ty/core-pytorch-utils.svg)](https://github.com/serend1p1ty/core-pytorch-utils/blob/main/LICENSE)

# Core PyTorch Utils (CPU)

This package is a light-weight core library that provides the most common and essential functionalities shared in various deep learning tasks:

- `Trainer`: does tedious training logic for you.
- `LRWarmupScheduler`: wraps all standard PyTorch LR scheduler to support warmup.
- `ConfigArgumentParser`: provides a argument parser that supports loading a YAML configuration file.
- ......

## Installation

```
git clone https://github.com/serend1p1ty/core-pytorch-utils.git
cd core-pytorch-utils
pip install -r requirements.txt
pip install -v -e .
```

## Getting Started

In [examples/](https://github.com/serend1p1ty/core-pytorch-utils/tree/main/examples) folder, we show how to use our Trainer to train a CNN on MINIST.

It is **strongly** recommended that you run this code before using the CPU library.

## Advanced

Learn more from our [documentaion](https://core-pytorch-utils.readthedocs.io/en/latest/).

## License

CPU is released under the [MIT License](LICENSE).

## Acknowledgments

We refered [mmcv](https://github.com/open-mmlab/mmcv.git), [detectron2](https://github.com/facebookresearch/detectron2.git) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) when develping CPU.
