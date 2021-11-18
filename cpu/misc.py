import argparse
import datetime
import logging
import os
import os.path as osp
import random
import subprocess
import sys
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch
from tabulate import tabulate
from torch.utils.cpp_extension import CUDA_HOME
from yacs.config import CfgNode

logger = logging.getLogger(__name__)

__all__ = [
    "get_time_str",
    "highlight",
    "set_random_seed",
    "collect_env",
    "symlink",
    "default_argparser",
    "save_config",
    "merge_from_args",
]


def collect_env() -> str:
    """Collect the information of the running environments.

    The following information are contained.

        - sys.platform: The variable of ``sys.platform``.
        - Python: Python version.
        - Numpy: Numpy version.
        - CUDA available: Bool, indicating if CUDA is available.
        - GPU devices: Device type of each GPU.
        - CUDA_HOME (optional): The env var ``CUDA_HOME``.
        - NVCC (optional): NVCC version.
        - GCC: GCC version, "n/a" if GCC is not installed.
        - PyTorch: PyTorch version.
        - PyTorch compiling details: The output of ``torch.__config__.show()``.
        - TorchVision (optional): TorchVision version.
        - OpenCV: OpenCV version.

    Returns:
        str: A string describing the running environment.
    """
    env_info = []
    env_info.append(("sys.platform", sys.platform))
    env_info.append(("Python", sys.version.replace("\n", "")))
    env_info.append(("Numpy", np.__version__))

    cuda_available = torch.cuda.is_available()
    env_info.append(("CUDA available", cuda_available))

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info.append(("GPU " + ",".join(device_ids), name))

        env_info.append(("CUDA_HOME", CUDA_HOME))

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, "bin/nvcc")
                nvcc = subprocess.check_output(f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode("utf-8").strip()
            except subprocess.SubprocessError:
                nvcc = "Not Available"
            env_info.append(("NVCC", nvcc))

    try:
        gcc = subprocess.check_output("gcc --version | head -n1", shell=True)
        gcc = gcc.decode("utf-8").strip()
        env_info.append(("GCC", gcc))
    except subprocess.CalledProcessError:
        env_info.append(("GCC", "Not Available"))

    env_info.append(("PyTorch", torch.__version__))

    try:
        import torchvision

        env_info.append(("TorchVision", torchvision.__version__))
    except ModuleNotFoundError:
        pass

    env_info.append(("OpenCV", cv2.__version__))
    torch_config = torch.__config__.show()
    env_str = tabulate(env_info) + "\n" + torch_config
    return env_str


def get_time_str() -> str:
    """Get formatted time string.

    Returns:
        str: Time string similar to "20210923_192721"
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def highlight(code: str, filename: str) -> str:
    """Highlight the code.

    Currently only ``.py`` and ``.yaml`` are supported.

    Example::

    >>> highlight(open("code.py", "r").read(), ".py")

    Args:
        code (str): Code string.
        filename (str): Full file name or extension name.

    Returns:
        str: Highlighted code string.
    """
    if not filename.endswith(".py") and not filename.endswith(".yaml"):
        logger.warning("Cannot highlight code, only .py and .yaml are supported.")
        return code

    try:
        import pygments
    except ImportError:
        logger.warning(
            "Cannot highlight code because of the ImportError of `pygments`. "
            "Try `pip install pygments`."
        )
        return code

    from pygments.formatters import Terminal256Formatter
    from pygments.lexers import Python3Lexer, YamlLexer

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def set_random_seed(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): If None or negative, will use a generated seed.
        deterministic (bool, optional): If True, CUDA will select the same and deterministic
            convolution algorithm each time an application is run.
    """
    if seed is None or seed < 0:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger.info(f"Using a generated random seed {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # While disabling CUDA convolution benchmarking ensures that CUDA
        # selects the same convolution algorithm each time an application
        # is run, that algorithm itself may be nondeterministic, unless
        # `torch.backends.cudnn.deterministic = True` is set.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def symlink(src: str, dst: str, overwrite: bool = True, **kwargs) -> None:
    """Create a symlink, dst -> src.

    Args:
        src (str): Path to source.
        dst (str): Path to target.
        overwrite (bool, optional): If True, remove existed target. Defaults to True.
    """
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def default_argparser():
    """Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", help="Path of the configuration file.")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from a checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Perform evaluation only.")
    parser.add_argument(
        "--checkpoint", default="", help="Path of the checkpoint to resume or evaluate."
    )
    parser.add_argument(
        "opts",
        help=(
            "Modify config options at the end of the command, "
            "using space-separated 'PATH.KEY VALUE' pairs."
        ),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def save_config(cfg: CfgNode, output: str):
    """Save :class:`yacs.config.CfgNode` to a ``.yaml`` file.

    Args:
        cfg (CfgNode): The config to be saved.
        output (str): A file name or a directory. If ends with ``.yaml``, assumed to
            be a file name. Otherwise, the config will be saved to ``output/config.yaml``.
    """
    if output.endswith(".yaml"):
        filename = output
    else:
        filename = osp.join(output, "config.yaml")
    os.makedirs(osp.dirname(osp.abspath(filename)), exist_ok=True)

    with open(filename, "w") as f:
        f.write(cfg.dump())
    logger.info(f"Full config is saved to {filename}")


def merge_from_args(cfg: CfgNode, args):
    """Merge config from the arguments parsed by default parser.

    Args:
        cfg (CfgNode): The config to be merged.
        args ([type]): Default argument parser returned by :meth:`default_argparser`.

    Returns:
        CfgNode: Merged config.
    """
    assert hasattr(args, "config_file") and hasattr(args, "opts")
    if args.config_file:
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, highlight(open(args.config_file, "r").read(), args.config_file)
            )
        )
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    logger.info(f"Running with full config:\n{highlight(cfg.dump(), '.yaml')}")
    return cfg
