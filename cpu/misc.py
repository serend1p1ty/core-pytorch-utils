import argparse
import logging
import os
import os.path as osp
import random
import subprocess
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from tabulate import tabulate
from torch.utils.cpp_extension import CUDA_HOME

logger = logging.getLogger(__name__)

__all__ = [
    "get_time_str",
    "highlight",
    "set_random_seed",
    "collect_env",
    "symlink",
    "default_argument_parser",
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


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): Seed to be used. Set to negative to randomize everything.
        deterministic (bool, optional): Whether to set the deterministic option for
            CUDNN backend. Defaults to False.
    """
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        logger.warning("Fail to set random seed, as you call this function with negative seed.")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def default_argument_parser():
    """Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", help="Path to config file.")
    parser.add_argument("--resume", default="", help="Resume from the given checkpoint.")
    parser.add_argument("--eval-only", action="store_true", help="Perform evaluation only.")
    parser.add_argument(
        "opts",
        help=(
            "Modify config options at the end of the command,"
            "using space-separated 'PATH.KEY VALUE' pairs."
        ),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser
