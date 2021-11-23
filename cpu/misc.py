import datetime
import logging
import os
import random
import sys
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from tabulate import tabulate

logger = logging.getLogger(__name__)

__all__ = [
    "get_time_str",
    "highlight",
    "set_random_seed",
    "collect_env",
    "symlink",
    "create_small_table",
]


def collect_env() -> str:
    """Collect the information of the running environments.

    The following information are contained.

        - sys.platform: The variable of ``sys.platform``.
        - Python: Python version.
        - Numpy: Numpy version.
        - CUDA available: Bool, indicating if CUDA is available.
        - GPU devices: Device type of each GPU.
        - PyTorch: PyTorch version.
        - PyTorch compiling details: The output of ``torch.__config__.show()``.
        - TorchVision (optional): TorchVision version.
        - OpenCV (optional): OpenCV version.

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

    env_info.append(("PyTorch", torch.__version__))

    try:
        import torchvision

        env_info.append(("TorchVision", torchvision.__version__))
    except ModuleNotFoundError:
        pass

    try:
        import cv2

        env_info.append(("OpenCV", cv2.__version__))
    except ModuleNotFoundError:
        pass

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


def create_small_table(small_dict):
    """Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table
