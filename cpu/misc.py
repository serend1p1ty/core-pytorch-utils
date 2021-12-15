import logging
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from tabulate import tabulate

logger = logging.getLogger(__name__)

__all__ = ["set_random_seed", "collect_env", "symlink", "create_small_table"]


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


def set_random_seed(seed: int, rank: int = 0) -> None:
    """Set random seed.

    Args:
        seed (int): Nonnegative integer.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert seed >= 0, f"Got invalid seed value {seed}."
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def symlink(src: str, dst: str, overwrite: bool = True, **kwargs) -> None:
    """Create a symlink, dst -> src.

    Args:
        src (str): Path to source.
        dst (str): Path to target.
        overwrite (bool): If True, remove existed target. Defaults to True.
    """
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def create_small_table(small_dict: Dict[str, Any]) -> str:
    """Create a small table using the keys of ``small_dict`` as headers.
    This is only suitable for small dictionaries.

    Args:
        small_dict (dict): A result dictionary of only a few items.

    Returns:
        str: The table as a string.
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
