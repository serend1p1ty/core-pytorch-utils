import logging
import os
import random
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from tabulate import tabulate

__all__ = ["collect_env", "set_random_seed", "symlink"]

logger = logging.getLogger(__name__)


def collect_env() -> str:
    """Collect the information of the running environments.

    The following information are contained.

        - sys.platform: The value of ``sys.platform``.
        - Python: Python version.
        - Numpy: Numpy version.
        - CUDA available: Bool, indicating if CUDA is available.
        - GPU devices: Device type of each GPU.
        - PyTorch: PyTorch version.
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

    return tabulate(env_info)


def set_random_seed(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): If None or negative, use a generated seed.
        deterministic (bool): If True, set the deterministic option for CUDNN backend.
    """
    if seed is None or seed < 0:
        new_seed = np.random.randint(2**32)
        logger.info(f"Got invalid seed: {seed}, will use the randomly generated seed: {new_seed}")
        seed = new_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set random seed to {seed}.")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("The CUDNN is set to deterministic. This will increase reproducibility, "
                    "but may slow down your training considerably.")


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
