import os.path as osp
import subprocess
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from tabulate import tabulate
from torch.utils.cpp_extension import CUDA_HOME


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
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info.append(("GCC", "n/a"))

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
