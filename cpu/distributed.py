"""The code of this module is modified from:

- https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
"""
import functools
import logging
import os
import socket
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch._C._distributed_c10d import ProcessGroup

__all__ = [
    "all_gather", "gather", "reduce_dict", "setup_print_for_distributed", "get_world_size",
    "get_rank", "is_main_process", "init_distributed"
]

logger = logging.getLogger(__name__)


@functools.lru_cache()
def _get_global_gloo_group() -> ProcessGroup:
    """Return a process group based on gloo backend, containing all ranks.
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data: Any, group: Optional[ProcessGroup] = None) -> List[Any]:
    """Run :meth:`all_gather` on arbitrary picklable data (not necessarily tensors).

    Args:
        data: Any picklable object.
        group (ProcessGroup): A torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: List of data gathered from each rank.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data: Any, dst: int = 0, group: Optional[ProcessGroup] = None) -> List[Any]:
    """Run :meth:`gather` on arbitrary picklable data (not necessarily tensors).

    Args:
        data: Any picklable object.
        dst (int): Destination rank.
        group (ProcessGroup): A torch process group. By default, will use a group which
            contains all ranks on ``gloo`` backend.

    Returns:
        list[data]: On ``dst``, a list of data gathered from each rank. Otherwise, an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    if dist.get_rank(group) == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def reduce_dict(input_dict: Dict[str, Tensor], average: bool = True) -> Dict[str, Tensor]:
    """Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.

    Args:
        input_dict (dict): All the values will be reduced.
        average (bool): Whether to do average or sum.

    Returns:
        dict: A dict with the same fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_print_for_distributed(is_master: bool) -> None:
    """This function disables printing when not in master process.

    Args:
        is_master (bool): If the current process is the master process or not.
    """
    import builtins
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def get_world_size() -> int:
    """Return the number of processes in the current process group."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return the rank of the current process in the current process group."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Return if the current process is the master process or not."""
    return get_rank() == 0


def _is_free_port(port: int) -> bool:
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append("localhost")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def init_distributed(auto: bool = False) -> Tuple[int]:
    """Initialize the distributed mode as follows:

    - Initialize the process group, with ``backend="nccl"`` and ``init_method="env://"``.
    - Set correct cuda device.
    - Disable printing when not in master process.

    Args:
        auto (bool): If True, when MASTER_PORT is not free, automatically find a free one.
            Defaults to False.

    Returns:
        tuple: (``rank``, ``local_rank``, ``world_size``)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # launched by `torch.distributed.launch`
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        # launched by slurm
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode.")
        return 0, 0, 1

    assert "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ, (
        "init_method='env://' requires the two environment variables: "
        "MASTER_ADDR and MASTER_PORT.")

    if auto:
        assert os.environ["MASTER_ADDR"] == "127.0.0.1", (
            "`auto` is not supported in multi-machine jobs.")
        port = os.environ["MASTER_PORT"]
        if not _is_free_port(port):
            new_port = _find_free_port()
            print(f"Port {port} is not free, use port {new_port} instead.")
            os.environ["MASTER_PORT"] = new_port

    print(f"| distributed init (rank {rank})", flush=True)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    dist.barrier()
    torch.cuda.set_device(local_rank)
    setup_print_for_distributed(rank == 0)
    return rank, local_rank, world_size
