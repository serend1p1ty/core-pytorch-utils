import logging
import os
from argparse import Namespace

import torch
from yacs.config import CfgNode

from .distributed import is_main_process
from .env import collect_env
from .logger import setup_logger
from .misc import highlight, set_random_seed

logger = logging.getLogger(__name__)


def setup(self, cfg: CfgNode, args: Namespace) -> None:
    """Some basic setups.

    Perform some basic setups at the beginning of a job, including:

    1. Set up the logger.
    2. Log basic information about environment, cmdline arguments, and config.
    3. Backup the config to the output directory.

    Args:
        cfg (CfgNode): An instance of ``yacs.config.CfgNode``.
        args (Namespace): A namespace returned by ``parser.parse_args()``.
    """
    if is_main_process() and self.work_dir:
        os.makedirs(self.work_dir, exist_ok=True)

    # setup root logger
    setup_logger(output=self.work_dir, rank=self.rank)

    logger.info(f"Rank of current process: {self.rank}. World size: {self.world_size}.")
    logger.info(f"Environment info:\n{collect_env()}")
    logger.info("Command line arguments: " + str(args))

    if hasattr(args, "config_file") and args.config_file != "":
        highlighted_config = highlight(open(args.config_file, "r").read(), args.config_file)
        logger.info(f"Contents of args.config_file={args.config_file}:\n{highlighted_config}")

    if is_main_process() and self.work_dir:
        # backup config
        path = os.path.join(self.work_dir, "config.yaml")
        logger.info(f"Running with full config:\n{highlight(cfg.dump(), '.yaml')}")
        with open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each process has a different, yet deterministic seed if specified
    set_random_seed(-1 if cfg.SEED < 0 else cfg.SEED + self.rank)

    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.get("CUDNN_BENCHMARK", False)
