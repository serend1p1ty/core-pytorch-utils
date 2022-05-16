import logging
import os
import sys
from typing import Optional

from termcolor import colored

logger_initialized = {}


def setup_logger(
    name: Optional[str] = None,
    output_dir: Optional[str] = None,
    rank: int = 0,
    log_level: int = logging.DEBUG,
    color: bool = True,
) -> logging.Logger:
    """Initialize the logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, only the logger of the master process
    is added console handler. If ``output_dir`` is specified, all loggers will be added file handler.

    Args:
        name (str): Logger name. Defaults to None to setup root logger.
        output_dir (str): The directory to save log.
        rank (int): Process rank in the distributed training. Defaults to 0.
        log_level (int): Verbosity level of the logger. Defaults to ``logging.DEBUG``.
        color (bool): If True, color the output. Defaults to True.

    Returns:
        logging.Logger: A initialized logger.
    """
    if name in logger_initialized:
        return logger_initialized[name]

    # get root logger if name is None
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # the messages of this logger will not be propagated to its parent
    logger.propagate = False

    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = colored("[%(asctime)s %(name)s]", "green") + \
                colored("(%(filename)s %(lineno)d)", "yellow") + ": %(levelname)s %(message)s"

    # create console handler for master process
    if rank == 0:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(fmt=color_fmt if color else fmt, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, f"log_rank{rank}.txt"))
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger_initialized[name] = logger
    return logger
