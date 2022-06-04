import logging
import os
import sys
from typing import Optional

from termcolor import colored

logger_initialized = {}


class _ColorfulFormatter(logging.Formatter):

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.DEBUG:
            prefix = colored("DEBUG", "magenta")
        elif record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(name: Optional[str] = None, output_dir: Optional[str] = None, rank: int = 0,
                 log_level: int = logging.INFO, color: bool = True) -> logging.Logger:
    """Initialize the logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, only the logger of the master
    process is added console handler. If ``output_dir`` is specified, all loggers
    will be added file handler.

    Args:
        name (str): Logger name. Defaults to None to setup root logger.
        output_dir (str): The directory to save log.
        rank (int): Process rank in the distributed training. Defaults to 0.
        log_level (int): Verbosity level of the logger. Defaults to ``logging.INFO``.
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

    formatter = logging.Formatter("[%(asctime)s %(name)s %(levelname)s]: %(message)s",
                                  datefmt="%m/%d %H:%M:%S")
    color_formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s", datefmt="%m/%d %H:%M:%S")

    # create console handler for master process
    if rank == 0:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(color_formatter if color else formatter)
        logger.addHandler(console_handler)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, f"log_rank{rank}.txt"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger_initialized[name] = logger
    return logger
