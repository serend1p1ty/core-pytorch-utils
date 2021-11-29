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


def setup_logger(
    name: Optional[str] = None,
    output: Optional[str] = None,
    log_level: int = logging.DEBUG,
    rank: int = 0,
    color: bool = True,
) -> logging.Logger:
    """Initialize the logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, only the logger of the master process
    may add handlers. A :class:`StreamHandler` will always be added. If ``output`` is specified,
    a :class:`FileHandler` will also be added.

    Here are some common uses. We suppose the project structure is as follows::

        project
        ├── module1
        └── module2

    - Only setup the parent logger (``project``), then all children loggers
      (``project.module1`` and ``project.module2``) will use the handlers of the parent logger.

    Example::

        >>> setup_logger(name="project")
        >>> logging.getLogger("project.module1")
        >>> logging.getLogger("project.module2")

    - Only setup the root logger, then all loggers will use the handlers of the root logger.

    Example::

        >>> setup_logger()
        >>> logging.getLogger(name="project")
        >>> logging.getLogger(name="project.module1")
        >>> logging.getLogger(name="project.module2")

    - Setup all loggers, each logger uses independent handlers.

    Example::

        >>> setup_logger(name="project")
        >>> setup_logger(name="project.module1")
        >>> setup_logger(name="project.module2")

    Args:
        name (str): Logger name. Defaults to None to setup root logger.
        output (str): A file name or a directory to save log. If None, will not save
            log file. If ends with ``.txt`` or ``.log``, assumed to be a file name. Otherwise, logs
            will be saved to ``output/log.txt``. Defaults to None.
        log_level (int): Verbosity level of the logger. Defaults to ``logging.DEBUG``.
        rank (int): Process rank in the distributed training. Defaults to 0.
        color (bool): If True, color the output. Defaults to True.

    Returns:
        logging.Logger: A initialized logger.
    """
    if name in logger_initialized:
        return logger_initialized[name]

    # get root logger if name is None
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # the messages of this logger are not propagated to its parent
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s %(name)s %(levelname)s]: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # stdout and file logging: master only
    if rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(log_level)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if output is not None:
            if output.endswith(".txt") or output.endswith(".log"):
                filename = output
            else:
                filename = os.path.join(output, "log.txt")
            # If a single file name is passed as argument, os.path.dirname() will return an empty
            # string. For example, os.path.dirname("log.txt") == "". This will cause an error
            # in os.makedirs(). So we need to wrap `filename` with os.path.abspath().
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            fh = logging.FileHandler(filename)
            fh.setLevel(log_level)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)

    logger_initialized[name] = logger
    return logger
