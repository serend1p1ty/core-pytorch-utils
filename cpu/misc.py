import logging
import os
import random
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["get_time_str", "highlight", "set_random_seed"]


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
