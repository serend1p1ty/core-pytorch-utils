import argparse
import logging
import os
import os.path as osp

from yacs.config import CfgNode

__all__ = ["ArgumentHookParser", "default_argparser", "save_config"]

logger = logging.getLogger(__name__)


class ArgumentHookParser(argparse.ArgumentParser):
    """ArgumentParser supporting to register hooks.

    These hooks will be called after :meth:`parse_args`.
    And they are executed in the order they are registered.

    Example::

        >>> def check(args):
                assert args.path
        >>> parser = ArgumentHookParser()
        >>> parser.add_argument("--path")
        >>> parser.register_hooks(check)
        >>> parser.parse_args()
    """

    def __init__(self, *args, **kwargs):
        self._hooks = kwargs.pop("hooks", [])
        super().__init__(args, kwargs)

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        for hook in self._hooks:
            hook(args)
        return args

    def register_hooks(self, hooks):
        if not isinstance(hooks, list):
            hooks = [hooks]
        self._hooks.extend(hooks)


def default_argparser():
    """Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = ArgumentHookParser()
    parser.add_argument(
        "--config-file", type=str, default="", help="Path of the configuration file."
    )
    parser.add_argument("--resume", action="store_true", help="Whether to resume from a checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Perform evaluation only.")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="Path of the checkpoint to resume or evaluate."
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="",
        help=(
            "Path of the working directory. 'args.config_file' will be "
            "set to 'work_dir/config.yaml', and 'args.checkpoint' will "
            "be set to 'work_dir/checkpoints/latest.pth'"
        ),
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help=(
            "Modify config options at the end of the command, "
            "using space-separated 'PATH.KEY VALUE' pairs."
        ),
    )

    def check_and_init(args):
        # check
        if args.resume or args.eval_only:
            assert args.checkpoint or args.work_dir
        if args.checkpoint or args.work_dir:
            assert args.resume or args.eval_only
        # init
        if args.work_dir:
            args.config_file = osp.join(args.work_dir, "config.yaml")
            args.checkpoint = osp.join(args.work_dir, "checkpoints", "latest.pth")

    parser.register_hooks(check_and_init)
    return parser


def save_config(cfg: CfgNode, output: str):
    """Save :class:`yacs.config.CfgNode` to a ``.yaml`` file.

    Args:
        cfg (CfgNode): The config to be saved.
        output (str): A file name or a directory. If ends with ``.yaml``, assumed to
            be a file name. Otherwise, the config will be saved to ``output/config.yaml``.
    """
    if output.endswith(".yaml"):
        filename = output
    else:
        filename = osp.join(output, "config.yaml")
    os.makedirs(osp.dirname(osp.abspath(filename)), exist_ok=True)

    with open(filename, "w") as f:
        f.write(cfg.dump())
    logger.info(f"Full config is saved to {filename}")
