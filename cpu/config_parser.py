import argparse
import logging
import sys
from argparse import Namespace, _AppendAction
from copy import deepcopy

import yaml

logger = logging.getLogger(__name__)


class ConfigArgumentParser:
    """Argument parser that supports loading a YAML configuration file.
    Code is modified from: https://gist.github.com/multun/ccf5a8b855de7c50968aac127bc5605b
    """

    def __init__(self, *args, **kwargs):
        self.options = []
        self.args = args
        self.kwargs = kwargs

    def add_argument(self, *args, **kwargs):
        self.options.append((args, kwargs))

    def parse_args(self, args=None):
        conf_parser = argparse.ArgumentParser(add_help=False)
        conf_parser.add_argument(
            "-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE"
        )

        res, remaining_argv = conf_parser.parse_known_args(args)

        config_vars = {}
        if res.config is not None:
            with open(res.config, "r") as stream:
                config_vars = yaml.safe_load(stream)

        parser = argparse.ArgumentParser(
            *self.args,
            # Inherit options from config_parser
            parents=[conf_parser],
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **self.kwargs,
        )

        for opt_args, opt_kwargs in self.options:
            parser_arg = parser.add_argument(*opt_args, **opt_kwargs)
            if parser_arg.dest in config_vars:
                config_default = config_vars.pop(parser_arg.dest)

                if parser_arg.type is not None:
                    expected_type = parser_arg.type
                else:
                    expected_type = type(parser_arg.default)

                if isinstance(parser_arg, _AppendAction) or parser_arg.nargs == "+":
                    if not isinstance(config_default, list) or not all(
                        isinstance(var, expected_type) for var in config_default
                    ):
                        parser.error(
                            "{} is expected to be a list of {}, but got {}".format(
                                parser_arg.dest, expected_type, config_default
                            )
                        )
                else:
                    if not isinstance(config_default, expected_type):
                        parser.error(
                            "{} is expected to be {}, but got {}".format(
                                parser_arg.dest, expected_type, config_default
                            )
                        )

                parser_arg.default = config_default

        if config_vars:
            parser.error("unexpected configuration entries: " + ", ".join(config_vars))

        return parser.parse_args(remaining_argv)


def save_args(args: Namespace, filepath: str) -> None:
    """Save args (excluding ``config`` field) to a ``.yaml`` file.

    Args:
        args (Namespace): The parsed arguments to be saved.
        filepath (str): A filepath ends with ".yaml".
    """
    assert filepath.endswith(".yaml")
    save_dict = deepcopy(args.__dict__)
    save_dict.pop("config")
    with open(filepath, "w") as f:
        yaml.dump(save_dict, f)
    logger.info(f"Args is saved to {filepath}")
