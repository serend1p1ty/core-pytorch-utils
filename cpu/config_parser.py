import argparse
import sys

import yaml


class ConfigArgumentParser:
    """Argument parser that supports loading a YAML configuration file.
    Code are borrowed from: https://gist.github.com/multun/ccf5a8b855de7c50968aac127bc5605b
    """

    def __init__(self, *args, **kwargs):
        self.options = []
        self.args = args
        self.kwargs = kwargs

    def add_argument(self, *args, **kwargs):
        self.options.append((args, kwargs))

    def parse_args(self, args=None):
        if args is None:
            args = sys.argv[1:]

        conf_parser = argparse.ArgumentParser(add_help=False)
        conf_parser.add_argument(
            "--config", default=None, help="where to load YAML configuration", metavar="FILE"
        )

        res, remaining_argv = conf_parser.parse_known_args(args)

        config_vars = {}
        if res.config is not None:
            with open(res.config, "r") as stream:
                config_vars = yaml.load(stream)

        parser = argparse.ArgumentParser(
            *self.args,
            # Inherit options from config_parser
            parents=[conf_parser],
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **self.kwargs,
        )

        for opt_args, opt_kwargs in self.options:
            opt_parser = parser.add_argument(*opt_args, **opt_kwargs)
            if opt_parser.dest in config_vars:
                config_default = config_vars.pop(opt_parser.dest)

                expected_type = str
                if opt_parser.type is not None:
                    expected_type = opt_parser.type
                if not isinstance(config_default, expected_type):
                    parser.error(
                        "YAML configuration entry {} "
                        "does not have type {}".format(opt_parser.dest, expected_type)
                    )

                opt_parser.default = config_default

        if config_vars:
            parser.error("unexpected configuration entries: " + ", ".join(config_vars))

        return parser.parse_args(remaining_argv)
