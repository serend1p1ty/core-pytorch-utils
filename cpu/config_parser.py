import argparse
import yaml


class ConfigArgumentParser(argparse.ArgumentParser):
    """Argument parser that supports loading a YAML configuration file.

    A small issue: config file values are processed using ArgumentParser.set_defaults(..)
    which means "required" and "choices" are not handled as expected. For example, if you
    specify a required value in a config file, you still have to specify it again on the
    command line. The ``ConfigArgParse`` library (http://pypi.python.org/pypi/ConfigArgParse)
    can be used as a substitute.
    """

    def __init__(self, *args, **kwargs):
        self.config_parser = argparse.ArgumentParser(add_help=False)
        self.config_parser.add_argument("-c", "--config", default=None, metavar="FILE",
                                        help="where to load YAML configuration")
        self.option_names = []
        super().__init__(*args,
                         # Inherit options from config_parser
                         parents=[self.config_parser],
                         # Don't mess with format of description
                         formatter_class=argparse.RawDescriptionHelpFormatter,
                         **kwargs)

    def add_argument(self, *args, **kwargs):
        arg = super().add_argument(*args, **kwargs)
        self.option_names.append(arg.dest)
        return arg

    def parse_args(self, args=None):
        res, remaining_argv = self.config_parser.parse_known_args(args)

        if res.config is not None:
            with open(res.config, "r") as f:
                config_vars = yaml.safe_load(f)
            for key in config_vars:
                if key not in self.option_names:
                    self.error(f"unexpected configuration entry: {key}")
            self.set_defaults(**config_vars)

        return super().parse_args(remaining_argv)
