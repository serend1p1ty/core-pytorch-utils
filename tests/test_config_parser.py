import os
import tempfile

import yaml

from cpu import ConfigArgumentParser


def test_config_parser():
    with tempfile.TemporaryDirectory() as dir:
        parser = ConfigArgumentParser()
        parser.add_argument("-x", "--arg-x", action="store_true")
        parser.add_argument("-y", "--arg-y", dest="y1", type=int, default=1)
        parser.add_argument("--arg-z", action="append", type=float, default=[2.0])
        parser.add_argument("-k", type=float)

        args = parser.parse_args(["--arg-x", "-y", "3", "--arg-z", "10.0"])
        assert args.arg_x is True
        assert args.y1 == 3
        assert args.arg_z == [2.0, 10.0]
        assert args.k is None

        data = {"arg_x": True, "arg_z": [2.0, 10.0], "k": 3.0}
        config_file = os.path.join(dir, "config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(data, f)

        args = parser.parse_args(["-c", config_file])
        assert args.arg_x is True
        assert args.y1 == 1
        assert args.arg_z == [2.0, 10.0]
        assert args.k == 3.0

        args = parser.parse_args(["-c", "config.yaml", "-y", "5", "--arg-z", "18", "-k", "8"])
        assert args.arg_x is True
        assert args.y1 == 5
        assert args.arg_z == [2.0, 10.0, 18.0]
        assert args.k == 8.0
