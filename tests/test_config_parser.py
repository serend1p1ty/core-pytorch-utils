import os
import tempfile

import pytest
import yaml

from cpu import ConfigArgumentParser, save_args


def test_config_parser():
    with tempfile.TemporaryDirectory() as dir:
        parser = ConfigArgumentParser()
        parser.add_argument("-x", action="store_true")
        parser.add_argument("-y", dest="y1", type=int, default=1)
        parser.add_argument("-z", action="append", type=float, default=[2.0])
        parser.add_argument("-k", type=float)

        args = parser.parse_args(["-x", "-y", "3", "-z", "10.0"])
        assert args.x is True
        assert args.y1 == 3
        assert args.z == [2.0, 10.0]
        assert args.k is None

        # do nothing for non-master process
        config_file = os.path.join(dir, "config.yaml")
        save_args(args, config_file, rank=1)
        assert not os.path.exists(config_file)

        # save config for master process
        save_args(args, config_file, rank=0)
        assert os.path.exists(config_file)
        with open(config_file, "r") as f:
            config_vars = yaml.safe_load(f)
        assert config_vars == {"x": True, "y1": 3, "z": [2.0, 10.0], "k": None}

        data = {"x": True, "z": [2.0, 10.0], "k": 3.0}
        with open(config_file, "w") as f:
            yaml.dump(data, f)

        # set the default values to the config file values
        args = parser.parse_args(["-c", config_file])
        assert args.x is True
        assert args.y1 == 1
        assert args.z == [2.0, 10.0]
        assert args.k == 3.0

        args = parser.parse_args(["-y", "5", "-z", "18", "-k", "8"])
        assert args.x is True
        assert args.y1 == 5
        assert args.z == [2.0, 10.0, 18.0]
        assert args.k == 8.0

        # throw an exception when the config file has unexpected keys
        with pytest.raises(Exception) as e_info:
            data = {"z": [2.0, 10.0], "unexp_key": 5}
            with open(config_file, "w") as f:
                yaml.dump(data, f)
            parser.parse_args(["-c", config_file])
        assert e_info.value.args[0] == "Unexpected configuration entry: unexp_key"
