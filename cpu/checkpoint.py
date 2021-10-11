import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .misc import symlink

logger = logging.getLogger(__name__)


class Checkpointer:
    """A checkpointer that can save/load checkpointable objects.

    It does the following things:

    - Identify :class:`DistributedDataParallel` and save its ``.module`` member variable.
    - Log missing and unexpected keys when loading model weights.
    - Only load specified fields of checkpoint.

    Example::

        >>> checkpointer = Checkpointer("save_dir", model=my_model)
        >>> checkpointer.save("ckpt.pth", epoch=5, version="v1.0")
        >>> checkpointer.load("ckpt.pth")
        {'epoch': 5, 'version': 'v1.0'}
    """

    def __init__(self, save_dir: str, **checkpointables: Dict[str, Any]) -> None:
        """
        Args:
            save_dir (str): The directory to save and find checkpoints.
            checkpointables (Dict[str, Any]): Any checkpointable objects, i.e., objects
                that have the :meth:`state_dict` and :meth:`load_state_dict` method.
        """
        if "model" in checkpointables:
            model = checkpointables["model"]
            if isinstance(model, (DistributedDataParallel, DataParallel)):
                checkpointables["model"] = model.module
        self.checkpointables = checkpointables
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def get_path(self, checkpoint_name: str) -> str:
        """Return the path of the given checkpoint name."""
        return os.path.join(self.save_dir, checkpoint_name)

    def save(self, file_name: str, **extra_data: Dict[str, Any]) -> None:
        """Dump checkpointables to a file.

        Args:
            filename (str): The name of the file to save.
            extra_data (Dict[str, Any]): Extra data to save.
        """
        if not self.save_dir:
            return

        data = {}
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(extra_data)

        file_path = self.get_path(file_name)
        logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

        # tag last checkpoint
        dst_file = self.get_path("latest.pth")
        symlink(file_path, dst_file)

    def load(self, path: str, which_to_load: Optional[List[str]] = None) -> Dict[str, Any]:
        """Load from the given checkpoint.

        Args:
            path (str): Path to the checkpoint. If empty, will not load anything.
            which_to_load (List[str]): List of checkpointable names to load.
                If None, will load all possible checkpointables. Defaults to None.

        Returns:
            dict: Extra data loaded from the checkpoint that has not been processed.
                For example, those saved with :meth:`.save(**extra_data)`.
        """
        if not os.path.isfile(path):
            logger.warning(f"Checkpoint {path} not found! Initializing model from scratch.")
            return {}
        logger.info(f"Loading checkpoint from {path} ...")
        checkpoint = torch.load(path, map_location="cpu")

        for key in self.checkpointables if which_to_load is None else which_to_load:
            assert key in checkpoint, f"Can not find key '{key}' in checkpoint."
            logger.info(f"Loading {key} ...")
            obj = self.checkpointables[key]

            if key == "model":
                # when loading model, log missing/unexpected keys, if exists
                incompatible = obj.load_state_dict(checkpoint.pop(key), strict=False)
                if incompatible.missing_keys:
                    logger.warning(
                        "Encounter missing keys when loading checkpoint:\n"
                        f"{incompatible.missing_keys}"
                    )
                if incompatible.unexpected_keys:
                    logger.warning(
                        "Encounter unexpected keys when loading checkpoint:\n"
                        f"{incompatible.unexpected_keys}"
                    )
            else:
                obj.load_state_dict(checkpoint.pop(key))

        # return unprocessed checkpoint data
        return checkpoint
