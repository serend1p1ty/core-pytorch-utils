import os
from typing import Any, Dict, List, Optional

from ..checkpoint import Checkpointer
from .hookbase import HookBase


class CheckpointerHook(HookBase):
    """Save checkpoints periodically.

    Save checkpoint, if current epoch is a multiple of period or ``max_epochs`` is reached.
    """

    def __init__(
        self, checkpointer: Checkpointer, period: int, max_to_keep: Optional[int] = None
    ) -> None:
        """
        Args:
            checkpointer: The checkpointer object used to save checkpoints.
            period (int): The period to save checkpoint.
            max_to_keep (int): Maximum number of most current checkpoints to keep,
                previous checkpoints will be deleted.
        """
        self.checkpointer = checkpointer
        self.period = period
        if max_to_keep is not None:
            assert max_to_keep > 0
        self.max_to_keep = max_to_keep

        self.recent_checkpoints: List[str] = []
        self.max_epochs = self.trainer.max_epochs

    def after_epoch(self) -> None:
        epoch = self.trainer.epoch  # ranged in [0, max_epochs - 1]
        if (epoch + 1) % self.period == 0 or epoch == self.max_epochs - 1:
            checkpoint_name = f"epoch_{epoch}.pth"
            self.checkpointer.save(checkpoint_name, epoch=epoch)

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(checkpoint_name)
                if len(self.recent_checkpoints) > self.max_to_keep:
                    # delete the oldest checkpointer
                    file_name = self.recent_checkpoints.pop(0)
                    file_path = self.checkpointer.get_path(file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "checkpointer"}
