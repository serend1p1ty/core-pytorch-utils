import os
import os.path as osp
from typing import Any, Dict, List, Optional

from .hookbase import HookBase


class CheckpointHook(HookBase):
    """Save checkpoint periodically.

    Save checkpoint, if current epoch / iteration is a multiple of ``period`` or
    ``max_epochs`` / ``max_iters`` is reached.

    Args:
        period (int): Save checkpoint every ``period`` epochs.
        max_to_keep (int): Maximum number of most current checkpoints to keep,
            previous checkpoints will be deleted. If None, save all checkpoints.
    """

    def __init__(self, period: int, max_to_keep: Optional[int] = None) -> None:
        self._period = period
        assert max_to_keep is None or max_to_keep > 0
        self._max_to_keep = max_to_keep

        self._recent_checkpoints: List[str] = []

    def after_iter(self) -> None:
        if self.trainer.train_by_epoch:
            return
        if self.every_n_iters(self._period) or self.is_last_iter():
            iter = self.trainer.cur_iter  # ranged in [0, max_iters - 1]
            checkpoint_name = f"iter_{iter}.pth"
            self.trainer.save_checkpoint(checkpoint_name)
            self._delete_oldest_checkpoint(checkpoint_name)

    def after_epoch(self) -> None:
        if not self.trainer.train_by_epoch:
            return
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            epoch = self.trainer.cur_epoch  # ranged in [0, max_epochs - 1]
            checkpoint_name = f"epoch_{epoch}.pth"
            self.trainer.save_checkpoint(checkpoint_name)
            self._delete_oldest_checkpoint(checkpoint_name)

    def _delete_oldest_checkpoint(self, checkpoint_name):
        if self._max_to_keep is not None:
            self._recent_checkpoints.append(checkpoint_name)
            if len(self._recent_checkpoints) > self._max_to_keep:
                file_name = self._recent_checkpoints.pop(0)
                file_path = osp.join(self.trainer.ckpt_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "trainer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
