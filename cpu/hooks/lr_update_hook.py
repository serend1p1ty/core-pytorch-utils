from typing import Optional
from .hookbase import HookBase


class LRUpdateHook(HookBase):
    def after_epoch(self, metric: Optional[float] = None) -> None:
        if self.trainer.lr_scheduler._is_plateau:
            self.trainer.lr_scheduler.epoch_update(metric)
        else:
            self.trainer.lr_scheduler.epoch_update()

    def after_iter(self) -> None:
        self.trainer.lr_scheduler.iter_update()
