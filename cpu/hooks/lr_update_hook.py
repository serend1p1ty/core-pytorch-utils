from .eval_hook import EvalHook
from .hookbase import HookBase


class LRUpdateHook(HookBase):
    """Adjust learning rate after each epoch and iteration.

    To use :class:`ReduceLROnPlateau` scheduler, user should register
    an :class:`EvalHook` which returns a dict containing 'Eval Metric' field.
    The :class:`EvalHook` should be called after each epoch (i.e., set ``period=1``),
    and before the :class:`LRUpdateHook`.
    """

    # should > the priority of EvalHook (level 1)
    priority = 2

    def __init__(self):
        self.checked = False

    def _check_for_plateau(self):
        eval_hook = None
        for hook in self.trainer._hooks:
            if isinstance(hook, EvalHook):
                eval_hook = hook
                break
        assert eval_hook, "To use ReduceLROnPlateau scheduler, you should register an EvalHook."
        assert eval_hook.priority < self.priority, "EvalHook must be called before LRUpdateHook"
        assert eval_hook._period == 1, "EvalHook should be called after each epoch."
        assert "Eval Metric" in self.metric_storage, (
            "EvalHook should return a dict containing 'Eval Metric' field.")

    def after_epoch(self) -> None:
        if self.trainer.lr_scheduler._is_plateau:
            if not self.checked:
                self._check_for_plateau()
                self.checked = True
            eval_metric = self.metric_storage["Eval Metric"].latest
            self.trainer.lr_scheduler.epoch_update(eval_metric)
        else:
            self.trainer.lr_scheduler.epoch_update()

    def after_iter(self) -> None:
        self.trainer.lr_scheduler.iter_update()
