from typing import Callable

from .hookbase import HookBase


class EvalHook(HookBase):
    """Run an evaluation function periodically.

    It is executed every ``period`` epochs and after the last epoch.

    Args:
        period (int): The period to run ``eval_func``. Set to 0 to
            not evaluate periodically, but still after the last epoch.
        eval_func (callable): A function which takes no arguments, and
            returns a dict of evaluation metrics.
    """

    priority = 1

    def __init__(self, period: int, eval_func: Callable) -> None:
        self._period = period
        self._eval_func = eval_func

    def _do_eval(self) -> None:
        res = self._eval_func()

        if res:
            assert isinstance(res, dict), f"Eval function must return a dict. Got {res} instead."
            for k, v in res.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        f"Eval function should return a dict of float. Got '{k}: {v}' instead."
                    ) from e
            self.log(self.trainer.epoch, **res, smooth=False)

    def after_epoch(self) -> None:
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._do_eval()
