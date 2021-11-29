from typing import Callable

from .hookbase import HookBase


class EvalHook(HookBase):
    """Run an evaluation function periodically.

    It is executed every ``period`` epochs and after the last epoch.
    """

    def __init__(self, period: int, eval_func: Callable):
        """
        Args:
            period (int): The period to run ``eval_func``. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_func (callable): A function which takes no arguments, and
                returns a dict of evaluation metrics.
        """
        self._period = period
        self._eval_func = eval_func

    def _do_eval(self):
        res = self._eval_func()

        if res:
            assert isinstance(res, dict), f"Eval function must return a dict. Got {res} instead."
            for k, v in res.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a dict of float. "
                        f"Got '{k}: {v}' instead."
                    ) from e
            self.storage.update(self.trainer.epoch, **res, smooth=False)

    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._do_eval()
