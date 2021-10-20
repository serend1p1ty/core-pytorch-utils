import datetime
import logging
import time

from .hookbase import HookBase

logger = logging.getLogger(__name__)


class TimerHook(HookBase):
    """Track the time spent of each iteration and the whole training process.

    This hook regards the time between :meth:`before_iter` and :meth:`after_iter` methods
    as iteration time. Under the convention that :meth:`before_iter` of all hooks should
    only take negligible amount of time, the :class:`IterationTimer` hook should be placed
    at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self) -> None:
        self._train_start_time: float
        self._iter_start_time: float
        self._total_iter_time: float = 0.0

    def before_train(self):
        self._train_start_time = time.perf_counter()

    def after_train(self):
        total_train_time = time.perf_counter() - self._train_start_time
        total_hook_time = total_train_time - self._total_iter_time

        assert self.trainer.iter == self.trainer.max_iters - 1
        num_iter = self.trainer.iter - self.trainer.start_iter

        logger.info(
            "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                num_iter,
                str(datetime.timedelta(seconds=int(self._total_iter_time))),
                self._total_iter_time / num_iter,
            )
        )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_train_time))),
                str(datetime.timedelta(seconds=int(total_hook_time))),
            )
        )

    def before_iter(self):
        self._iter_start_time = time.perf_counter()

    def after_iter(self):
        iter_time = time.perf_counter() - self._iter_start_time
        self._total_iter_time += iter_time
        self.storage.update(self.trainer.iter, iter_time=iter_time)
