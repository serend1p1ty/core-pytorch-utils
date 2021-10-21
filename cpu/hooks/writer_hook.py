import datetime
import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from .hookbase import HookBase

logger = logging.getLogger(__name__)


class TensorboardWriterHook(HookBase):
    """Write all metrics to a tensorboard file."""

    def __init__(self, period: int = 20, log_dir: str = "log_dir", **kwargs) -> None:
        """
        Args:
            period (int): The period to write metrics.
            log_dir (str): The directory to save the output events
            kwargs: other arguments passed to ``torch.utils.tensorboard.SummaryWriter(...)``
        """
        self._period = period
        self._writer = SummaryWriter(log_dir, **kwargs)

    def write(self):
        for key, (iter, value) in self.storage.values_maybe_smooth.items():
            self._writer.add_scalar(key, value, iter)

    def after_iter(self) -> None:
        if self.every_n_inner_iters(self._period):
            self.write()

    def after_train(self) -> None:
        self._writer.close()


class TerminalWriterHook(HookBase):
    """Write all metrics to the terminal.

    The hook tracks the time spent of each iteration and the whole training process.
    It regards the time between :meth:`before_iter` and :meth:`after_iter` methods
    as iteration time. Under the convention that :meth:`before_iter` of all hooks should
    only take negligible amount of time, the :class:`TerminalWriterHook` hook should be placed
    at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, period: int = 20) -> None:
        """
        Args:
            period (int): The period to write metrics.
        """
        self._period = period
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

    def _get_eta(self) -> str:
        iter = self.trainer.iter
        max_iters = self.trainer.max_iters
        avg_iter_time = self.storage.global_avg["iter_time"][1]
        eta_seconds = avg_iter_time * (max_iters - iter - 1)
        return str(datetime.timedelta(seconds=int(eta_seconds)))

    def write(self) -> None:
        # "data_time" may does not exist when user overwrites `self.trainer.train_one_iter()`
        data_time = self.storage.values_maybe_smooth.get("data_time", None)
        iter_time = self.storage.global_avg["iter_time"]
        lr = self.storage.values_maybe_smooth["lr"]
        eta_string = self._get_eta()

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        loss_strings = [
            f"{key}: {value:.4g}"
            for key, (_, value) in self.storage.values_maybe_smooth.items()
            if "loss" in key
        ]

        process_string = (
            f"Epoch: [{self.trainer.epoch}][{self.trainer.inner_iter}/{self.trainer.epoch_len - 1}]"
        )

        logger.info(
            "{process}  {eta}  {losses}  {iter_time}  {data_time}{lr}  {memory}".format(
                process=process_string,
                eta=f"ETA: {eta_string}",
                losses="  ".join(loss_strings),
                iter_time=f"iter_time: {iter_time[1]:.4f}",
                data_time=f"data_time: {data_time[1]:.4f}  " if data_time is not None else "",
                lr=f"lr: {lr[1]:.5g}",
                memory=f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else "",
            )
        )

    def after_iter(self) -> None:
        iter_time = time.perf_counter() - self._iter_start_time
        self._total_iter_time += iter_time
        self.storage.update(self.trainer.iter, iter_time=iter_time)

        if self.every_n_inner_iters(self._period):
            self.write()
