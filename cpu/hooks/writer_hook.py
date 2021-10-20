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
    """Write all metrics to the terminal."""

    def __init__(self, period: int = 20) -> None:
        """
        Args:
            period (int): The period to write metrics.
        """
        self._period = period
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA.

    def _get_eta(self) -> str:
        iter = self.trainer.iter
        max_iters = self.trainer.max_iters
        avg_iter_time = self.storage.global_avg.get("iter_time", None)

        if avg_iter_time is not None:
            # "iter_time" does not exist when user didn't register `TimerHook`
            eta_seconds = avg_iter_time[1] * (max_iters - iter - 1)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        else:
            # estimate iteration time on our own - more noisy
            eta_string = None
            if self._last_write is not None:
                estimated_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iter - self._last_write[0]
                )
                eta_seconds = estimated_iter_time * (max_iters - iter - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iter, time.perf_counter())
            return eta_string

    def write(self) -> None:
        data_time = self.storage.values_maybe_smooth["data_time"]
        iter_time = self.storage.global_avg.get("iter_time", None)
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
            "{process}  {eta}{losses}  {iter_time}{data_time}lr: {lr}  {memory}".format(
                process=process_string,
                eta=f"ETA: {eta_string}  " if eta_string else "",
                losses="  ".join(loss_strings),
                iter_time=f"iter_time: {iter_time[1]:.4f}  " if iter_time is not None else "",
                data_time=f"data_time: {data_time[1]:.4f}  " if data_time is not None else "",
                lr=f"{lr[1]:.5g}",
                memory=f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else "",
            )
        )

    def after_iter(self) -> None:
        if self.every_n_inner_iters(self._period):
            self.write()
