import datetime
import logging

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

    def after_step(self) -> None:
        if self.every_n_iters(self._period) or self.is_last_iter():
            for key, smoothed_value in self.trainer.metric_storage:
                iter = smoothed_value.latest[1]
                if "loss" in key or "accuracy" in key:
                    self._writer.add_scalar(key, smoothed_value.median, iter)
                else:
                    self._writer.add_scalar(key, smoothed_value.latest[0], iter)

    def after_train(self) -> None:
        self._writer.close()


class TerminalWriterHook(HookBase):
    """Write metrics to the terminal."""

    def __init__(self, period: int = 20):
        """
        Args:
            period (int): The period to write metrics.
        """
        self._period = period
        self._max_iter = self.trainer.max_iters
        self._storage = self.trainer.metric_storage

    def _get_eta(self) -> str:
        avg_iter_time = self.storage.global_avg.get("iteration_time", None)
        # If not register `TimerHook`, we don't know iteration time
        if not avg_iter_time:
            return ""
        eta_seconds = avg_iter_time * (self._max_iter - self.trainer.iter - 1)
        return str(datetime.timedelta(seconds=int(eta_seconds)))

    def after_step(self) -> None:
        if self.every_n_iters(self._period) or self.is_last_iter():
            data_time = self.storage.avg["data_time"]
            iter_time = self.storage.global_avg["iteration_time"]
            lr = f"{self.storage.latest['learning_rate']:.5g}"
            eta = self._get_eta(self.storage)

            if torch.cuda.is_available():
                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            else:
                max_mem_mb = None

            loss_list = [f"{k}: {v:.4g}" for k, v in self.storage.median.items() if "loss" in k]

            logger.info(
                "Epoch: [{epoch}][{iter}/{epoch_len}]"
                "{eta}{losses}  {iter_time}{data_time}lr: {lr}  {memory}".format(
                    epoch=self.trainer.epoch,
                    iter=self.trainer.iter,
                    epoch_len=len(self.trainer.data_loader),
                    eta=f"ETA: {eta}  " if eta else "",
                    losses="  ".join(loss_list),
                    iter_time=f"iter_time: {iter_time:.4f}  " if iter_time is not None else "",
                    data_time=f"data_time: {data_time:.4f}  " if data_time is not None else "",
                    lr=lr,
                    memory=f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else "",
                )
            )
