import logging
import os
import time
import weakref
from typing import Dict, List, Optional

import numpy as np
import torch

from .checkpoint import Checkpointer
from .hooks import HookBase
from .lr_scheduler import LRWarmupScheduler
from .metric_storage import MetricStorage

logger = logging.getLogger(__name__)


class Trainer:
    """An epoch-based trainer.

    .. Note::

        Currently only support single GPU training.

    Attributes:
        model (torch.nn.Module)
        optimizer (torch.optim.Optimizer)
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler)
        data_loader (torch.utils.data.DataLoader): Training data loader.
        work_dir (str): The working directory to save checkpoints and logs. Defaults to "work_dir".
        iter (int): The current iteration.
        epoch (int): The current epoch.
        start_iter (int): The iteration to start with. The minimum possible value is 0.
        start_epoch (int): The epoch to start with. The minimum possible value is 0.
        max_epochs (int): Total training epochs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        data_loader: torch.utils.data.DataLoader,
        max_epochs: int,
        work_dir: str = "work_dir",
        warmup_method: Optional[str] = None,
        warmup_iters: int = 1000,
        warmup_factor: float = 0.001,
    ):
        self.model = model
        self.optimizer = optimizer
        # convert epoch-based scheduler to iteration-based scheduler
        self.lr_scheduler = LRWarmupScheduler(
            lr_scheduler, len(data_loader), warmup_method, warmup_iters, warmup_factor
        )
        self.data_loader = data_loader
        self.work_dir = work_dir
        self.iter = 0
        self.epoch = 0
        self.start_iter = 0
        self.start_epoch = 0
        self.max_epochs = max_epochs
        self.max_iters = max_epochs * len(data_loader)
        self.metric_storage = MetricStorage()
        self.checkpointer = Checkpointer(
            os.path.join(self.trainer.work_dir, "checkpoints"),
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            **self._get_checkpointable_hooks(),
        )

        self._hooks: List[HookBase] = []
        self._data_iter = iter(data_loader)

    def _get_checkpointable_hooks(self) -> Dict[str, HookBase]:
        checkpointable_hooks = {}
        for hook in self._hooks:
            if hook.checkpointable:
                class_name = hook.__class__.__name__
                checkpointable_hooks[class_name] = hook
        return checkpointable_hooks

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """Register hooks to the trainer.

        The hooks are executed in the order they are registered.

        Args:
            hooks (List[Optional[HookBase]]): List of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def _before_train(self) -> None:
        for h in self._hooks:
            h.before_train()

    def _after_train(self) -> None:
        for h in self._hooks:
            h.after_train()

    def _before_epoch(self) -> None:
        for h in self._hooks:
            h.before_epoch()

    def _after_epoch(self) -> None:
        for h in self._hooks:
            h.after_epoch()

    def _before_iter(self) -> None:
        for h in self._hooks:
            h.before_iter()

    def _after_iter(self) -> None:
        for h in self._hooks:
            h.after_iter()

    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _train_one_iter(self) -> None:
        start = time.perf_counter()
        data = next(self._data_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        if not np.isfinite(losses):
            raise FloatingPointError(
                f"Loss became infinite or NaN at epoch={self.epoch}! loss_dict = {loss_dict}."
            )

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        # iteration-based scheduler need to be called after every iteration
        self.lr_scheduler.step()

        self.metric_storage.update(
            self.iter, total_loss=losses, data_time=data_time, learning_rate=self.learning_rate
        )

    def _train_one_epoch(self) -> None:
        self.model.train()
        for _ in range(len(self.data_loader)):
            self._before_iter()
            self._train_one_iter()
            self._after_iter()
            self.iter += 1
        # update data iter to prevent StopIteration exception
        self._data_iter = next(self.data_loader)

    def fit(self) -> None:
        """Start training."""
        self._before_train()
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            self._before_epoch()
            self.train_one_epoch()
            self._after_epoch()
        self._after_train()

    def resume(self, path: str, which_to_load: Optional[List[str]] = None) -> None:
        """Resume training from given checkpoint.

        Args:
            path (str): Path to the checkpoint.
            which_to_load (Optional[List[str]], optional): List of checkpointable names to load.
                Defaults to None.
        """
        extra_data = self.checkpointer.load(path, which_to_load)
        self.start_epoch = extra_data["epoch"] + 1
