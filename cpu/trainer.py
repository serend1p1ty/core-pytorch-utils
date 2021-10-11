import logging
import os
import time
import weakref
from typing import Dict, List, Optional

import numpy as np
import torch

from .checkpoint import Checkpointer
from .hooks import CheckpointerHook, HookBase, TensorboardWriterHook, TerminalWriterHook, TimerHook
from .logger import setup_logger
from .lr_scheduler import LRWarmupScheduler
from .metric_storage import MetricStorage

logger = logging.getLogger(__name__)


class Trainer:
    """An epoch-based trainer.

    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source epoch-based optimization
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.
    4. Adjust the learning rate.

    All other tasks during training (checkpointing, logging, evaluation) are maintained
    by hooks, which can be registered by :meth:`register_hooks`.

    If you want to do anything fancier than this, either subclass this class
    and implement your own :meth:`train_one_iter`, or write your own trainer.

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
        max_num_checkpoints: int = None,
        checkpoint_period: int = 1,
        log_period: int = 50,
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
        self.epoch_len = len(data_loader)
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
        self._max_num_checkpoints = max_num_checkpoints
        self._checkpoint_period = checkpoint_period
        self._log_period = log_period

        self.register_hooks(self._build_default_hooks())
        # setup the root logger of the `cpu` library to show
        # the log messages generated from this library
        setup_logger("cpu")
        logger.info(f"Registered default hooks: {self.get_registered_hook_names()}")

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """Register hooks to the trainer.

        The hooks are executed in the order they are registered.

        Args:
            hooks (List[Optional[HookBase]]): List of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other. This normally
            # does not matter, but will cause memory leak if the involved objects contain __del__.
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def get_registered_hook_names(self) -> List[str]:
        """Return the names of all registered hooks."""
        return [h.__class__.__name__ for h in self._hooks]

    def _call_hooks(self, stage: str) -> None:
        for h in self._hooks:
            getattr(h, stage)()

    def _get_checkpointable_hooks(self) -> Dict[str, HookBase]:
        return {hook.__class__.__name__: hook for hook in self._hooks if hook.checkpointable}

    def _build_default_hooks(self) -> List[HookBase]:
        return [
            TimerHook(),
            CheckpointerHook(self.checkpointer, self._checkpoint_period, self._max_num_checkpoints),
            TerminalWriterHook(self._log_period),
            TensorboardWriterHook(self._log_period),
        ]

    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def train_one_iter(self) -> None:
        """Train one iteration.

        It does the following things:

        1. Load batch data.
        2. Forward batch and calculate loss.
        3. Backward loss to calculate gradients.
        4. Update model parameters by optimizer.
        5. Adjust the learning rate of the optimizer.

        .. Note::

            Standard PyTorch LR scheduler is epoch-based and called at the end of epoch.
            However, our scheduler is iteration-based, so it should be called after every iteration.

        Subclass :class:`cpu.Trainer` and implement your :meth:`train_one_iter`
        to do something fancier.
        """
        ######################
        # 1. Load batch data #
        ######################
        # we choose to read data by iterator instead of `for data in data_loader`
        # in order to calculate the data loading time
        start = time.perf_counter()
        batch = next(self._data_iter)
        data_time = time.perf_counter() - start

        #####################
        # 2. Calculate loss #
        #####################
        loss_dict = self.model(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        if not np.isfinite(losses):
            raise FloatingPointError(
                f"Loss became infinite or NaN at epoch={self.epoch}! loss_dict = {loss_dict}."
            )

        ##########################
        # 3. Calculate gradients #
        ##########################
        self.optimizer.zero_grad()
        losses.backward()

        ##############################
        # 4. Update model parameters #
        ##############################
        self.optimizer.step()

        ###########################
        # 5. Adjust learning rate #
        ###########################
        self.lr_scheduler.step()

        self.metric_storage.update(
            self.iter, total_loss=losses, data_time=data_time, lr=self.learning_rate
        )

    def _train_one_epoch(self) -> None:
        # evaluation hook changes the model to `eval` mode after finishing epoch
        self.model.train()
        for _ in range(self.epoch_len):
            self._call_hooks("before_iter")
            self.train_one_iter()
            self._call_hooks("after_iter")
            self.iter += 1
        # update data iterator to avoid `StopIteration` exception
        self._data_iter = next(self.data_loader)

    def train(self) -> None:
        """Start training."""
        logger.info(f"Start training from epoch {self.start_epoch}")
        self._call_hooks("before_train")
        for self.epoch in range(self.start_epoch, self.max_epochs):
            self._call_hooks("before_epoch")
            self._train_one_epoch()
            self._call_hooks("after_epoch")
        self._call_hooks("after_train")

    def resume(self, path: str, which_to_load: Optional[List[str]] = None) -> None:
        """Resume training from given checkpoint.

        Args:
            path (str): Path to the checkpoint.
            which_to_load (Optional[List[str]]): List of checkpointable names to load.
                If None, will load all possible checkpointables. Defaults to None.
        """
        extra_data = self.checkpointer.load(path, which_to_load)
        self.start_epoch = extra_data["epoch"] + 1
        self.start_iter = self.start_epoch * self.epoch_len
