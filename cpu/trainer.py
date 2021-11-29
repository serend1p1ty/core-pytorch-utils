import logging
import os
import os.path as osp
import time
import weakref
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .hooks import CheckpointerHook, HookBase, TensorboardWriterHook, TerminalWriterHook
from .lr_scheduler import LRWarmupScheduler
from .metric_storage import MetricStorage
from .misc import symlink

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
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        data_loader: DataLoader,
        max_epochs: int,
        work_dir: str = "work_dir",
        max_num_checkpoints: int = None,
        checkpoint_period: int = 1,
        log_period: int = 50,
        clip_grad_norm: float = 0.0,
        enable_amp=False,
        warmup_method: Optional[str] = None,
        warmup_iters: int = 1000,
        warmup_factor: float = 0.001,
    ):
        """
        Args:
            model (torch.nn.Module)
            optimizer (torch.optim.Optimizer)
            lr_scheduler (optim.lr_scheduler._LRScheduler)
            data_loader (torch.utils.data.DataLoader): Training data loader.
            max_epochs (int): Total training epochs.
            work_dir (str): The working directory to save checkpoints and logs.
                Defaults to "work_dir".
            max_num_checkpoints (int): The maximum number of checkpoints to save.
                If None, save all checkpoints. Defaults to None.
            checkpoint_period (int): The period (epoch-based) to save checkpoint. Defaults to 1.
            log_period (int): The period (iter-based) to log. Defaults to 50.
            clip_grad_norm (float): [description]. Defaults to 0.0.
            enable_amp (bool): Enable the Automatic Mixed Precision (AMP) training.
                Defaults to False.
            warmup_method (str): Type of warmup used. It can be None (no warmup),
                "constant", "linear" or "exp". Defaults to None.
            warmup_iters (int): The number of iterations that warmup lasts. Defaults to 1000.
            warmup_factor (float): LR used at the beginning of warmup equals to
                ``warmup_factor * initial_lr``. Defaults to 0.001.
        """
        self.model = model
        self.optimizer = optimizer
        # convert epoch-based scheduler to iteration-based scheduler
        self.lr_scheduler = LRWarmupScheduler(
            lr_scheduler, len(data_loader), warmup_method, warmup_iters, warmup_factor
        )
        self.data_loader = data_loader
        self.work_dir = work_dir
        self.metric_storage = MetricStorage()

        # counters
        self.inner_iter: int  # [0, epoch_len - 1]
        self.epoch: int  # [0, max_epochs - 1]
        self.start_epoch = 0  # [0, max_epochs - 1]
        self.max_epochs = max_epochs

        self._hooks: List[HookBase] = []
        self._data_iter = iter(data_loader)
        self._max_num_checkpoints = max_num_checkpoints
        self._checkpoint_period = checkpoint_period
        self._log_period = log_period
        self._clip_grad_norm = clip_grad_norm
        self._enable_amp = enable_amp

        self._default_setup()

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @property
    def epoch_len(self) -> int:
        return len(self.data_loader)

    @property
    def max_iters(self) -> int:
        return self.max_epochs * self.epoch_len

    @property
    def iter(self) -> int:
        """Returns the current iteration ranged in [0, max_iters - 1]."""
        return self.epoch * self.epoch_len + self.inner_iter

    @property
    def start_iter(self) -> int:
        """The iteration to start from. The minimum possible value is 0."""
        return self.start_epoch * self.epoch_len

    @property
    def ckpt_dir(self) -> str:
        return osp.join(self.work_dir, "checkpoints")

    @property
    def tb_dir(self) -> str:
        return osp.join(self.work_dir, "tb_logs")

    @property
    def model_or_module(self) -> nn.Module:
        if isinstance(self.model, (DistributedDataParallel, DataParallel)):
            return self.model.module
        return self.model

    @property
    def registered_hook_names(self) -> List[str]:
        """The names of all registered hooks."""
        return [h.__class__.__name__ for h in self._hooks]

    def _default_setup(self):
        if self._enable_amp:
            logger.info("Automatic Mixed Precision (AMP) training is on.")
            self._scaler = GradScaler()

        self.register_hooks(self._build_default_hooks())
        logger.info(f"Registered default hooks: {self.registered_hook_names}")

        logger.info(
            f"Work directory: '{self.work_dir}'. "
            f"Checkpoint directory: '{self.ckpt_dir}'. "
            f"Tensorboard directory: '{self.tb_dir}'. "
        )

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
            # keep `TensorboardWriterHook` is the last hook
            if self._hooks and isinstance(self._hooks[-1], TensorboardWriterHook):
                self._hooks.insert(len(self._hooks) - 1, h)
            else:
                self._hooks.append(h)

    def _call_hooks(self, stage: str) -> None:
        for h in self._hooks:
            getattr(h, stage)()

    def _build_default_hooks(self) -> List[HookBase]:
        return [
            CheckpointerHook(self._checkpoint_period, self._max_num_checkpoints),
            TerminalWriterHook(self._log_period),
            TensorboardWriterHook(self._log_period, log_dir=self.tb_dir),
        ]

    def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float) -> None:
        """
        Args:
            loss_dict (dict): Dict of scalar losses.
            data_time (float): Time taken by the dataloader iteration.
        """
        self.metric_storage.update(self.iter, data_time=data_time)
        self.metric_storage.update(self.iter, lr=self.lr, smooth=False)

        loss_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        loss_value = sum(loss_dict.values())
        if not np.isfinite(loss_value):
            raise FloatingPointError(
                f"Loss became infinite or NaN at epoch={self.epoch}! loss_dict = {loss_dict}."
            )

        self.metric_storage.update(self.iter, total_loss=loss_value)
        if len(loss_dict) > 1:
            self.metric_storage.update(self.iter, **loss_dict)

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
        if self._enable_amp:
            with autocast():
                loss_dict = self.model(batch)
        else:
            loss_dict = self.model(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        self._write_metrics(loss_dict, data_time)

        ##########################
        # 3. Calculate gradients #
        ##########################
        self.optimizer.zero_grad()
        if self._enable_amp:
            self._scaler.scale(losses).backward()
        else:
            losses.backward()
        if self._clip_grad_norm > 0:
            if self._enable_amp:
                self._scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self._clip_grad_norm)

        ##############################
        # 4. Update model parameters #
        ##############################
        if self._enable_amp:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        ###########################
        # 5. Adjust learning rate #
        ###########################
        self.lr_scheduler.step()

    def _train_one_epoch(self) -> None:
        # evaluation hook changes the model to `eval` mode after finishing epoch
        self.model.train()
        for self.inner_iter in range(self.epoch_len):
            self._call_hooks("before_iter")
            self.train_one_iter()
            self._call_hooks("after_iter")
        # update data iterator to avoid `StopIteration` exception
        self._data_iter = iter(self.data_loader)

    def train(self) -> None:
        """Start training."""
        logger.info(f"Start training from epoch {self.start_epoch}")
        self._call_hooks("before_train")
        for self.epoch in range(self.start_epoch, self.max_epochs):
            self._call_hooks("before_epoch")
            self._train_one_epoch()
            self._call_hooks("after_epoch")
        self._call_hooks("after_train")

    def save_checkpoint(self, file_name: str) -> None:
        """Dump checkpointables to a file.

        Args:
            filename (str): The name of the file to save.
        """
        data = {
            "epoch": self.epoch,
            "model": self.model_or_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "metric_storage": self.metric_storage.state_dict(),
        }
        hooks_state = {h.class_name: h.state_dict() for h in self._hooks if h.checkpointable}
        if hooks_state:
            data["hooks"] = hooks_state

        os.makedirs(self.ckpt_dir, exist_ok=True)
        file_path = osp.join(self.ckpt_dir, file_name)
        logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

        # tag last checkpoint
        dst_file = osp.join(self.ckpt_dir, "latest.pth")
        symlink(file_name, dst_file)

    def load_checkpoint(
        self,
        path: str = "",
        checkpoint: Dict[str, Any] = None,
        which_to_load: Optional[List[str]] = None,
    ):
        """Load the given checkpoint.

        Args:
            checkpoint (dict): The checkpoint to load.
            path (str): Path to the checkpoint. If empty, will not load anything.
                `checkpoint` and `path` can only be specified one.
            which_to_load (list[str]): List of checkpointable names to load.
                If None, will load all possible checkpointables. Defaults to None.
        """
        assert checkpoint or path, "Either `checkpoint` or `path` must be specified."
        assert not (checkpoint and path), "`checkpoint` and `path` can only be specified one."

        if path:
            logger.info(f"Loading checkpoint from {path} ...")
            checkpoint = torch.load(path, map_location="cpu")

        self.start_epoch = checkpoint["epoch"] + 1

        if which_to_load is None or "optimizer" in which_to_load:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if which_to_load is None or "lr_scheduler" in which_to_load:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if which_to_load is None or "metric_storage" in which_to_load:
            self.metric_storage.load_state_dict(checkpoint["metric_storage"])

        incompatible = self.model_or_module.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.missing_keys:
            logger.warning(
                f"Encounter missing keys when loading model weights:\n{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "Encounter unexpected keys when loading model weights:\n"
                f"{incompatible.unexpected_keys}"
            )

        hook_state = checkpoint.get("hooks", {})
        hook_names = [h.class_name for h in self._hooks if h.checkpointable]
        missing_keys = [name for name in hook_names if name not in hook_state]
        unexpected_keys = [key for key in hook_state if key not in hook_names]
        if missing_keys:
            logger.warning(f"Encounter missing keys when loading hook state dict:\n{missing_keys}")
        if unexpected_keys:
            logger.warning(
                f"Encounter unexpected keys when loading hook state dict:\n{unexpected_keys}"
            )

        for key, value in hook_state.items():
            for h in self._hooks:
                if h.class_name == key and h.checkpointable:
                    h.load_state_dict(value)
                    break
