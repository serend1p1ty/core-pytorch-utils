from typing import Optional

from torch.optim.lr_scheduler import _LRScheduler


class LRWarmupScheduler(_LRScheduler):
    """LR scheduler supporting warmup.

    This class wraps the standard PyTorch LR scheduler to supporting warmup.
    The scheduler should be called after every iteration demonstrated in the following snippet:

    .. code-block:: python
        :emphasize-lines: 15-18

        lr_scheduler = LRWarmupScheduler(
            StepLR(optimizer, step_size=10, gamma=0.1),
            epoch_len=9999,  # 9999 iterations per epoch
            warmup_method="linear",
            warmup_iters=1000
            warmup_factor=0.001
        )
        for epoch in range(start_epoch, max_epochs):
            for batch in data_loader:
                output = model(batch)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # should be here
                lr_scheduler.step()
            # not here
            # lr_scheduler.step()

    .. Note::

        PyTorch's LR scheduler assumes the LR changes only on epoch boundaries. However, we use
        iteration based schedules. As a result, "epoch" (e.g., self.last_epoch)
        should be understood to mean "iteration" instead.

    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler): Standard epoch-based LR scheduler.
        epoch_len (int): Length of one epoch, used for calling epoch-based
            LR scheduler at epoch end.
        warmup_method (str): Type of warmup used. It can be None (no warmup),
            "constant", "linear" or "exp". Defaults to None.
        warmup_iters (int): The number of iterations that warmup lasts. Defaults to 1000.
        warmup_factor (float): LR used at the beginning of warmup equals to
            ``warmup_factor * initial_lr``. Defaults to 0.001.
        last_epoch (int): Create a member variable ``self.last_epoch`` to represent
            the last iteration. Defaults to -1.
    """

    def __init__(
        self,
        scheduler: _LRScheduler,
        epoch_len: int,
        warmup_method: Optional[str] = None,
        warmup_iters: int = 1000,
        warmup_factor: float = 0.001,
        last_epoch: int = -1,
    ):
        self.scheduler = scheduler
        self.epoch_len = epoch_len
        self.warmup_method = warmup_method
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor

        if self._enable_warmup():
            assert warmup_method in ["constant", "linear", "exp"], (
                f"'{warmup_method}' is not a supported type for warmup, "
                "valid types are 'constant', 'linear' or 'exp'"
            )
            assert callable(
                getattr(scheduler, "_get_closed_form_lr", None)
            ), "`scheduler` must implement `_get_closed_form_lr()` method"
            assert warmup_iters > 0, "'warmup_iters' must be a positive integer"
            assert 0 < warmup_factor <= 1.0, "'warmup_ratio' must be in range (0,1]"

        # expected lr if no warming up is performed
        self.regular_lrs = scheduler.get_last_lr()

        super().__init__(scheduler.optimizer, last_epoch)

    def _enable_warmup(self) -> bool:
        return self.warmup_method is not None

    def _reach_epoch_end(self) -> bool:
        return self.last_epoch and self.last_epoch % self.epoch_len == 0

    def _get_warmup_factor(self) -> float:
        # `self.last_epoch` should be understood as `self.last_iter`
        if not self._enable_warmup() or self.last_epoch >= self.warmup_iters:
            return 1.0

        alpha = self.last_epoch / self.warmup_iters
        if self.warmup_method == "constant":
            return self.warmup_factor
        elif self.warmup_method == "linear":
            return self.warmup_factor * (1 - alpha) + alpha
        else:
            return self.warmup_factor ** (1 - alpha)

    def get_lr(self) -> float:
        warmup_factor = self._get_warmup_factor()
        if self._reach_epoch_end():
            # `self.scheduler.last_epoch` is really the last epoch
            self.scheduler.last_epoch += 1
            self.regular_lrs = self.scheduler._get_closed_form_lr()
        return [warmup_factor * lr for lr in self.regular_lrs]

    def state_dict(self):
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key != "optimizer" and key != "scheduler"
        }
        state["scheduler_state_dict"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict.pop("scheduler_state_dict"))
        self.__dict__.update(state_dict)
