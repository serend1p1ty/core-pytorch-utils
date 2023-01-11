from typing import Any, Dict, List, Optional, Union

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class LRWarmupScheduler:
    """This class wraps the standard PyTorch LR scheduler to support warmup.

    The usage is demonstrated in the following snippet:

    .. code-block:: python
        :emphasize-lines: 6-9

        torch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
        warmup_scheduler = LRWarmupScheduler(torch_scheduler)
        for epoch in range(max_epochs):
            for iter in range(epoch_len):
                train_one_iter()
                # call iter_update() after each iteration
                warmup_scheduler.iter_update()
            # call epoch_update() after each epoch
            warmup_scheduler.epoch_update()

    Args:
        torch_scheduler (_LRScheduler)
        by_epoch (bool): If True, the ``torch_scheduler`` is epoch-based, else iteration-based.
            Defaults to True. # 这里指的是torch scheduler的性质，下面的warmup_by_epoch指的是warm up的性质，不要弄混。
        epoch_len (int): The number of iterations in an epoch. 每个epoch的step数量
            Required only when ``by_epoch=True & warmup_by_epoch=False``.
        warmup_t (int): How many iterations / epochs in warmup stage. If ``warmup_by_epoch=True``,
            "**t**" means epoch, else iteration. Defaults to 0 to disable warmup.
        warmup_by_epoch (bool): If True, perform warmup at each epoch end, else iteration end.
            Defaults to False.
        warmup_mode (str): "fix", "auto", or "factor". Defaults to "fix".
        warmup_init_lr (float): The initial warmup lr. Required in "fix" mode. Defaults to None.
        warmup_factor (float): The factor of initial warmup lr relative to base lr.
            Required in "auto" and "factor" mode. Defaults to None.
    """

    def __init__(
        self,
        torch_scheduler: _LRScheduler,
        by_epoch: bool = True,
        epoch_len: Optional[int] = None,
        # the following settings are related to warmup
        warmup_t: int = 0,
        warmup_by_epoch: bool = False,
        warmup_mode: str = "fix",
        warmup_init_lr: Optional[float] = None,
        warmup_factor: Optional[float] = None,
    ):
        self.torch_scheduler = torch_scheduler
        self.by_epoch = by_epoch
        self.epoch_len = epoch_len
        self.warmup_t = warmup_t
        self.warmup_by_epoch = warmup_by_epoch
        self.warmup_mode = warmup_mode
        self.warmup_init_lr = warmup_init_lr
        self.warmup_factor = warmup_factor

        if warmup_by_epoch: # 如果要按照epoch进行warmup，则要求schduler必须是 by_epoch
            assert by_epoch
        if by_epoch and warmup_t and not warmup_by_epoch:
            assert epoch_len is not None
        if self._is_plateau: # 如果是帕累托scheduler，则只能是by epoch的scheduler
            assert by_epoch

        self.param_groups = self.torch_scheduler.optimizer.param_groups
        #初始设置的学习率
        self.base_lrs = [param_group["lr"] for param_group in self.param_groups]

        if warmup_t:
            # pre-compute the regular lr if no warmup is performed
            # 预先计算好，没有warmup时，在warmup_t各个点上的学习率
            max_t = warmup_t // epoch_len if by_epoch and not warmup_by_epoch else warmup_t
            self.regular_lrs_per_t = self._pre_compute_regular_lrs_per_t(max_t)

        self.last_iter = self.last_epoch = 0
        # 用于标记是否在warmup的过程中
        self.in_iter_warmup = False

        if warmup_t > 0:
            if warmup_mode == "fix":
                # fix模式下，用设定好的warmup_init_lr，设置各个paramgroup的学习率。
                assert isinstance(warmup_init_lr, float)
                self._set_lrs(warmup_init_lr)
            elif warmup_mode == "factor":
                # factor模式下，将初始学习率设置为base_lr的
                assert isinstance(warmup_factor, float)
                self._set_lrs([base_lr * warmup_factor for base_lr in self.base_lrs])
            elif warmup_mode == "auto":
                assert isinstance(warmup_factor, float)
                # auto模式和factor模式比较像，只是在warmup最后一步时，将学习率强制设为正常schduler计算的值，
                # 也就是这里保存的warmup_end_lrs
                self.warmup_end_lrs = self.regular_lrs_per_t[-1]
                self._set_lrs([base_lr * warmup_factor for base_lr in self.base_lrs])
            else:
                raise ValueError(f"Invalid warmup mode: {warmup_mode}")

    @property
    def _is_plateau(self) -> bool:
        """
        帕累托优化是依据某个metric来决定是否调整学习率的。
        Reduce learning rate when a metric has stopped improving. 
        Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
        This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, 
        the learning rate is reduced."""
        return isinstance(self.torch_scheduler, ReduceLROnPlateau)

    def _pre_compute_regular_lrs_per_t(self, max_t: int) -> List[List[float]]:
        '''
        计算没有warmup的情况下，正常的lr scheduler学习率的变化情况。
        '''
        regular_lrs_per_t = [self.base_lrs]
        if self._is_plateau:
            return regular_lrs_per_t * (max_t + 1)
        for _ in range(max_t):
            self.torch_scheduler.step()
            regular_lrs_per_t.append([param_group["lr"] for param_group in self.param_groups])
        return regular_lrs_per_t

    def _get_warmup_lrs(self, t: int, regular_lrs: List[float]) -> List[float]:
        '''
        计算warm up的学习率，就是在正常应该的学习率上做调整。
        '''
        alpha = t / self.warmup_t
        if self.warmup_mode == "fix":
            # fix模式下，warmup达到的目的是warmup_init_lr --> base_lr
            return [
                self.warmup_init_lr * (1 - alpha) + base_lr * alpha for base_lr in self.base_lrs
            ]
        elif self.warmup_mode == "factor":
            # factor = self.warmup_factor + alpha*(1-self.warmup_factor)
            # 上面的公式更能清晰看出factor的变化规律，warmup_factor --> 1
            # factor模式下，每个warmup step的作用是，将正常应该的学习率乘以factor。
            # 最后一个step factor为1
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * factor for lr in regular_lrs]
        else:
            # auto 模式下，只考虑base_lrs和warmup结束时应当达到的学习率，不考虑regular_lrs
            # 其变化规律是 base_lr --> warmup_end_lrs
            return [
                base_lr * self.warmup_factor * (1 - alpha) + end_lr * alpha
                for base_lr, end_lr in zip(self.base_lrs, self.warmup_end_lrs)
            ]

    def _set_lrs(self, lrs: Union[float, List[float]]) -> None:
        '''
        设置计算出来的学习率
        '''
        if not isinstance(lrs, (list, tuple)):
            lrs = [lrs] * len(self.param_groups)
        for param_group, lr in zip(self.param_groups, lrs):
            param_group["lr"] = lr

    def epoch_update(self, metric: Optional[float] = None) -> None:
        """Prepare the learning rate for the next epoch.
        The method should be called after finishing each epoch.

        Args:
            metric (float): Metric value used in :class:`ReduceLROnPlateau`. Defaults to None.
        """
        if not self.by_epoch:
            return

        self.last_epoch += 1
        if self.warmup_by_epoch and self.last_epoch < self.warmup_t:
            # 在warmup阶段，计算warmup的学习率，并设置
            self._set_lrs(
                self._get_warmup_lrs(self.last_epoch, self.regular_lrs_per_t[self.last_epoch]))
        elif self.warmup_by_epoch and self.last_epoch == self.warmup_t:
            # warmup最后一步，设置成正常的学习率
            self._set_lrs(self.regular_lrs_per_t[-1])
        elif not self.in_iter_warmup:
            # 已经过了warmup阶段，则按照scheduler更新学习率。
            if self._is_plateau:
                self.torch_scheduler.step(metric)
            else:
                self.torch_scheduler.step()

    def iter_update(self) -> None:
        """Prepare the learning rate for the next iteration.
        The method should be called after finishing each iteration.
        """
        if self.warmup_by_epoch:
            return

        self.last_iter += 1
        if self.last_iter < self.warmup_t:
            self.in_iter_warmup = True
            t = self.last_iter // self.epoch_len if self.by_epoch else self.last_iter
            self._set_lrs(self._get_warmup_lrs(self.last_iter, self.regular_lrs_per_t[t]))
        elif self.last_iter == self.warmup_t:
            self._set_lrs(self.regular_lrs_per_t[-1])
        else:
            #超出warmup阶段后，根据原有的lr scheduler进行学习率的更新。
            self.in_iter_warmup = False
            if not self.by_epoch:
                self.torch_scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the scheduler as a dict."""
        state = {key: value for key, value in self.__dict__.items() if key != "torch_scheduler"}
        state["torch_scheduler"] = self.torch_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the scheduler state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.torch_scheduler.load_state_dict(state_dict.pop("torch_scheduler"))
        self.__dict__.update(state_dict)
