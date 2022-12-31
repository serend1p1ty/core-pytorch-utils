from typing import Any, Dict, List, Optional, Union

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

class LRWarmupScheduler:
    """This class wraps the standard PyTorch LR scheduler to support warmup."""

    def __init__(self, torch_scheduler, by_epoch, epoch_len=None,
                 # 以下为warmup相关设置
                 warmup_t=0, warmup_by_epoch=False, warmup_mode="fix",
                 warmup_init_lr=None, warmup_factor=None):
        # PyTorch原生的lr scheduler
        self.torch_scheduler = torch_scheduler
        # 表示torch_scheduler是by epoch还是by iter
        self.by_epoch = by_epoch
        # 每个epoch的长度，只有by_epoch=True且warmup_by_epoch=False时才需要传入这个参数
        self.epoch_len = epoch_len
        # 后面有很多代码需要同时处理by epoch和by iter的情况，将变量命名为epoch或iter都不合适，
        # 所以选择用t来代表这层含义。当warmup_by_epoch=True时，warmup_t代表warmup_epochs，
        # 反之代表warmup_iters。
        self.warmup_t = warmup_t
        # 表示warmup是by epoch还是by iter
        self.warmup_by_epoch = warmup_by_epoch
        # 取值为fix、auto、factor
        self.warmup_mode = warmup_mode
        # fix模式的warmup初始学习率
        self.warmup_init_lr = warmup_init_lr
        # auto和factor模式下，warmup的初始学习率为base_lr * warmup_factor
        self.warmup_factor = warmup_factor

        self.param_groups = self.torch_scheduler.optimizer.param_groups
        self.base_lrs = [param_group["lr"] for param_group in self.param_groups]
        # 因为factor模式需要知道常规学习率才能推导出warmup学习率，所以需要预先计算出torch_scheduler在每个t的常规学习率。
        # 假设by_epoch=True & warmup_by_epoch=False & warmup_t=25 & epoch_len=10，
        # 说明warmup阶段跨越了3个epoch，我们需要预先计算出torch_scheduler在前三个epoch的
        # 常规学习率（保存在self.regular_lrs_per_t中）。
        # PS：虽然很多PyTorch原生的lr scheduler（StepLR、MultiStepLR、CosineAnnealingLR）
        # 提供了学习率的封闭形式，即_get_closed_form_lr()函数，可以通过传入的epoch参数直接计算出对应的学习率。
        # 但仍有些scheduler并未提供此功能，例如CosineAnnealingWarmRestarts。
        # 所以这里只能通过step()函数来一步步地计算出学习率。
        max_t = warmup_t // epoch_len if by_epoch and not warmup_by_epoch else warmup_t
        self.regular_lrs_per_t = self._pre_compute_regular_lrs_per_t(max_t)

        self.last_iter = self.last_epoch = 0
        self.in_iter_warmup = False

        if warmup_by_epoch:
            assert by_epoch
        if by_epoch and not warmup_by_epoch:
            assert epoch_len is not None
        if self._is_plateau:
            assert by_epoch
        if warmup_t > 0:
            if warmup_mode == "fix":
                assert isinstance(warmup_init_lr, float)
                # 为第0个t准备好学习率
                self._set_lrs(warmup_init_lr)
            elif warmup_mode == "factor":
                assert isinstance(warmup_factor, float)
                self._set_lrs([base_lr * warmup_factor for base_lr in self.base_lrs])
            elif warmup_mode == "auto":
                assert isinstance(warmup_factor, float)
                self.warmup_end_lrs = self.regular_lrs_per_t[-1]
                self._set_lrs([base_lr * warmup_factor for base_lr in self.base_lrs])
            else:
                raise ValueError(f"Invalid warmup mode: {warmup_mode}")

    @property
    def _is_plateau(self):
        return isinstance(self.torch_scheduler, ReduceLROnPlateau)

    def _pre_compute_regular_lrs_per_t(self, max_t):
        regular_lrs_per_t = [self.base_lrs]
        if self._is_plateau:
            return regular_lrs_per_t * (max_t + 1)
        for _ in range(max_t):
            self.torch_scheduler.step()
            regular_lrs_per_t.append([param_group["lr"] for param_group in self.param_groups])
        return regular_lrs_per_t

    def _get_warmup_lrs(self, t, regular_lrs):
        # 为了简单，不再计算斜率，而是通过线性插值的方式获得warmup lr
        alpha = t / self.warmup_t
        if self.warmup_mode == "fix":
            return [self.warmup_init_lr * (1 - alpha) + base_lr * alpha for base_lr in self.base_lrs]
        elif self.warmup_mode == "factor":
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * factor for lr in regular_lrs]
        else:
            return [
                base_lr * self.warmup_factor * (1 - alpha) + end_lr * alpha
                for base_lr, end_lr in zip(self.base_lrs, self.warmup_end_lrs)
            ]

    def _set_lrs(self, lrs):
        if not isinstance(lrs, (list, tuple)):
            lrs = [lrs] * len(self.param_groups)
        for param_group, lr in zip(self.param_groups, lrs):
            param_group['lr'] = lr

    def epoch_update(self, metric=None):
        if not self.by_epoch:
            return

        self.last_epoch += 1
        if self.warmup_by_epoch and self.last_epoch < self.warmup_t:
            # 0 <= t < warmup_t时，根据不同的策略设置相应的warmup学习率
            self._set_lrs(self._get_warmup_lrs(self.last_epoch, self.regular_lrs_per_t[self.last_epoch]))
        elif self.warmup_by_epoch and self.last_epoch == self.warmup_t:
            # t == warmup_t时，将学习率恢复为常规学习率
            self._set_lrs(self.regular_lrs_per_t[-1])
        # in_iter_warmup=True时代表正在进行by iter的warmup，
        # lr已经被设置好了，此时torch_scheduler不能再执行step()函数修改lr
        elif not self.in_iter_warmup:
            if self._is_plateau:
                self.torch_scheduler.step(metric)
            else:
                self.torch_scheduler.step()

    def iter_update(self):
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
            # warmup结束之后，将in_iter_warmup变量置为False，此时torch_scheduler才可以进行正常的step()
            self.in_iter_warmup = False
            if not self.by_epoch:
                self.torch_scheduler.step()

    def state_dict(self):
        state = {key: value for key, value in self.__dict__.items() if key != "torch_scheduler"}
        state["torch_scheduler"] = self.torch_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.torch_scheduler.load_state_dict(state_dict.pop("torch_scheduler"))
        self.__dict__.update(state_dict)