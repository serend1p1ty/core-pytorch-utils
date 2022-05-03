import mmcv
import numpy as np
import torch
import torch.nn as nn
from cpu.lr_scheduler import LRWarmupScheduler
from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    LinearParamScheduler,
    CosineParamScheduler,
    MultiStepParamScheduler
)
from mmcv.runner.hooks import HOOKS
from timm.scheduler import (
    CosineLRScheduler,
    MultiStepLRScheduler,
    PlateauLRScheduler,
    StepLRScheduler
)
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR
)


class WarmupParamScheduler(CompositeParamScheduler):
    def __init__(self, scheduler, warmup_factor, warmup_length, warmup_method="linear"):
        end_value = scheduler(warmup_length)  # the value to reach when warmup ends
        start_value = warmup_factor * scheduler(0.0)
        if warmup_method == "constant":
            warmup = ConstantParamScheduler(start_value)
        elif warmup_method == "linear":
            warmup = LinearParamScheduler(start_value, end_value)
        else:
            raise ValueError("Unknown warmup method: {}".format(warmup_method))
        super().__init__(
            [warmup, scheduler],
            interval_scaling=["rescaled", "fixed"],
            lengths=[warmup_length, 1 - warmup_length],
        )


class LRMultiplier(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, multiplier, max_iter, last_iter=-1):
        self._multiplier = multiplier
        self._max_iter = max_iter
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self):
        multiplier = self._multiplier(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]


def get_lrs_d2(max_epochs, epoch_len, opt_config, lr_config):
    optimizer = _get_optimizer_from_config(opt_config)

    max_iters = max_epochs * epoch_len
    if lr_config["type"] == "multistep":
        steps = lr_config["steps"]
        sched = MultiStepParamScheduler(
            values=[0.1 ** k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=max_iters,
        )
    elif lr_config["type"] == "cosine":
        sched = CosineParamScheduler(1, 0)

    sched = WarmupParamScheduler(sched, lr_config["warmup_factor"],
                                 min(lr_config["warmup_iters"] / max_iters, 1.0), "linear")
    lr_scheduler = LRMultiplier(optimizer, multiplier=sched, max_iter=max_iters)

    lrs = []
    for _ in range(max_epochs):
        for _ in range(epoch_len):
            lrs.append(_get_optimizer_lr(optimizer))
            optimizer.step()
            lr_scheduler.step()
    return lrs


class MMCVRunner:
    def __init__(self, max_epochs, epoch_len, opt_config, lr_config):
        self.max_epochs = max_epochs
        self.epoch_len = epoch_len
        self.optimizer = _get_optimizer_from_config(opt_config)
        self.hooks = []
        self.register_hook_from_cfg(lr_config)

    def run(self):
        lrs = []
        self.iter = 0
        self.call_hook('before_run')
        for self.epoch in range(self.max_epochs):
            self.call_hook('before_train_epoch')
            for self.inner_iter in range(self.epoch_len):
                self.call_hook('before_train_iter')
                lr = []
                for param_group in self.optimizer.param_groups:
                    lr.append(param_group["lr"])
                lrs.append(lr)
                self.optimizer.step()
                self.call_hook('after_train_iter')
                self.iter += 1
            self.call_hook('after_train_epoch')
        self.call_hook('after_run')
        return lrs

    def register_hook_from_cfg(self, hook_cfg):
        hook = mmcv.build_from_cfg(hook_cfg, HOOKS)
        self.hooks.append(hook)

    def call_hook(self, fn_name):
        for hook in self.hooks:
            getattr(hook, fn_name)(self)


def _get_optimizer_from_config(opt_config):
    params = []
    for i in range(opt_config["num_pram_groups"]):
        params.append({"params": nn.Parameter(torch.zeros(0)), "lr": opt_config["base_lr"] * (i + 1)})
    optimizer = torch.optim.SGD(params)
    return optimizer


def _get_optimizer_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group["lr"])
    return lr


def get_lrs_timm(max_epochs, epoch_len, opt_config, lr_config, epoch_metircs=None):
    optimizer = _get_optimizer_from_config(opt_config)

    type = lr_config["type"]
    if type == "step":
        lr_scheduler = StepLRScheduler(
            optimizer, decay_t=lr_config["decay_t"], decay_rate=0.1,
            warmup_t=lr_config["warmup_t"], warmup_lr_init=lr_config["warmup_lr_init"],
            t_in_epochs=lr_config["t_in_epochs"])
    elif type == "multistep":
        lr_scheduler = MultiStepLRScheduler(
            optimizer, decay_t=lr_config["decay_t"], decay_rate=0.1,
            warmup_t=lr_config["warmup_t"], warmup_lr_init=lr_config["warmup_lr_init"],
            t_in_epochs=lr_config["t_in_epochs"])
    elif type == "cosine_restart":
        lr_scheduler = CosineLRScheduler(
            optimizer, t_initial=lr_config["t_initial"], warmup_t=lr_config["warmup_t"],
            warmup_lr_init=lr_config["warmup_lr_init"], t_in_epochs=lr_config["t_in_epochs"],
            cycle_limit=lr_config["cycle_limit"])
    elif type == "plateau":
        lr_scheduler = PlateauLRScheduler(
            optimizer, decay_rate=0.1, patience_t=lr_config["patience_t"], mode="min",
            warmup_t=lr_config["warmup_t"], warmup_lr_init=lr_config["warmup_lr_init"])

    if epoch_metircs:
        assert len(epoch_metircs) == max_epochs

    lrs = []
    for epoch in range(max_epochs):
        for step in range(epoch_len):
            lrs.append(_get_optimizer_lr(optimizer))
            optimizer.step()
            lr_scheduler.step_update(epoch * epoch_len + step + 1)
        if epoch_metircs:
            lr_scheduler.step(epoch + 1, epoch_metircs[epoch])
        else:
            lr_scheduler.step(epoch + 1)
    return lrs


def get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config, epoch_metircs=None):
    optimizer = _get_optimizer_from_config(opt_config)

    type = lr_config.pop("type")
    if type == "step":
        torch_scheduler = StepLR(optimizer, step_size=lr_config.pop("step_size"), gamma=0.1)
    elif type == "multistep":
        torch_scheduler = MultiStepLR(optimizer, milestones=lr_config.pop("milestones"), gamma=0.1)
    elif type == "cosine":
        T_max = max_epochs if lr_config["by_epoch"] else max_epochs * epoch_len
        torch_scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif type == "cosine_restart":
        torch_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=lr_config.pop("T_0"))
    elif type == "plateau":
        torch_scheduler = ReduceLROnPlateau(optimizer, patience=lr_config.pop("patience"))
    lr_scheduler = LRWarmupScheduler(torch_scheduler, **lr_config)

    if epoch_metircs:
        assert len(epoch_metircs) == max_epochs

    lrs = []
    for epoch in range(max_epochs):
        for _ in range(epoch_len):
            lrs.append(_get_optimizer_lr(optimizer))
            optimizer.step()
            lr_scheduler.iter_update()
        if epoch_metircs:
            lr_scheduler.epoch_update(epoch_metircs[epoch])
        else:
            lr_scheduler.epoch_update()
    return lrs


def get_lrs_torch(max_epochs, epoch_len, opt_config, lr_config, epoch_metircs=None):
    optimizer = _get_optimizer_from_config(opt_config)

    type = lr_config.pop("type")
    if type == "step":
        lr_scheduler = StepLR(optimizer, step_size=lr_config.pop("step_size"), gamma=0.1)
    elif type == "multistep":
        lr_scheduler = MultiStepLR(optimizer, milestones=lr_config.pop("milestones"), gamma=0.1)
    elif type == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif type == "cosine_restart":
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=lr_config["T_0"])
    elif type == "plateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=lr_config["patience"])

    if epoch_metircs:
        assert len(epoch_metircs) == max_epochs

    lrs = []
    for epoch in range(max_epochs):
        for _ in range(epoch_len):
            lrs.append(_get_optimizer_lr(optimizer))
            optimizer.step()
        if epoch_metircs:
            lr_scheduler.step(epoch_metircs[epoch])
        else:
            lr_scheduler.step()
    return lrs


def allclose(list1, list2):
    for a, b in zip(list1, list2):
        if not np.allclose(a, b):
            return False
    return True


def test_no_warmup():
    """cpu vs torch"""
    max_epochs = 10
    epoch_len = 3
    base_lr = 5
    by_epoch = True

    for num_pram_groups in [1, 2]:
        opt_config = dict(num_pram_groups=num_pram_groups, base_lr=base_lr)

        #### StepLR
        lr_config = dict(type="step", step_size=3, by_epoch=by_epoch, epoch_len=epoch_len)
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config)

        lr_config = dict(type="step", step_size=3)
        lrs_torch = get_lrs_torch(max_epochs, epoch_len, opt_config, lr_config)
        assert allclose(lrs_cpu, lrs_torch)

        #### MultiStepLR
        lr_config = dict(type="multistep", milestones=[4, 7], by_epoch=by_epoch, epoch_len=epoch_len)
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config)

        lr_config = dict(type="multistep", milestones=[4, 7])
        lrs_torch = get_lrs_torch(max_epochs, epoch_len, opt_config, lr_config)
        assert allclose(lrs_cpu, lrs_torch)

        #### CosineAnnealingLR
        lr_config = dict(type="cosine", by_epoch=by_epoch, epoch_len=epoch_len)
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config)

        lr_config = dict(type="cosine")
        lrs_torch = get_lrs_torch(max_epochs, epoch_len, opt_config, lr_config)
        assert allclose(lrs_cpu, lrs_torch)

        #### ReduceLROnPlateau
        lr_config = dict(type="plateau", patience=2, by_epoch=by_epoch, epoch_len=epoch_len)
        epoch_metrics = list(range(10, 0, -1))
        epoch_metrics[3:7] = [7, 7, 7, 7]
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config, epoch_metrics)

        lr_config = dict(type="plateau", patience=2)
        lrs_torch = get_lrs_torch(max_epochs, epoch_len, opt_config, lr_config, epoch_metrics)
        assert allclose(lrs_cpu, lrs_torch)


def test_fix():
    """cpu vs timm"""
    epoch = 20
    epoch_len = 3
    base_lr = 5
    for by_epoch in [True, False]:
        for num_pram_groups in [1, 2]:
            opt_config = dict(num_pram_groups=num_pram_groups, base_lr=base_lr)

            #### StepLR
            lr_config = dict(type="step", step_size=3, by_epoch=by_epoch, epoch_len=epoch_len,
                             warmup_t=5, warmup_by_epoch=by_epoch, warmup_mode="fix", warmup_init_lr=0.005)
            lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_config, lr_config)

            lr_config = dict(type="step", decay_t=3, decay_rate=0.1,
                             warmup_t=5, warmup_lr_init=0.005, t_in_epochs=by_epoch)
            lrs_timm = get_lrs_timm(epoch, epoch_len, opt_config, lr_config)
            assert allclose(lrs_cpu, lrs_timm)

            #### MultiStepLR
            lr_config = dict(type="multistep", milestones=[12, 16], by_epoch=by_epoch, epoch_len=epoch_len,
                             warmup_t=5, warmup_by_epoch=by_epoch, warmup_mode="fix", warmup_init_lr=0.005)
            lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_config, lr_config)

            lr_config = dict(type="multistep", decay_t=[13, 17], decay_rate=0.1,
                             warmup_t=5, warmup_lr_init=0.005, t_in_epochs=by_epoch)
            lrs_timm = get_lrs_timm(epoch, epoch_len, opt_config, lr_config)
            assert allclose(lrs_cpu, lrs_timm)

            #### CosineAnnealingRestarts
            lr_config = dict(type="cosine_restart", T_0=10, by_epoch=by_epoch, epoch_len=epoch_len,
                             warmup_t=5, warmup_by_epoch=by_epoch, warmup_mode="fix", warmup_init_lr=0.005)
            lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_config, lr_config)

            lr_config = dict(type="cosine_restart", t_initial=10, warmup_t=5, warmup_lr_init=0.005,
                             t_in_epochs=by_epoch, cycle_limit=20)
            lrs_timm = get_lrs_timm(epoch, epoch_len, opt_config, lr_config)
            assert allclose(lrs_cpu, lrs_timm)

            if by_epoch:
                #### ReduceLROnPlateau - test 1
                lr_config = dict(type="plateau", patience=2, by_epoch=by_epoch, epoch_len=epoch_len,
                                 warmup_t=5, warmup_by_epoch=by_epoch, warmup_mode="fix", warmup_init_lr=0.005)
                epoch_metrics = list(range(20, 0, -1))
                epoch_metrics[7:11] = [13, 13, 13, 13]
                epoch_metrics[13:17] = [7, 7, 7, 7]
                lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_config, lr_config, epoch_metrics)

                lr_config = dict(type="plateau", decay_rate=0.1, patience_t=2,
                                 mode="min", warmup_t=5, warmup_lr_init=0.005)
                lrs_timm = get_lrs_timm(epoch, epoch_len, opt_config, lr_config, epoch_metrics)
                assert allclose(lrs_cpu, lrs_timm)

                #### ReduceLROnPlateau - test 2
                lr_config = dict(type="plateau", patience=2, by_epoch=by_epoch, epoch_len=epoch_len,
                                 warmup_t=8, warmup_by_epoch=by_epoch, warmup_mode="fix", warmup_init_lr=0.005)
                epoch_metrics = list(range(20, 0, -1))
                epoch_metrics[7:11] = [13, 13, 13, 13]
                epoch_metrics[13:17] = [7, 7, 7, 7]
                lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_config, lr_config, epoch_metrics)

                lr_config = dict(type="plateau", decay_rate=0.1, patience_t=2,
                                 mode="min", warmup_t=8, warmup_lr_init=0.005)
                lrs_timm = get_lrs_timm(epoch, epoch_len, opt_config, lr_config, epoch_metrics)
                assert allclose(lrs_cpu, lrs_timm)


def test_factor():
    max_epochs = 20
    epoch_len = 3
    base_lr = 5
    warmup_by_epoch = False
    for by_epoch in [True, False]:
        for num_pram_groups in [1, 2]:
            opt_config = dict(num_pram_groups=num_pram_groups, base_lr=base_lr)

            #### StepLR
            lr_config = dict(type="step", step_size=3, by_epoch=by_epoch, epoch_len=epoch_len,
                             warmup_t=5, warmup_by_epoch=warmup_by_epoch, warmup_mode="factor", warmup_factor=0.001)
            lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config)

            lr_config = dict(type='StepLrUpdaterHook', warmup='linear', warmup_iters=5,
                             warmup_ratio=0.001, step=3, by_epoch=by_epoch)
            runner = MMCVRunner(max_epochs, epoch_len, opt_config, lr_config)
            lrs_mmcv = runner.run()
            assert allclose(lrs_cpu, lrs_mmcv)

            #### MultiStepLR
            lr_config = dict(type="multistep", milestones=[8, 11], by_epoch=by_epoch, epoch_len=epoch_len,
                             warmup_t=5, warmup_by_epoch=warmup_by_epoch, warmup_mode="factor", warmup_factor=0.001)
            lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config)

            lr_config = dict(type='StepLrUpdaterHook', warmup='linear', warmup_iters=5,
                             warmup_ratio=0.001, step=[8, 11], by_epoch=by_epoch)
            runner = MMCVRunner(max_epochs, epoch_len, opt_config, lr_config)
            lrs_mmcv = runner.run()
            assert allclose(lrs_cpu, lrs_mmcv)

            #### CosineAnnealingRestarts
            lr_config = dict(type="cosine_restart", T_0=10, by_epoch=by_epoch, epoch_len=epoch_len,
                             warmup_t=5, warmup_by_epoch=warmup_by_epoch, warmup_mode="factor", warmup_factor=0.001)
            lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config)

            lr_config = dict(type='CosineRestartLrUpdaterHook', warmup='linear', warmup_iters=5,
                             warmup_ratio=0.001, periods=[10] * 2 if by_epoch else [10] * 6,
                             restart_weights=[1] * 2 if by_epoch else [1] * 6,
                             min_lr=0, by_epoch=by_epoch)
            runner = MMCVRunner(max_epochs, epoch_len, opt_config, lr_config)
            lrs_mmcv = runner.run()
            assert allclose(lrs_cpu, lrs_mmcv)


def test_auto():
    max_epochs = 20
    epoch_len = 3
    base_lr = 5
    by_epoch = False
    warmup_by_epoch = False
    for num_pram_groups in [1, 2]:
        opt_config = dict(num_pram_groups=num_pram_groups, base_lr=base_lr)

        #### MultiStepLR
        lr_config = dict(type="multistep", milestones=[30, 50], by_epoch=by_epoch, epoch_len=epoch_len,
                         warmup_t=10, warmup_by_epoch=warmup_by_epoch, warmup_mode="auto", warmup_factor=0.001)
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config)

        lr_config = dict(type="multistep", steps=[30, 50], warmup_factor=0.001, warmup_iters=10)
        lrs_d2 = get_lrs_d2(max_epochs, epoch_len, opt_config, lr_config)
        assert allclose(lrs_cpu, lrs_d2)

        #### CosineAnnealing
        lr_config = dict(type="cosine", by_epoch=by_epoch, epoch_len=epoch_len,
                         warmup_t=10, warmup_by_epoch=warmup_by_epoch, warmup_mode="auto", warmup_factor=0.001)
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config)

        lr_config = dict(type="cosine", warmup_factor=0.001, warmup_iters=10)
        lrs_d2 = get_lrs_d2(max_epochs, epoch_len, opt_config, lr_config)
        assert allclose(lrs_cpu, lrs_d2)


def test_other_cases():
    # from detectron2 test
    lr_config = dict(type="multistep", milestones=[10, 15, 20], by_epoch=False,
                     warmup_t=5, warmup_by_epoch=False, warmup_mode="fix", warmup_init_lr=0.005)
    opt_config = dict(num_pram_groups=1, base_lr=5)
    lrs_cpu = get_lrs_cpu(30, 1, opt_config, lr_config)
    lrs_cpu = [lr[0] for lr in lrs_cpu]
    assert np.allclose(lrs_cpu[:5], [0.005, 1.004, 2.003, 3.002, 4.001])
    assert np.allclose(lrs_cpu[5:10], 5.0)
    assert np.allclose(lrs_cpu[10:15], 0.5)
    assert np.allclose(lrs_cpu[15:20], 0.05)
    assert np.allclose(lrs_cpu[20:], 0.005)

    # warmup_iters % epoch_len == 0
    max_epochs = 3
    epoch_len = 3
    opt_config = dict(num_pram_groups=1, base_lr=5)
    lr_config = dict(type="step", step_size=1, by_epoch=True, epoch_len=epoch_len,
                     warmup_t=3, warmup_by_epoch=False, warmup_mode="factor", warmup_factor=0.001)
    lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_config, lr_config)

    lr_config = dict(type='StepLrUpdaterHook', warmup='linear', warmup_iters=3,
                     warmup_ratio=0.001, step=1, by_epoch=True)
    runner = MMCVRunner(max_epochs, epoch_len, opt_config, lr_config)
    lrs_mmcv = runner.run()
    assert allclose(lrs_cpu, lrs_mmcv)
