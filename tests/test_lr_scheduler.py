import math

import mmcv
import numpy as np
import torch
import torch.nn as nn
from fvcore.common.param_scheduler import (CompositeParamScheduler, ConstantParamScheduler,
                                           CosineParamScheduler, LinearParamScheduler,
                                           MultiStepParamScheduler)
from mmcv.runner.hooks import HOOKS
from timm.scheduler import (CosineLRScheduler, MultiStepLRScheduler, PlateauLRScheduler,
                            StepLRScheduler)
from torch.optim.lr_scheduler import (CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR,
                                      ReduceLROnPlateau, StepLR)

from cpu.lr_scheduler import LRWarmupScheduler


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


def get_lrs_d2(max_epochs, epoch_len, opt_cfg, sche_cfg):
    optimizer = _get_optimizer_from_config(opt_cfg)

    max_iters = max_epochs * epoch_len
    if sche_cfg["type"] == "multistep":
        steps = sche_cfg["steps"]
        sche = MultiStepParamScheduler(
            values=[0.1**k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=max_iters,
        )
    elif sche_cfg["type"] == "cosine":
        sche = CosineParamScheduler(1, 0)

    sche = WarmupParamScheduler(sche, sche_cfg["warmup_factor"],
                                min(sche_cfg["warmup_iters"] / max_iters, 1.0), "linear")
    lr_scheduler = LRMultiplier(optimizer, multiplier=sche, max_iter=max_iters)

    lrs = []
    for _ in range(max_epochs):
        for _ in range(epoch_len):
            lrs.append(_get_optimizer_lr(optimizer))
            optimizer.step()
            lr_scheduler.step()
    return lrs


class MMCVRunner:

    def __init__(self, max_epochs, epoch_len, opt_cfg, sche_cfg):
        self.max_epochs = max_epochs
        self.epoch_len = epoch_len
        self.optimizer = _get_optimizer_from_config(opt_cfg)
        self.hooks = []
        self.register_hook_from_cfg(sche_cfg)

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


def _get_optimizer_from_config(opt_cfg):
    params = []
    for i in range(opt_cfg["num_pram_groups"]):
        params.append({"params": nn.Parameter(torch.zeros(0)), "lr": opt_cfg["base_lr"] * (i + 1)})
    optimizer = torch.optim.SGD(params)
    return optimizer


def _get_optimizer_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group["lr"])
    return lr


def get_lrs_timm(max_epochs, epoch_len, opt_cfg, sche_cfg, epoch_metircs=None):
    optimizer = _get_optimizer_from_config(opt_cfg)

    type = sche_cfg["type"]
    if type == "step":
        lr_scheduler = StepLRScheduler(optimizer, decay_t=sche_cfg["decay_t"], decay_rate=0.1,
                                       warmup_t=sche_cfg["warmup_t"],
                                       warmup_lr_init=sche_cfg["warmup_lr_init"],
                                       t_in_epochs=sche_cfg["t_in_epochs"])
    elif type == "multistep":
        lr_scheduler = MultiStepLRScheduler(optimizer, decay_t=sche_cfg["decay_t"], decay_rate=0.1,
                                            warmup_t=sche_cfg["warmup_t"],
                                            warmup_lr_init=sche_cfg["warmup_lr_init"],
                                            t_in_epochs=sche_cfg["t_in_epochs"])
    elif type == "cosine_restart":
        lr_scheduler = CosineLRScheduler(optimizer, t_initial=sche_cfg["t_initial"],
                                         warmup_t=sche_cfg["warmup_t"],
                                         warmup_lr_init=sche_cfg["warmup_lr_init"],
                                         t_in_epochs=sche_cfg["t_in_epochs"],
                                         cycle_limit=sche_cfg["cycle_limit"])
    elif type == "plateau":
        lr_scheduler = PlateauLRScheduler(optimizer, decay_rate=0.1,
                                          patience_t=sche_cfg["patience_t"], mode="min",
                                          warmup_t=sche_cfg["warmup_t"],
                                          warmup_lr_init=sche_cfg["warmup_lr_init"])

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


def get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg, epoch_metircs=None):
    optimizer = _get_optimizer_from_config(opt_cfg)

    type = sche_cfg.pop("type")
    if type == "step":
        torch_scheduler = StepLR(optimizer, step_size=sche_cfg.pop("step_size"), gamma=0.1)
    elif type == "multistep":
        torch_scheduler = MultiStepLR(optimizer, milestones=sche_cfg.pop("milestones"), gamma=0.1)
    elif type == "cosine":
        T_max = max_epochs if sche_cfg.get("by_epoch", True) else max_epochs * epoch_len
        torch_scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif type == "cosine_restart":
        torch_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=sche_cfg.pop("T_0"))
    elif type == "plateau":
        torch_scheduler = ReduceLROnPlateau(optimizer, patience=sche_cfg.pop("patience"))
    lr_scheduler = LRWarmupScheduler(torch_scheduler, **sche_cfg)

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


def get_lrs_torch(max_epochs, epoch_len, opt_cfg, sche_cfg, epoch_metircs=None):
    optimizer = _get_optimizer_from_config(opt_cfg)

    type = sche_cfg["type"]
    if type == "step":
        lr_scheduler = StepLR(optimizer, step_size=sche_cfg["step_size"], gamma=0.1)
    elif type == "multistep":
        lr_scheduler = MultiStepLR(optimizer, milestones=sche_cfg["milestones"], gamma=0.1)
    elif type == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif type == "cosine_restart":
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=sche_cfg["T_0"])
    elif type == "plateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=sche_cfg["patience"])

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

    for num_pram_groups in [1, 2]:
        opt_cfg = dict(num_pram_groups=num_pram_groups, base_lr=base_lr)

        #### StepLR
        sche_cfg = dict(type="step", step_size=3)
        # get_lrs_cpu() will modify sche_cfg, so we copy it
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg.copy())
        lrs_torch = get_lrs_torch(max_epochs, epoch_len, opt_cfg, sche_cfg)
        assert allclose(lrs_cpu, lrs_torch)

        #### MultiStepLR
        sche_cfg = dict(type="multistep", milestones=[4, 7])
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg.copy())
        lrs_torch = get_lrs_torch(max_epochs, epoch_len, opt_cfg, sche_cfg)
        assert allclose(lrs_cpu, lrs_torch)

        #### CosineAnnealingLR
        sche_cfg = dict(type="cosine")
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg.copy())
        lrs_torch = get_lrs_torch(max_epochs, epoch_len, opt_cfg, sche_cfg)
        assert allclose(lrs_cpu, lrs_torch)

        #### ReduceLROnPlateau
        sche_cfg = dict(type="plateau", patience=2)
        epoch_metrics = list(range(10, 0, -1))
        epoch_metrics[3:7] = [7, 7, 7, 7]
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg.copy(), epoch_metrics)
        lrs_torch = get_lrs_torch(max_epochs, epoch_len, opt_cfg, sche_cfg, epoch_metrics)
        assert allclose(lrs_cpu, lrs_torch)


def test_fix():
    """cpu vs timm"""
    epoch = 20
    epoch_len = 3
    base_lr = 5
    for by_epoch in [True, False]:
        for num_pram_groups in [1, 2]:
            opt_cfg = dict(num_pram_groups=num_pram_groups, base_lr=base_lr)

            #### StepLR
            sche_cfg = dict(type="step", step_size=3, by_epoch=by_epoch, epoch_len=epoch_len,
                            warmup_t=5, warmup_by_epoch=by_epoch, warmup_mode="fix",
                            warmup_init_lr=0.005)
            lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_cfg, sche_cfg)

            sche_cfg = dict(type="step", decay_t=3, decay_rate=0.1, warmup_t=5,
                            warmup_lr_init=0.005, t_in_epochs=by_epoch)
            lrs_timm = get_lrs_timm(epoch, epoch_len, opt_cfg, sche_cfg)
            assert allclose(lrs_cpu, lrs_timm)

            #### MultiStepLR
            sche_cfg = dict(type="multistep", milestones=[12, 16], by_epoch=by_epoch,
                            epoch_len=epoch_len, warmup_t=5, warmup_by_epoch=by_epoch,
                            warmup_mode="fix", warmup_init_lr=0.005)
            lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_cfg, sche_cfg)

            sche_cfg = dict(type="multistep", decay_t=[13, 17], decay_rate=0.1, warmup_t=5,
                            warmup_lr_init=0.005, t_in_epochs=by_epoch)
            lrs_timm = get_lrs_timm(epoch, epoch_len, opt_cfg, sche_cfg)
            assert allclose(lrs_cpu, lrs_timm)

            #### CosineAnnealingRestarts
            sche_cfg = dict(type="cosine_restart", T_0=10, by_epoch=by_epoch, epoch_len=epoch_len,
                            warmup_t=5, warmup_by_epoch=by_epoch, warmup_mode="fix",
                            warmup_init_lr=0.005)
            lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_cfg, sche_cfg)

            sche_cfg = dict(type="cosine_restart", t_initial=10, warmup_t=5, warmup_lr_init=0.005,
                            t_in_epochs=by_epoch, cycle_limit=20)
            lrs_timm = get_lrs_timm(epoch, epoch_len, opt_cfg, sche_cfg)
            assert allclose(lrs_cpu, lrs_timm)

            if by_epoch:
                #### ReduceLROnPlateau - test 1
                sche_cfg = dict(type="plateau", patience=2, by_epoch=by_epoch, epoch_len=epoch_len,
                                warmup_t=5, warmup_by_epoch=by_epoch, warmup_mode="fix",
                                warmup_init_lr=0.005)
                epoch_metrics = list(range(20, 0, -1))
                epoch_metrics[7:11] = [13, 13, 13, 13]
                epoch_metrics[13:17] = [7, 7, 7, 7]
                lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_cfg, sche_cfg, epoch_metrics)

                sche_cfg = dict(type="plateau", decay_rate=0.1, patience_t=2, mode="min",
                                warmup_t=5, warmup_lr_init=0.005)
                lrs_timm = get_lrs_timm(epoch, epoch_len, opt_cfg, sche_cfg, epoch_metrics)
                assert allclose(lrs_cpu, lrs_timm)

                #### ReduceLROnPlateau - test 2
                sche_cfg = dict(type="plateau", patience=2, by_epoch=by_epoch, epoch_len=epoch_len,
                                warmup_t=8, warmup_by_epoch=by_epoch, warmup_mode="fix",
                                warmup_init_lr=0.005)
                epoch_metrics = list(range(20, 0, -1))
                epoch_metrics[7:11] = [13, 13, 13, 13]
                epoch_metrics[13:17] = [7, 7, 7, 7]
                lrs_cpu = get_lrs_cpu(epoch, epoch_len, opt_cfg, sche_cfg, epoch_metrics)

                sche_cfg = dict(type="plateau", decay_rate=0.1, patience_t=2, mode="min",
                                warmup_t=8, warmup_lr_init=0.005)
                lrs_timm = get_lrs_timm(epoch, epoch_len, opt_cfg, sche_cfg, epoch_metrics)
                assert allclose(lrs_cpu, lrs_timm)


def test_factor():
    max_epochs = 20
    epoch_len = 3
    base_lr = 5
    warmup_by_epoch = False
    for by_epoch in [True, False]:
        for num_pram_groups in [1, 2]:
            opt_cfg = dict(num_pram_groups=num_pram_groups, base_lr=base_lr)

            #### StepLR
            sche_cfg = dict(type="step", step_size=3, by_epoch=by_epoch, epoch_len=epoch_len,
                            warmup_t=5, warmup_by_epoch=warmup_by_epoch, warmup_mode="factor",
                            warmup_factor=0.001)
            lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg)

            sche_cfg = dict(type='StepLrUpdaterHook', warmup='linear', warmup_iters=5,
                            warmup_ratio=0.001, step=3, by_epoch=by_epoch)
            runner = MMCVRunner(max_epochs, epoch_len, opt_cfg, sche_cfg)
            lrs_mmcv = runner.run()
            assert allclose(lrs_cpu, lrs_mmcv)

            #### MultiStepLR
            sche_cfg = dict(type="multistep", milestones=[8, 11], by_epoch=by_epoch,
                            epoch_len=epoch_len, warmup_t=5, warmup_by_epoch=warmup_by_epoch,
                            warmup_mode="factor", warmup_factor=0.001)
            lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg)

            sche_cfg = dict(type='StepLrUpdaterHook', warmup='linear', warmup_iters=5,
                            warmup_ratio=0.001, step=[8, 11], by_epoch=by_epoch)
            runner = MMCVRunner(max_epochs, epoch_len, opt_cfg, sche_cfg)
            lrs_mmcv = runner.run()
            assert allclose(lrs_cpu, lrs_mmcv)

            #### CosineAnnealingRestarts
            sche_cfg = dict(type="cosine_restart", T_0=10, by_epoch=by_epoch, epoch_len=epoch_len,
                            warmup_t=5, warmup_by_epoch=warmup_by_epoch, warmup_mode="factor",
                            warmup_factor=0.001)
            lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg)

            sche_cfg = dict(type='CosineRestartLrUpdaterHook', warmup='linear', warmup_iters=5,
                            warmup_ratio=0.001, periods=[10] * 2 if by_epoch else [10] * 6,
                            restart_weights=[1] * 2 if by_epoch else [1] * 6, min_lr=0,
                            by_epoch=by_epoch)
            runner = MMCVRunner(max_epochs, epoch_len, opt_cfg, sche_cfg)
            lrs_mmcv = runner.run()
            assert allclose(lrs_cpu, lrs_mmcv)


def test_auto():
    max_epochs = 20
    epoch_len = 3
    base_lr = 5
    by_epoch = False
    warmup_by_epoch = False
    for num_pram_groups in [1, 2]:
        opt_cfg = dict(num_pram_groups=num_pram_groups, base_lr=base_lr)

        #### MultiStepLR
        sche_cfg = dict(type="multistep", milestones=[30, 50], by_epoch=by_epoch,
                        epoch_len=epoch_len, warmup_t=10, warmup_by_epoch=warmup_by_epoch,
                        warmup_mode="auto", warmup_factor=0.001)
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg)

        sche_cfg = dict(type="multistep", steps=[30, 50], warmup_factor=0.001, warmup_iters=10)
        lrs_d2 = get_lrs_d2(max_epochs, epoch_len, opt_cfg, sche_cfg)
        assert allclose(lrs_cpu, lrs_d2)

        #### CosineAnnealing
        sche_cfg = dict(type="cosine", by_epoch=by_epoch, epoch_len=epoch_len, warmup_t=10,
                        warmup_by_epoch=warmup_by_epoch, warmup_mode="auto", warmup_factor=0.001)
        lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg)

        sche_cfg = dict(type="cosine", warmup_factor=0.001, warmup_iters=10)
        lrs_d2 = get_lrs_d2(max_epochs, epoch_len, opt_cfg, sche_cfg)
        assert allclose(lrs_cpu, lrs_d2)


def test_other_cases():
    # simplest usage
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(0))], lr=5)
    torch_scheduler = CosineAnnealingLR(optimizer, T_max=10)
    lr_scheduler = LRWarmupScheduler(torch_scheduler)
    for _ in range(10):
        for _ in range(10):
            lr_scheduler.iter_update()
        lr_scheduler.epoch_update()

    # from detectron2 test_warmup_multistep()
    sche_cfg = dict(type="multistep", milestones=[10, 15, 20], by_epoch=False, warmup_t=5,
                    warmup_by_epoch=False, warmup_mode="auto", warmup_factor=0.001)
    opt_cfg = dict(num_pram_groups=1, base_lr=5)
    lrs_cpu = get_lrs_cpu(30, 1, opt_cfg, sche_cfg)
    lrs_cpu = [lr[0] for lr in lrs_cpu]
    assert np.allclose(lrs_cpu[:5], [0.005, 1.004, 2.003, 3.002, 4.001])
    assert np.allclose(lrs_cpu[5:10], 5.0)
    assert np.allclose(lrs_cpu[10:15], 0.5)
    assert np.allclose(lrs_cpu[15:20], 0.05)
    assert np.allclose(lrs_cpu[20:], 0.005)

    # from detectron2 test_warmup_cosine()
    sche_cfg = dict(type="cosine", by_epoch=False, warmup_t=5, warmup_by_epoch=False,
                    warmup_mode="auto", warmup_factor=0.001)
    opt_cfg = dict(num_pram_groups=1, base_lr=5)
    lrs_cpu = get_lrs_cpu(30, 1, opt_cfg, sche_cfg)
    lrs_cpu = [lr[0] for lr in lrs_cpu]
    for idx, lr in enumerate(lrs_cpu):
        expected_cosine = 2.5 * (1.0 + math.cos(math.pi * idx / 30))
        if idx >= 5:
            assert np.allclose(lr, expected_cosine)
        else:
            assert not np.allclose(lr, expected_cosine)

    # warmup_iters % epoch_len == 0
    max_epochs = 3
    epoch_len = 3
    opt_cfg = dict(num_pram_groups=1, base_lr=5)
    sche_cfg = dict(type="step", step_size=1, by_epoch=True, epoch_len=epoch_len, warmup_t=3,
                    warmup_by_epoch=False, warmup_mode="factor", warmup_factor=0.001)
    lrs_cpu = get_lrs_cpu(max_epochs, epoch_len, opt_cfg, sche_cfg)

    sche_cfg = dict(type='StepLrUpdaterHook', warmup='linear', warmup_iters=3, warmup_ratio=0.001,
                    step=1, by_epoch=True)
    runner = MMCVRunner(max_epochs, epoch_len, opt_cfg, sche_cfg)
    lrs_mmcv = runner.run()
    assert allclose(lrs_cpu, lrs_mmcv)
