import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from cpu.lr_scheduler import LRWarmupScheduler


def get_lrs1(epoch, epoch_len, optimizer, lr_scheduler):
    lrs = []
    for _ in range(epoch):
        for _ in range(epoch_len):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
        lr_scheduler.step()
    return lrs


def get_lrs2(epoch, epoch_len, optimizer, lr_scheduler):
    lrs = []
    for _ in range(epoch):
        for _ in range(epoch_len):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
    return lrs


def test_warmup_multistep():
    param = nn.Parameter(torch.zeros(0))
    optimizer = torch.optim.SGD([param], lr=5)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.1)
    warmup_scheduler = LRWarmupScheduler(
        scheduler=scheduler,
        epoch_len=1,
        warmup_method="linear",
        warmup_iters=5,
        warmup_factor=0.001,
    )

    lrs = get_lrs2(30, 1, optimizer, warmup_scheduler)

    assert np.allclose(lrs[:5], [0.005, 1.004, 2.003, 3.002, 4.001])
    assert np.allclose(lrs[5:10], 5.0)
    assert np.allclose(lrs[10:15], 0.5)
    assert np.allclose(lrs[15:20], 0.05)
    assert np.allclose(lrs[20:], 0.005)


def test_multistep():
    param = nn.Parameter(torch.zeros(0))
    optimizer = torch.optim.SGD([param], lr=5.0)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.1)
    lrs1 = get_lrs1(30, 3, optimizer, scheduler)

    optimizer = torch.optim.SGD([param], lr=5.0)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.1)
    warmup_scheduler = LRWarmupScheduler(scheduler=scheduler, epoch_len=3)
    lrs2 = get_lrs2(30, 3, optimizer, warmup_scheduler)

    assert np.allclose(lrs1, lrs2)


def test_cosine():
    param = nn.Parameter(torch.zeros(0))
    optimizer = torch.optim.SGD([param], lr=5.0)
    scheduler = CosineAnnealingLR(optimizer, 1)
    lrs1 = get_lrs1(30, 3, optimizer, scheduler)

    optimizer = torch.optim.SGD([param], lr=5.0)
    scheduler = CosineAnnealingLR(optimizer, 1)
    warmup_scheduler = LRWarmupScheduler(scheduler=scheduler, epoch_len=3)
    lrs2 = get_lrs2(30, 3, optimizer, warmup_scheduler)

    assert np.allclose(lrs1, lrs2)
