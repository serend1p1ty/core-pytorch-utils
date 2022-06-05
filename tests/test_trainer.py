import logging
import math
import os
import re
import tempfile

import mock
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import cpu.logger as logger
from cpu.hooks import EvalHook, HookBase
from cpu.trainer import MetricStorage, Trainer

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    _TF_AVAILABLE = False
else:
    _TF_AVAILABLE = True
    tf.disable_v2_behavior()


class _SimpleModel(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.device = device
        self.to(device)

    def forward(self, data):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)
        return F.mse_loss(self.fc(x), y)


class _SimpleDataset:

    def __init__(self):
        self.data = torch.rand(10, 10)
        self.target = torch.rand(10, 10)

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return self.data[index], self.target[index]


# a random but unchanged dataset
_simple_dataset = _SimpleDataset()


def _reset_logger():
    cpu_logger = logging.Logger.manager.loggerDict["cpu"]
    if hasattr(cpu_logger, "handlers"):
        cpu_logger.handlers = []
    logger.logger_initialized.clear()


def _create_new_trainer(
    max_epochs=10,
    log_period=1,
    checkpoint_period=1,
    work_dir="work_dir",
    max_num_checkpoints=None,
    enable_amp=False,
    device="cpu",
    plateau=False,
    step_size=3,
    patience=2,
):
    _reset_logger()

    model = _SimpleModel(device)
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    if not plateau:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
    data_loader = DataLoader(_simple_dataset)
    trainer = Trainer(
        model,
        optimizer,
        lr_scheduler,
        data_loader,
        max_epochs=max_epochs,
        log_period=log_period,
        checkpoint_period=checkpoint_period,
        work_dir=work_dir,
        max_num_checkpoints=max_num_checkpoints,
        enable_amp=enable_amp,
    )
    return trainer


def test_basic_run():
    for log_period in [1, 4]:
        with tempfile.TemporaryDirectory() as dir:
            trainer = _create_new_trainer(log_period=log_period, work_dir=dir)
            trainer.train()

            # check counter
            assert trainer.cur_iter == trainer.max_iters - 1
            assert trainer.inner_iter == trainer.epoch_len - 1
            assert trainer.epoch == trainer.max_epochs - 1
            assert trainer.lr_scheduler.last_iter == trainer.max_iters
            assert trainer.lr_scheduler.last_epoch == trainer.max_epochs

            log_file = os.path.join(dir, "log_rank0.txt")
            assert os.path.exists(log_file)
            log_content = open(log_file).readlines()
            assert len(log_content) != 0

            # check lr/logging period/ETA
            for epoch in range(10):
                if epoch >= 0 and epoch < 3:
                    lr = 0.1
                elif epoch < 6:
                    lr = 0.01
                elif epoch < 9:
                    lr = 0.001
                else:
                    lr = 0.0001
                if log_period == 4:
                    cnt = 0
                    for line in log_content:
                        if f"Epoch: [{epoch}]" in line:
                            assert f"lr: {lr:.4g}" in line
                            assert "[3/9]" in line or "[7/9]" in line
                            assert "ETA: " in line
                            cnt += 1
                    assert cnt == 2
                else:
                    iter = 0
                    for line in log_content:
                        if f"Epoch: [{epoch}][{iter}/9]" in line:
                            iter += 1
                            assert f"lr: {lr:.4g}" in line
                            assert "ETA: " in line
                    assert iter == 10


@pytest.mark.skipif(not _TF_AVAILABLE, reason="The test needs tensorflow")
def test_tensorboard_logging():

    def run_one_test(eval_hook_period, logger_hook_period, simple_hook_period, max_epochs=9):

        class SimpleHook(HookBase):

            def __init__(self, period=1):
                self.period = period

            def after_iter(self) -> None:
                if self.every_n_inner_iters(self.period):
                    self.log(self.trainer.cur_iter, metric1=self.trainer.cur_iter, smooth=False)

        with tempfile.TemporaryDirectory() as dir:
            trainer = _create_new_trainer(max_epochs=max_epochs, work_dir=dir,
                                          log_period=logger_hook_period)
            test_func = mock.Mock(return_value={"metric2": 3.0})
            trainer.register_hooks([
                EvalHook(eval_hook_period, test_func),
                SimpleHook(simple_hook_period),
            ])
            trainer.train()

            tb_log_file = os.listdir(os.path.join(dir, "tb_logs"))
            assert len(tb_log_file) == 1
            tb_log_file = os.path.join(dir, "tb_logs", tb_log_file[0])

            lrs = []
            metric1s = []
            metric2s = []
            for event in tf.train.summary_iterator(tb_log_file):
                for value in event.summary.value:
                    if value.tag == "lr":
                        lrs.append(value.simple_value)
                    if value.tag == "metric1":
                        metric1s.append(value.simple_value)
                    if value.tag == "metric2":
                        metric2s.append(value.simple_value)
            return lrs, metric1s, metric2s

    # no period
    lrs, metric1s, metric2s = run_one_test(eval_hook_period=1, logger_hook_period=1,
                                           simple_hook_period=1)
    assert len(lrs) == 90
    true_lrs = [0.1] * 30 + [0.01] * 30 + [0.001] * 30
    for lr, true_lr in zip(lrs, true_lrs):
        assert np.allclose(lr, true_lr)
    assert len(metric1s) == 90
    assert len(metric2s) == 9

    # only eval_hook_period
    lrs, metric1s, metric2s = run_one_test(eval_hook_period=3, logger_hook_period=1,
                                           simple_hook_period=1)
    assert len(lrs) == 90
    true_lrs = [0.1] * 30 + [0.01] * 30 + [0.001] * 30
    for lr, true_lr in zip(lrs, true_lrs):
        assert np.allclose(lr, true_lr)
    assert len(metric1s) == 90
    assert len(metric2s) == 3

    # only logger_hook_period
    lrs, metric1s, metric2s = run_one_test(eval_hook_period=1, logger_hook_period=3,
                                           simple_hook_period=1)
    assert len(lrs) == 36
    true_lrs = [0.1] * 12 + [0.01] * 12 + [0.001] * 12
    for lr, true_lr in zip(lrs, true_lrs):
        assert np.allclose(lr, true_lr)
    for metric1 in metric1s:
        assert metric1 % 10 in [2, 5, 8, 9]
    assert len(metric1s) == 36
    assert len(metric2s) == 9

    # only simple_hook_period
    lrs, metric1s, metric2s = run_one_test(eval_hook_period=1, logger_hook_period=1,
                                           simple_hook_period=3)
    assert len(lrs) == 90
    true_lrs = [0.1] * 30 + [0.01] * 30 + [0.001] * 30
    for lr, true_lr in zip(lrs, true_lrs):
        assert np.allclose(lr, true_lr)
    for metric1 in metric1s:
        assert metric1 % 10 in [2, 5, 8]
    assert len(metric1s) == 27
    assert len(metric2s) == 9

    # eval_hook_period + logger_hook_period
    lrs, metric1s, metric2s = run_one_test(eval_hook_period=3, logger_hook_period=4,
                                           simple_hook_period=1)
    assert len(lrs) == 27
    true_lrs = [0.1] * 9 + [0.01] * 9 + [0.001] * 9
    for lr, true_lr in zip(lrs, true_lrs):
        assert np.allclose(lr, true_lr)
    for metric1 in metric1s:
        assert metric1 % 10 in [3, 7, 9]
    assert len(metric1s) == 27
    assert len(metric2s) == 3

    # eval_hook_period + simple_hook_period
    lrs, metric1s, metric2s = run_one_test(eval_hook_period=3, logger_hook_period=1,
                                           simple_hook_period=4)
    assert len(lrs) == 90
    true_lrs = [0.1] * 30 + [0.01] * 30 + [0.001] * 30
    for lr, true_lr in zip(lrs, true_lrs):
        assert np.allclose(lr, true_lr)
    for metric1 in metric1s:
        assert metric1 % 10 in [3, 7]
    assert len(metric1s) == 18
    assert len(metric2s) == 3

    # logger_hook_period + simple_hook_period
    lrs, metric1s, metric2s = run_one_test(eval_hook_period=1, logger_hook_period=3,
                                           simple_hook_period=4)
    assert len(lrs) == 36
    true_lrs = [0.1] * 12 + [0.01] * 12 + [0.001] * 12
    for lr, true_lr in zip(lrs, true_lrs):
        assert np.allclose(lr, true_lr)
    for metric1 in metric1s:
        assert metric1 % 10 in [3, 7]
    assert len(metric1s) == 18
    assert len(metric2s) == 9

    # eval_hook_period + logger_hook_period + simple_hook_period
    lrs, metric1s, metric2s = run_one_test(eval_hook_period=3, logger_hook_period=3,
                                           simple_hook_period=4)
    assert len(lrs) == 36
    true_lrs = [0.1] * 12 + [0.01] * 12 + [0.001] * 12
    for lr, true_lr in zip(lrs, true_lrs):
        assert np.allclose(lr, true_lr)
    for metric1 in metric1s:
        assert metric1 % 10 in [3, 7]
    assert len(metric1s) == 18
    assert len(metric2s) == 3


def test_checkpoint_and_resume(device="cpu"):
    for enable_amp in [False] if device == "cpu" else [True, False]:
        with tempfile.TemporaryDirectory() as dir1:
            trainer = _create_new_trainer(
                max_epochs=4,
                work_dir=dir1,
                checkpoint_period=3,
                enable_amp=enable_amp,
                device=device,
            )
            trainer.train()

            assert (trainer.lr - 0.01) < 1e-7
            assert trainer.lr_scheduler.last_iter == 40

            # test periodically checkpointing
            for should_ckpt_epoch in [2, 3]:
                assert os.path.exists(
                    os.path.join(dir1, f"checkpoints/epoch_{should_ckpt_epoch}.pth"))
            assert os.path.exists(os.path.join(dir1, "checkpoints/latest.pth"))

            total_losses = trainer.metric_storage._history["total_loss"]._history

            epoch_3_smoothed_losses = []
            for line in open(os.path.join(dir1, "log_rank0.txt")):
                if "Epoch: [3]" not in line:
                    continue
                res = re.findall(r"total_loss: \S+", line)
                epoch_3_smoothed_losses.append(res[0])

            # resume training from the "epoch_2.pth"
            with tempfile.TemporaryDirectory() as dir2:
                trainer = _create_new_trainer(
                    max_epochs=4,
                    work_dir=dir2,
                    checkpoint_period=3,
                    enable_amp=enable_amp,
                    device=device,
                )
                trainer.load_checkpoint(os.path.join(dir1, "checkpoints/epoch_2.pth"))
                assert (trainer.lr - 0.01) < 1e-7
                assert trainer.lr_scheduler.last_iter == 30
                trainer.train()

                # test periodically checkpointing
                assert os.path.exists(os.path.join(dir2, "checkpoints/epoch_3.pth"))
                assert os.path.exists(os.path.join(dir2, "checkpoints/latest.pth"))

                total_losses_resume = trainer.metric_storage._history["total_loss"]._history

                epoch_3_smoothed_losses_resume = []
                for line in open(os.path.join(dir2, "log_rank0.txt")):
                    if "Epoch: [3]" not in line:
                        continue
                    res = re.findall(r"total_loss: \S+", line)
                    epoch_3_smoothed_losses_resume.append(res[0])

                # If the model/optimizer/lr_scheduler resumes correctly,
                # the training losses should be the same.
                for loss1, loss2 in zip(total_losses, total_losses_resume):
                    if device == "cpu":
                        assert loss1 == loss2
                    else:
                        assert abs(loss1 - loss2) < 1e-6

                # If the metric storage resumes correctly,
                # the training smoothed losses should be the same too.
                for loss1, loss2 in zip(epoch_3_smoothed_losses, epoch_3_smoothed_losses_resume):
                    assert loss1 == loss2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The test needs cuda")
def test_checkpoint_and_resume_cuda():
    test_checkpoint_and_resume(device="cuda")


def test_eval_hook():
    for total_epochs, period, eval_count in [(30, 15, 2), (31, 15, 3), (20, 0, 1)]:
        with tempfile.TemporaryDirectory() as dir:
            test_func = mock.Mock(return_value={"metric": 3.0})
            trainer = _create_new_trainer(max_epochs=total_epochs, work_dir=dir)
            trainer.register_hooks([EvalHook(period, test_func)])
            trainer.train()
            assert test_func.call_count == eval_count


def test_checkpoint_hook():
    with tempfile.TemporaryDirectory() as dir:
        trainer = _create_new_trainer(max_epochs=10, work_dir=dir, max_num_checkpoints=3,
                                      checkpoint_period=2)
        trainer.train()
        for epoch in range(10):
            if epoch in [5, 7, 9]:
                assert os.path.exists(os.path.join(dir, f"checkpoints/epoch_{epoch}.pth"))
            else:
                assert not os.path.exists(os.path.join(dir, f"checkpoints/epoch_{epoch}.pth"))


def test_lr_update_hook():
    # test ReduceLROnPlateau
    with tempfile.TemporaryDirectory() as dir:
        eval_cnt = [-1]
        eval_metrics = [10, 9, 8, 7, 7, 7, 7, 3, 2, 1]

        def eval_func():
            eval_cnt[0] += 1
            return {"Eval Metric": eval_metrics[eval_cnt[0]]}

        class CollectLRHook(HookBase):

            def __init__(self):
                self.lrs = []

            def after_iter(self):
                self.lrs.append(self.metric_storage["lr"].latest)

        trainer = _create_new_trainer(max_epochs=10, work_dir=dir, plateau=True)
        collect_lr_hook = CollectLRHook()
        trainer.register_hooks([EvalHook(1, eval_func), collect_lr_hook])
        trainer.train()
        true_lrs = [0.1] * 70 + [0.01] * 30
        assert np.allclose(collect_lr_hook.lrs, true_lrs)


def test_hook_priority():

    class Hook1(HookBase):
        priority = 1

    class Hook2(HookBase):
        priority = 2

    class Hook3(HookBase):
        priority = 3

    with tempfile.TemporaryDirectory() as dir:
        trainer = _create_new_trainer(work_dir=dir)
        trainer.register_hooks([Hook3()])
        trainer.register_hooks([Hook1()])
        hooks = [hook for hook in trainer._hooks if isinstance(hook, (Hook1, Hook2, Hook3))]
        assert isinstance(hooks[0], Hook1)
        assert isinstance(hooks[1], Hook3)
        trainer.register_hooks([Hook2()])
        hooks = [hook for hook in trainer._hooks if isinstance(hook, (Hook1, Hook2, Hook3))]
        assert isinstance(hooks[0], Hook1)
        assert isinstance(hooks[1], Hook2)
        assert isinstance(hooks[2], Hook3)


def test_metric_storage():
    # without smooth
    metric_storage = MetricStorage(window_size=4)
    metric_storage.update(0, loss=0.7, accuracy=0.1, smooth=False)
    metric_storage.update(1, loss=0.6, accuracy=0.2, smooth=False)
    metric_storage.update(2, loss=0.4, accuracy=0.3, smooth=False)
    metric_storage.update(3, loss=0.3, accuracy=0.7, smooth=False)
    assert metric_storage.values_maybe_smooth["loss"] == (3, 0.3)
    assert metric_storage.values_maybe_smooth["accuracy"] == (3, 0.7)
    assert abs(metric_storage["loss"].global_avg - 0.5) < 1e-7
    assert metric_storage["accuracy"].global_avg == 0.325
    metric_storage.update(4, loss=0.5, accuracy=0.6, smooth=False)
    metric_storage.update(5, loss=0.1, accuracy=0.8, smooth=False)
    assert metric_storage.values_maybe_smooth["loss"] == (5, 0.1)
    assert metric_storage.values_maybe_smooth["accuracy"] == (5, 0.8)
    assert metric_storage["loss"].global_avg == 2.6 / 6
    assert metric_storage["accuracy"].global_avg == 0.45

    # with smooth
    metric_storage = MetricStorage(window_size=4)
    metric_storage.update(0, loss=0.7, accuracy=0.1)
    metric_storage.update(1, loss=0.6, accuracy=0.2)
    metric_storage.update(2, loss=0.4, accuracy=0.3)
    metric_storage.update(3, loss=0.3, accuracy=0.7)
    assert metric_storage.values_maybe_smooth["loss"][0] == 3
    assert abs(metric_storage.values_maybe_smooth["loss"][1] - 0.5) < 1e-7
    assert metric_storage.values_maybe_smooth["accuracy"] == (3, 1.3 / 4)
    assert abs(metric_storage["loss"].global_avg - 0.5) < 1e-7
    assert metric_storage["accuracy"].global_avg == 0.325
    metric_storage.update(4, loss=0.5, accuracy=0.6)
    metric_storage.update(5, loss=0.1, accuracy=0.8)
    assert metric_storage.values_maybe_smooth["loss"] == (5, 1.3 / 4)
    assert math.isclose(metric_storage.values_maybe_smooth["accuracy"][0], 5)
    assert math.isclose(metric_storage.values_maybe_smooth["accuracy"][1], 2.4 / 4)
    assert metric_storage["loss"].global_avg == 2.6 / 6
    assert metric_storage["accuracy"].global_avg == 0.45
