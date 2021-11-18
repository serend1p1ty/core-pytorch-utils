import logging
import os
import re
import tempfile
import time

import mock
import torch
from torch import nn
from torch.utils.data import DataLoader

import cpu.logger as logger
from cpu.hooks import EvalHook, HookBase
from cpu.trainer import Trainer


class _SimpleModel(nn.Module):
    def __init__(self, sleep_sec=0):
        super().__init__()
        self.fc = nn.Linear(3, 3)
        self.sleep_sec = sleep_sec

    def forward(self, x):
        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)
        return {"loss": x.sum() + sum([x.mean() for x in self.parameters()])}


class _SimpleDataset:
    def __init__(self):
        self.data = torch.rand(10, 3)

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return self.data[index]


# a random but unchanged dataset in the whole testing phase
_simple_dataset = _SimpleDataset()


def _reset_logger():
    cpu_logger = logging.Logger.manager.loggerDict["cpu"]
    if hasattr(cpu_logger, "handlers"):
        cpu_logger.handlers = []
    logger.logger_initialized.clear()


def _create_new_trainer(max_epochs=10, log_period=1, checkpoint_period=1, work_dir="work_dir"):
    _reset_logger()

    model = _SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
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
    )
    return trainer


def test_basic_run_with_log_period():
    with tempfile.TemporaryDirectory() as dir:
        trainer = _create_new_trainer(log_period=4, work_dir=dir)
        trainer.train()

        # check counter
        assert trainer.iter == trainer.max_iters - 1
        assert trainer.inner_iter == trainer.epoch_len - 1
        assert trainer.epoch == trainer.max_epochs - 1
        assert trainer.lr_scheduler.last_epoch == trainer.max_iters

        log_file = os.path.join(dir, "log.txt")
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
            cnt = 0
            for line in log_content:
                if f"Epoch: [{epoch}]" in line:
                    assert f"lr: {lr:.4g}" in line
                    assert "[3/9]" in line or "[7/9]" in line
                    assert "ETA: " in line
                    cnt += 1
            assert cnt == 2


def test_basic_run_without_log_period():
    with tempfile.TemporaryDirectory() as dir:
        trainer = _create_new_trainer(work_dir=dir)
        trainer.train()

        # check counter
        assert trainer.iter == trainer.max_iters - 1
        assert trainer.inner_iter == trainer.epoch_len - 1
        assert trainer.epoch == trainer.max_epochs - 1
        assert trainer.lr_scheduler.last_epoch == trainer.max_iters

        log_file = os.path.join(dir, "log.txt")
        assert os.path.exists(log_file)
        log_content = open(log_file).readlines()
        assert len(log_content) != 0

        # check lr/ETA
        for epoch in range(10):
            if epoch >= 0 and epoch < 3:
                lr = 0.1
            elif epoch < 6:
                lr = 0.01
            elif epoch < 9:
                lr = 0.001
            else:
                lr = 0.0001
            iter = 0
            for line in log_content:
                if f"Epoch: [{epoch}][{iter}/9]" in line:
                    iter += 1
                    assert f"lr: {lr:.4g}" in line
                    assert "ETA: " in line
            assert iter == 10


def test_tensorboard_logging():
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        return
    else:
        tf.disable_v2_behavior()

    class SimpleHook(HookBase):
        def after_iter(self) -> None:
            self.storage.update(self.trainer.iter, metric1=self.trainer.iter, smooth=False)

    with tempfile.TemporaryDirectory() as dir:
        trainer = _create_new_trainer(max_epochs=9, work_dir=dir)
        test_func = mock.Mock(return_value={"metric2": 3.0})
        trainer.register_hooks([EvalHook(3, test_func), SimpleHook()])
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
        assert len(lrs) == 90
        true_lrs = [0.1] * 30 + [0.01] * 30 + [0.001] * 30
        assert sum([lrs[i] - true_lrs[i] for i in range(90)]) < 1e-7
        assert len(metric1s) == 90
        assert len(metric2s) == 3


def test_checkpoint_and_resume():
    with tempfile.TemporaryDirectory() as dir1:
        trainer = _create_new_trainer(max_epochs=4, work_dir=dir1, checkpoint_period=3)
        trainer.train()

        assert (trainer.lr - 0.01) < 1e-7
        assert trainer.lr_scheduler.last_epoch == 40

        # test periodically checkpointing
        for should_ckpt_epoch in [2, 3]:
            assert os.path.exists(os.path.join(dir1, f"checkpoints/epoch_{should_ckpt_epoch}.pth"))
        assert os.path.exists(os.path.join(dir1, "checkpoints/latest.pth"))

        epoch_3_losses = [
            loss
            for (iter, loss) in trainer.metric_storage._history["total_loss"]._history
            if iter >= 30 and iter < 40
        ]

        epoch_3_smoothed_losses = []
        for line in open(os.path.join(dir1, "log.txt")):
            if "Epoch: [3]" not in line:
                continue
            res = re.findall(r"total_loss: \d+.\d+", line)
            epoch_3_smoothed_losses.append(res[0])

        # resume training from the "epoch_2.pth"
        with tempfile.TemporaryDirectory() as dir2:
            trainer = _create_new_trainer(max_epochs=4, work_dir=dir2, checkpoint_period=3)
            trainer.load_checkpoint(os.path.join(dir1, "checkpoints/epoch_2.pth"))
            assert (trainer.lr - 0.01) < 1e-7
            assert trainer.lr_scheduler.last_epoch == 30
            trainer.train()

            # test periodically checkpointing
            assert os.path.exists(os.path.join(dir2, "checkpoints/epoch_3.pth"))
            assert os.path.exists(os.path.join(dir2, "checkpoints/latest.pth"))

            epoch_3_losses_resume = [
                loss
                for (iter, loss) in trainer.metric_storage._history["total_loss"]._history
                if iter >= 30 and iter < 40
            ]

            epoch_3_smoothed_losses_resume = []
            for line in open(os.path.join(dir2, "log.txt")):
                if "Epoch: [3]" not in line:
                    continue
                res = re.findall(r"total_loss: \d+.\d+", line)
                epoch_3_smoothed_losses_resume.append(res[0])

            # If the model/optimizer/lr_scheduler resumes correctly,
            # the training losses should be the same.
            for loss1, loss2 in zip(epoch_3_losses, epoch_3_losses_resume):
                assert loss1 == loss2

            # If the metric storage resumes correctly,
            # the training smoothed losses should be the same too.
            for loss1, loss2 in zip(epoch_3_smoothed_losses, epoch_3_smoothed_losses_resume):
                assert loss1 == loss2


def test_eval_hook():
    with tempfile.TemporaryDirectory() as dir:
        for total_epochs, period, eval_count in [(30, 15, 2), (31, 15, 3), (20, 0, 1)]:
            test_func = mock.Mock(return_value={"metric": 3.0})
            trainer = _create_new_trainer(max_epochs=total_epochs, work_dir=dir)
            trainer.register_hooks([EvalHook(period, test_func)])
            trainer.train()
            assert test_func.call_count == eval_count
