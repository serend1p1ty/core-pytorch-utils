import math

from cpu.metric_storage import MetricStorage, SmoothedValue


def test_smoothed_value():
    smoothed_value = SmoothedValue(window_size=4)
    smoothed_value.update(0.1)
    smoothed_value.update(0.2)
    smoothed_value.update(0.5)
    smoothed_value.update(0.8)
    assert smoothed_value.avg == 0.4
    assert smoothed_value.global_avg == 0.4
    smoothed_value.update(1.0)
    assert smoothed_value.avg == 0.625
    assert smoothed_value.global_avg == 0.52


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
