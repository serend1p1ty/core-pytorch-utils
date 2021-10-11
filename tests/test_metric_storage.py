import math

from cpu.metric_storage import MetricStorage, _SmoothedValue


def test_smoothed_value():
    smoothed_value = _SmoothedValue(window_size=4)
    smoothed_value.update(0.1, 0)
    smoothed_value.update(0.2, 2)
    smoothed_value.update(0.5, 4)
    smoothed_value.update(0.8, 8)
    assert smoothed_value.median == (8, 0.35)
    assert smoothed_value.global_avg == (8, 0.4)
    smoothed_value.update(1.0, 9)
    assert smoothed_value.median == (9, 0.65)
    assert smoothed_value.global_avg == (9, 0.52)


def test_metric_storage():
    # without smooth
    metric_storage = MetricStorage(window_size=4)
    metric_storage.update(0, loss=0.7, accuracy=0.1, smooth=False)
    metric_storage.update(1, loss=0.6, accuracy=0.2, smooth=False)
    metric_storage.update(2, loss=0.4, accuracy=0.3, smooth=False)
    metric_storage.update(3, loss=0.3, accuracy=0.7, smooth=False)
    assert metric_storage.values_maybe_smooth["loss"] == (3, 0.3)
    assert metric_storage.values_maybe_smooth["accuracy"] == (3, 0.7)
    assert metric_storage.global_avg["loss"] == (3, 0.5)
    assert metric_storage.global_avg["accuracy"] == (3, 0.325)
    metric_storage.update(4, loss=0.5, accuracy=0.6, smooth=False)
    metric_storage.update(5, loss=0.1, accuracy=0.8, smooth=False)
    assert metric_storage.values_maybe_smooth["loss"] == (5, 0.1)
    assert metric_storage.values_maybe_smooth["accuracy"] == (5, 0.8)
    assert metric_storage.global_avg["loss"] == (5, 2.6 / 6)
    assert metric_storage.global_avg["accuracy"] == (5, 0.45)

    # with smooth
    metric_storage.clear()
    metric_storage.update(0, loss=0.7, accuracy=0.1)
    metric_storage.update(1, loss=0.6, accuracy=0.2)
    metric_storage.update(2, loss=0.4, accuracy=0.3)
    metric_storage.update(3, loss=0.3, accuracy=0.7)
    assert metric_storage.values_maybe_smooth["loss"] == (3, 0.5)
    assert metric_storage.values_maybe_smooth["accuracy"] == (3, 0.25)
    assert metric_storage.global_avg["loss"] == (3, 0.5)
    assert metric_storage.global_avg["accuracy"] == (3, 0.325)
    metric_storage.update(4, loss=0.5, accuracy=0.6)
    metric_storage.update(5, loss=0.1, accuracy=0.8)
    assert metric_storage.values_maybe_smooth["loss"] == (5, 0.35)
    assert math.isclose(metric_storage.values_maybe_smooth["accuracy"][0], 5)
    assert math.isclose(metric_storage.values_maybe_smooth["accuracy"][1], 0.65)
    assert metric_storage.global_avg["loss"] == (5, 2.6 / 6)
    assert metric_storage.global_avg["accuracy"] == (5, 0.45)
