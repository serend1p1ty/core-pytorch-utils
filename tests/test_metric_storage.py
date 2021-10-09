import math

from cpu.metric_storage import MetricStorage, _SmoothedValue


def test_smoothed_value():
    smoothed_value = _SmoothedValue(window_size=4)
    smoothed_value.update(0.1, 0)
    smoothed_value.update(0.2, 2)
    smoothed_value.update(0.5, 4)
    smoothed_value.update(0.8, 8)
    assert smoothed_value.avg == 0.4
    assert smoothed_value.median == 0.35
    assert smoothed_value._global_avg == 0.4
    smoothed_value.update(1.0, 9)
    assert smoothed_value.avg == 0.625
    assert smoothed_value.median == 0.65
    assert smoothed_value._global_avg == 0.52


def test_metric_storage():
    metric_storage = MetricStorage()
    for i in range(25):
        metric_storage.update(i, loss=1 - 0.1 * i, accuracy=0.1 * i)
    avg = metric_storage.avg
    assert math.isclose(avg["loss"][0], -0.45)
    assert avg["loss"][1] == 24
    assert avg["accuracy"] == (1.45, 24)
