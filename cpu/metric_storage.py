import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple


class SmoothedValue:
    """The class tracks a series of values and provides access to the smoothed
    value over a window or the global average / sum of the series.

    Example::

        >>> smoothed_value = SmoothedValue()
        >>> smoothed_value.update(0.1)
        >>> smoothed_value.update(0.2)
        >>> smoothed_value.avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        """
        Args:
            window_size (int): The maximal number of values that can
                be stored in the buffer. Defaults to 20.
        """
        self._history = deque(maxlen=window_size)
        self._count: int = 0
        self._sum: float = 0.0

    def update(self, value: float) -> None:
        """Add a new scalar value. If the length of queue exceeds ``window_size``,
        the oldest element will be removed from the queue.
        """
        self._history.append(value)
        self._count += 1
        self._sum += value

    @property
    def latest(self) -> float:
        return self._history[-1]

    @property
    def avg(self) -> float:
        return np.mean(self._history)

    @property
    def global_avg(self) -> float:
        return self._sum / self._count

    @property
    def global_sum(self) -> float:
        return self._sum


class MetricStorage(dict):
    """The class stores the values of multiple metrics (some of them may be noisy, e.g., loss,
    batch time) in training process, and provides access to the smoothed values for better logging.

    Example::

        >>> metric_storage = MetricStorage()
        >>> metric_storage.update(iter=0, loss=0.2)
        >>> metric_storage.update(iter=0, lr=0.01, smooth=False)
        >>> metric_storage.update(iter=1, loss=0.1)
        >>> metric_storage.update(iter=1, lr=0.001, smooth=False)
        >>> # loss will be smoothed, but lr will not
        >>> metric_storage.values_maybe_smooth
        {"loss": (1, 0.15), "lr": (1, 0.001)}
        >>> # like dict, can be indexed by string
        >>> metric_storage["loss"].avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._history: Dict[str, SmoothedValue] = self
        self._smooth: Dict[str, bool] = {}
        self._latest_iter: Dict[str, int] = {}

    def update(self, iter: Optional[int] = None, smooth: bool = True, **kwargs) -> None:
        """Add new scalar values of multiple metrics produced at a certain iteration.

        Args:
            iter (int): The iteration in which these values are produced.
                If None, use the built-in counter starting from 0.
            smooth (bool): If True, return the smoothed values of these metrics when
                calling :meth:`values_maybe_smooth`. Otherwise, return the latest values.
                The same metric must have the same ``smooth`` in different calls to :meth:`update`.
        """
        for key, value in kwargs.items():
            if key in self._smooth:
                assert self._smooth[key] == smooth
            else:
                self._smooth[key] = smooth
                self._history[key] = SmoothedValue(window_size=self._window_size)
                self._latest_iter[key] = -1
            if iter is not None:
                assert iter > self._latest_iter[key]
                self._latest_iter[key] = iter
            else:
                self._latest_iter[key] += 1
            self._history[key].update(value)

    @property
    def values_maybe_smooth(self) -> Dict[str, Tuple[int, float]]:
        """Return the smoothed values or the latest values of multiple metrics.
        The specific behavior depends on the ``smooth`` when updating metrics.

        With this function, we can determine whether these metrics should be
        smoothed when performing automatic tensorboard logging.

        Returns:
            dict[str -> (int, float)]: Mapping from metric name to its
                (the latest iteration, the avg/latest value) pair.
        """
        return {
            key: (self._latest_iter[key], smoothed_value.avg if self._smooth[key] else smoothed_value.latest)
            for key, smoothed_value in self._history.items()
        }
