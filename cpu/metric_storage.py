from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np


class _SmoothedValue:
    """The class tracks a series of values and provides access to the smoothed
    value (i.e., median) over a window or the global series average. The returned
    value is (iteration, value) pair.

    Example::

        >>> smoothed_value = _SmoothedValue()
        >>> smoothed_value.update(value=0.1, iter=0)
        >>> smoothed_value.update(value=0.2, iter=1)
        >>> smoothed_value.median
        (1, 0.15)
    """

    def __init__(self, window_size: int = 20) -> None:
        self._history = deque(maxlen=window_size)  # (iteration, value) pairs
        self._count: int = 0
        self._global_avg: float = 0.0

    def update(self, value: float, iter: Optional[int] = None) -> None:
        """Add a new scalar value produced at a certain iteration. If the length of queue
        exceeds ``window_size``, the oldest element will be removed from the queue.
        """
        if iter is None:
            iter = self._count
        self._history.append((iter, value))
        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    @property
    def latest(self) -> Tuple[int, float]:
        """Return (the latest iteration, the latest value) pair."""
        return self._history[-1]

    @property
    def median(self) -> Tuple[int, float]:
        """Return (the latest iteration, the median of the latest ``window_size`` values) pair."""
        return (self.latest[0], np.median([x[1] for x in self._history]))

    @property
    def global_avg(self) -> Tuple[int, float]:
        """Return (the latest iteration, the global average) pair."""
        return (self.latest[0], self._global_avg)


class MetricStorage:
    """The class store the values of multiple metrics (some of them may be noisy, e.g., loss,
    accuracy) in training process, and provides access to the smoothed values for better logging.

    Example::

        >>> metric_storage = MetricStorage()
        >>> metric_storage.update(iter=0, loss=0.2, accuracy=0.1, smooth=True)
        >>> metric_storage.update(iter=1, loss=0.1, accuracy=0.3, smooth=True)
        >>> metric_storage.values
        {'loss': (1, 0.15), 'accuracy': (1, 0.2)}
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._history: Dict[str, _SmoothedValue] = {}
        self._smooth: Dict[str, bool] = {}

    def clear(self) -> None:
        self._history.clear()
        self._smooth.clear()

    def update(self, iter: Optional[int] = None, smooth: bool = True, **kwargs) -> None:
        """Add new scalar values of multiple metrics produced at a certain iteration.

        Args:
            iter (Optional[int]): The iteration in which these values are produced.
                If None, use the built-in counter starting from 0.
            smooth (Optional[bool]): If True, return the smoothed values for these metrics when
                calling :meth:`values_maybe_smooth`. Otherwise, return the latest values.
                The same metric must have the same `smooth` in different calls to :meth:`update`.
        """
        for key, value in kwargs.items():
            if key in self._smooth:
                assert self._smooth[key] == smooth
            else:
                self._smooth[key] = smooth
            if key not in self._history:
                self._history[key] = _SmoothedValue(window_size=self._window_size)
            self._history[key].update(value, iter)

    @property
    def global_avg(self) -> Dict[str, Tuple[int, float]]:
        """Return (the latest iteration, the global average) pair of multiple metrics."""
        return {key: smoothed_value.global_avg for key, smoothed_value in self._history.items()}

    @property
    def values_maybe_smooth(self) -> Dict[str, Tuple[int, float]]:
        """Return the smoothed values or the latest values of multiple metrics. The specific
        behavior depends on the ``smooth`` when updating metrics. See docs above.

        Returns:
            Dict[str, Tuple[int, float]]: Mapping from metric name to its (iteration, value) pair.
        """
        return {
            key: smoothed_value.median if self._smooth[key] else smoothed_value.latest
            for key, smoothed_value in self._history.items()
        }
