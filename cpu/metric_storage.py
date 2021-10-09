from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

import numpy as np


class _SmoothedValue:
    """Metrics (e.g., loss, accuracy) may be noisy. The class tracks
    a series of values and provides access to smoothed values over a
    window or the global series average.

    Example::

        >>> smoothed_value = SmoothedValue()
        >>> smoothed_value.update(0.1, 0)
        >>> smoothed_value.update(0.2, 1)
        >>> smoothed_value.avg()
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
        self._history = deque(maxlen=window_size)  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0.0

    def update(self, value: float, iteration: Optional[int] = None) -> None:
        """Add a new scalar value produced at a certain iteration. If the length of queue
        exceeds ``window_size``, the oldest element will be removed from the queue.
        """
        if iteration is None:
            iteration = self._count
        self._count += 1
        self._history.append((value, iteration))
        self._global_avg += (value - self._global_avg) / self._count

    @property
    def median(self) -> float:
        """Return (median, the lastest iteration) pair."""
        return (np.median([x[0] for x in self._history]), self._history[-1][1])

    @property
    def avg(self) -> float:
        """Return (average, the lastest iteration) pair."""
        return (np.mean([x[0] for x in self._history]), self._history[-1][1])

    @property
    def global_avg(self) -> float:
        """Return (global average, the lastest iteration) pair."""
        return (self._global_avg, self._history[-1][1])

    @property
    def latest(self) -> float:
        """Return (the latest value, the lastest iteration) pair."""
        return self._history[-1]


class MetricStorage:
    """This class provides access to the smoothed values of multiple metrics.

    Example::

        >>> metric_storage = MetricStorage()
        >>> metric_storage.update(0, loss=0.2, accuracy=0.1)
        >>> metric_storage.update(1, loss=0.1, accuracy=0.3)
        >>> metric_storage.avg
        {'loss': (0.15, 1), 'accuracy': (0.2, 1)}
    """

    def __init__(self) -> None:
        self._history = defaultdict(_SmoothedValue)

    def clear(self) -> None:
        self._history.clear()

    def update(self, iteration: Optional[int] = None, **kwargs) -> None:
        """Add new scalar values of multiple metrics produced at a certain iteration."""
        for key, value in kwargs.items():
            self._history[key].update(value, iteration)

    @property
    def median(self) -> Dict[str, Tuple[float, int]]:
        """Return the (median, the lastest iteration) pair of all available metrics.

        Returns:
            Dict[str, Tuple[float, int]]: Mapping from the name of each metric to
                (median, the lastest iteration) pair.
        """
        return {k: smoothed_value.median for k, smoothed_value in self._history.items()}

    @property
    def avg(self) -> Dict[str, Tuple[float, int]]:
        """Similar to :meth:`median` but return (average, the lastest iteration) pair."""
        return {k: smoothed_value.avg for k, smoothed_value in self._history.items()}

    @property
    def global_avg(self) -> Dict[str, Tuple[float, int]]:
        """Similar to :meth:`median` but return (global average, the lastest iteration) pair."""
        return {k: smoothed_value.global_avg for k, smoothed_value in self._history.items()}

    @property
    def latest(self) -> Dict[str, Tuple[float, int]]:
        """Similar to :meth:`median` but return (the lastest value, the lastest iteration) pair."""
        return {k: smoothed_value.latest for k, smoothed_value in self._history.items()}
