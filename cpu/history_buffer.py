from collections import deque

import numpy as np


class HistoryBuffer:
    """The class tracks a series of values and provides access to the smoothed
    value over a window or the global average / sum of the sequence.

    Args:
        window_size (int): The maximal number of values that can
            be stored in the buffer. Defaults to 20.

    Example::

        >>> his_buf = HistoryBuffer()
        >>> his_buf.update(0.1)
        >>> his_buf.update(0.2)
        >>> his_buf.avg
        0.15
    """

    def __init__(self, window_size: int = 20) -> None:
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
        """The latest value of the queue."""
        return self._history[-1]

    @property
    def avg(self) -> float:
        """The average over the window."""
        return np.mean(self._history)

    @property
    def global_avg(self) -> float:
        """The global average of the queue."""
        return self._sum / self._count

    @property
    def global_sum(self) -> float:
        """The global sum of the queue."""
        return self._sum
