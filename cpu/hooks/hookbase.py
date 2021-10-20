from ..metric_storage import MetricStorage


class HookBase:
    """Base class for hooks.

    Hooks can be registered in :class:`Trainer`. Each hook can implement 6 methods
    (:meth:`before_train`, :meth:`after_train`, :meth:`before_epoch`, :meth:`after_epoch`,
    :meth:`before_iter`, :meth:`after_iter`).

    The way they are called is demonstrated in the following snippet:

    .. code-block:: python

        hook.before_train()
        for epoch in range(start_epoch, max_epochs):
            hook.before_epoch()
            for iter in range(epoch_len):
                hook.before_iter()
                train_one_iter()
                hook.after_iter()
            hook.after_epoch()
        hook.after_train()

    .. Note::

        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config).
        2. A hook that does something in :meth:`before_iter` can often be implemented
           equivalently in :meth:`after_iter`. If the hook takes non-trivial time, it
           is strongly recommended to implement the hook in :meth:`after_iter` instead
           of :meth:`before_iter`. The convention is that :meth:`before_iter` should only
           take negligible time. Following this convention will allow hooks that do care about
           the difference between :meth:`before_iter` and :meth:`after_iter` (e.g., timer) to
           function properly.
    """

    # A weak reference to the trainer object. Set by the trainer when the hook is registered.
    trainer = None

    def before_train(self) -> None:
        """Called before the first iteration."""
        pass

    def after_train(self) -> None:
        """Called after the last iteration."""
        pass

    def before_epoch(self) -> None:
        """Called before each epoch."""
        pass

    def after_epoch(self) -> None:
        """Called after each epoch."""
        pass

    def before_iter(self) -> None:
        """Called before each iteration."""
        pass

    def after_iter(self) -> None:
        """Called after each iteration."""
        pass

    @property
    def storage(self) -> MetricStorage:
        """The abbreviation of ``self.trainer.metric_storage``."""
        return self.trainer.metric_storage

    @property
    def checkpointable(self) -> bool:
        """If a hook has :meth:`state_dict` method, it is checkpointable.
        Its state will be saved into checkpoint.
        """
        return callable(getattr(self, "state_dict", None))

    @property
    def class_name(self) -> bool:
        """Return the class name of the hook."""
        return self.__class__.__name__

    # belows are helper functions that are often used in hook
    def every_n_epochs(self, n: int) -> bool:
        return (self.trainer.epoch + 1) % n == 0

    def every_n_iters(self, n: int) -> bool:
        return (self.trainer.iter + 1) % n == 0

    def every_n_inner_iters(self, n: int) -> bool:
        return (self.trainer.inner_iter + 1) % n == 0

    def is_last_epoch(self) -> bool:
        return self.trainer.epoch == self.trainer.max_epochs - 1

    def is_last_iter(self) -> bool:
        return self.trainer.iter == self.trainer.max_iters - 1

    def is_last_inner_iter(self) -> bool:
        return self.trainer.inner_iter == self.trainer.epoch_len - 1
