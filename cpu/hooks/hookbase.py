import cpu


class HookBase:
    """Base class for hooks.

    Hooks can be registered in :class:`cpu.trainer.Trainer`. Each hook can implement 6 methods
    (:meth:`before_train`, :meth:`after_train`, :meth:`before_epoch`, :meth:`after_epoch`,
    :meth:`before_iter`, :meth:`after_iter`). The way they are called is demonstrated
    in the following snippet:

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

    In the hook method, users can access ``self.trainer`` to access more
    properties about the context (e.g., model, optimizer, current epoch).

    Each hook has a priority, which is an integer from 1 to 10.
    The smaller the number, the higher the priority. Hooks are executed
    in order of priority from high to low. If two hooks have the same priority,
    they are executed in the order they are registered.
    """

    # A weak reference to the trainer object. Set by the trainer when the hook is registered.
    trainer: "cpu.Trainer" = None
    priority: int = 5

    def before_train(self) -> None:
        """Called before the first epoch."""
        pass

    def after_train(self) -> None:
        """Called after the last epoch."""
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
    def checkpointable(self) -> bool:
        """A hook is checkpointable when it implements :meth:`state_dict` method.
        Its state will be saved into checkpoint.
        """
        return callable(getattr(self, "state_dict", None))

    @property
    def class_name(self) -> str:
        """The class name of the hook."""
        return self.__class__.__name__

    @property
    def metric_storage(self) -> "cpu.trainer.MetricStorage":
        return self.trainer.metric_storage

    def log(self, *args, **kwargs) -> None:
        self.trainer.log(*args, **kwargs)

    # belows are some helper functions that are often used in hook
    def every_n_epochs(self, n: int) -> bool:
        return (self.trainer.cur_epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, n: int) -> bool:
        return (self.trainer.cur_iter + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, n: int) -> bool:
        return (self.trainer.inner_iter + 1) % n == 0 if n > 0 else False

    def is_last_epoch(self) -> bool:
        return self.trainer.cur_epoch == self.trainer.max_epochs - 1

    def is_last_iter(self) -> bool:
        return self.trainer.cur_iter == self.trainer.max_iters - 1

    def is_last_inner_iter(self) -> bool:
        return self.trainer.inner_iter == self.trainer.epoch_len - 1
