"""Run slow work (takes, resampling, analysis) off the GUI thread."""

from __future__ import annotations

import traceback
from collections.abc import Callable

from PySide6.QtCore import QThread, Signal

_live: set["Worker"] = set()  # keep running workers referenced until they finish


class Worker(QThread):
    """Calls ``fn(*args, **kwargs)`` in a thread; emits ``done(result)`` or ``failed(tb)``."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, *args, **kwargs) -> None:
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs
        self.finished.connect(lambda: _live.discard(self))

    def run(self) -> None:
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception:  # noqa: BLE001 — shown to the user, never swallowed
            self.failed.emit(traceback.format_exc())


def run_async(
    fn: Callable, *args, on_done: Callable | None = None, on_fail: Callable | None = None, **kwargs
) -> Worker:
    """Start a worker and wire its callbacks."""
    w = Worker(fn, *args, **kwargs)
    if on_done:
        w.done.connect(on_done)
    if on_fail:
        w.failed.connect(on_fail)
    _live.add(w)
    w.start()
    return w
