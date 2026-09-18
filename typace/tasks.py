"""Application-owned executors for state, frame, and planning work."""

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import ParamSpec, TypeVar

from typace.config.simulation import MAX_COMPUTE_WORKERS

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")

_PLANNING_WORKERS = 1


class ThreadTaskManager:
    """Own background executors and preserve deterministic result ordering."""

    def __init__(self, *, compute_workers: int = MAX_COMPUTE_WORKERS) -> None:
        if compute_workers <= 0:
            raise ValueError("compute_workers must be positive")
        self._serial = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="typace-state"
        )
        self._compute = ThreadPoolExecutor(
            max_workers=compute_workers, thread_name_prefix="typace-compute"
        )
        self._planning = ProcessPoolExecutor(max_workers=_PLANNING_WORKERS)
        self._closed = False

    def submit_serial(
        self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """Run one stateful operation after previously submitted operations."""
        if self._closed:
            raise RuntimeError("task manager is closed")
        return self._serial.submit(function, *args, **kwargs)

    def map_compute(
        self, function: Callable[[T], R], values: Iterable[T]
    ) -> tuple[R, ...]:
        """Run independent work concurrently and preserve input ordering."""
        if self._closed:
            raise RuntimeError("task manager is closed")
        futures = tuple(self._compute.submit(function, value) for value in values)
        return tuple(future.result() for future in futures)

    def submit_planning(
        self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """Run long planning work without blocking frame computation workers."""
        if self._closed:
            raise RuntimeError("task manager is closed")
        return self._planning.submit(function, *args, **kwargs)

    def shutdown(self, *, wait: bool = True) -> None:
        """Reject new work and release every managed worker."""
        if self._closed:
            return
        self._closed = True
        self._serial.shutdown(wait=wait, cancel_futures=True)
        self._compute.shutdown(wait=wait, cancel_futures=True)
        self._planning.shutdown(wait=wait, cancel_futures=True)
