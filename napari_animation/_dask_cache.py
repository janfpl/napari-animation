"""Dask caching for overlapping GUI, prefetch, and projection computations.

Dask's Cache keeps task timings on the callback instance. One scheduler's
finish clears timings still needed by another scheduler, causing KeyError on
keys such as finalize-hlgfinalizecompute-*. Keep those timings in each
scheduler's own state instead, and protect only shared cache bookkeeping.
"""

import sys
from threading import RLock
from timeit import default_timer

from dask.cache import Cache, overhead
from dask.callbacks import Callback


class ConcurrentCache(Cache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_concurrency()

    def _initialize_concurrency(self):
        self._cache_lock = RLock()
        self._context_users = 0
        self._registered = False
        self._enabled = True
        self._registration_sets = []

    def _start(self, dsk):
        with self._cache_lock:
            if self._enabled:
                for key in set(dsk).intersection(self.cache.data):
                    dsk[key] = self.cache.get(key)

    def _start_state(self, dsk, state):
        state[self] = ({}, {})

    def _pretask(self, key, dsk, state):
        state[self][0][key] = default_timer()

    def _posttask(self, key, value, dsk, state, worker_id):
        starts, durations = state[self]
        duration = default_timer() - starts.pop(key)
        dependencies = state["dependencies"][key]
        if dependencies:
            duration += max(durations.get(dependency, 0) for dependency in dependencies)
        durations[key] = duration
        nbytes = self._nbytes(value) + overhead + sys.getsizeof(key) * 4
        with self._cache_lock:
            if self._enabled and key not in self.cache.data:
                # Cachey counts a second put for the same key twice. Another
                # scheduler may have cached this identical chunk meanwhile.
                self.cache.put(key, value, cost=duration / nbytes / 1e9, nbytes=nbytes)

    def _finish(self, dsk, state, errored):
        if state is not None:
            state.pop(self, None)

    def register(self):
        with self._cache_lock:
            self._registered = self._enabled = True
            self._activate_callback()

    def _activate_callback(self):
        callbacks = Callback.active
        if not any(callbacks is previous for previous in self._registration_sets):
            self._registration_sets.append(callbacks)
        callbacks.add(self._callback)

    def unregister(self):
        with self._cache_lock:
            self._registered = False
            self._release_if_unused()

    def _release_if_unused(self):
        if not self._registered and self._context_users == 0:
            self._enabled = False
            # A running Dask scheduler may have temporarily removed global
            # callbacks. Remove from its saved set too, or its finally block
            # would resurrect the retired cache after export/dock cleanup.
            for callbacks in self._registration_sets:
                callbacks.discard(self._callback)
            self._registration_sets.clear()
            Callback.active.discard(self._callback)

    def __enter__(self):
        with self._cache_lock:
            self._context_users += 1
            self._enabled = True
            self._activate_callback()
        return self

    def __exit__(self, *args):
        with self._cache_lock:
            self._context_users -= 1
            self._release_if_unused()


def protect_napari_cache():
    """Upgrade the existing napari cache, including already captured references.

    configure_dask closures retain the cache instance. Replacing the module's
    variable alone leaves existing layers using the unsafe instance. Upgrade
    only the known Dask Cache type, preserving its contents and configured
    budget. No installed dependency files or global Cache class are modified.
    """
    from napari.utils import _dask_utils

    cache = getattr(_dask_utils, "_DASK_CACHE", None)
    if type(cache) is Cache:
        old_callback = cache._callback
        registered = old_callback in Callback.active
        ConcurrentCache._initialize_concurrency(cache)
        cache.__class__ = ConcurrentCache
        if registered:
            Callback.active.discard(old_callback)
            cache.register()
    return cache
