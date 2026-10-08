"""Bounded, replaceable slab reads for interactive 2D sections.

The worker receives only an array and immutable indices.  All napari layer and
Qt operations remain on the caller's GUI thread.  Headless ViewerModel callers
use the same computation synchronously, which also makes export deterministic.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from .._projection_core import project_slab_core

logger = logging.getLogger("napari_animation")
_TARGET_READ_BYTES = 64 * 1024**2
_CACHE_BYTES = 128 * 1024**2


class CancelledProjection(Exception):
    """A newer interaction superseded this bounded read."""


def project_slab(volume, axis, start, stop, projection, cancel=None):
    """Reduce a slab using bounded, chunk-aligned basic ROI reads.

    A thick XZ/YZ slab can have a small 2D result but a huge 3D input. Tiling
    its two output dimensions retains exact voxels while avoiding repeated
    decompression of the same HDF5 chunks across projection-axis subreads.
    """
    return project_slab_core(
        volume, axis, start, stop, projection, cancel,
        target_read_bytes=_TARGET_READ_BYTES,
        cancel_error=CancelledProjection,
    )


class ProjectionCache:
    """Byte-bounded LRU of finished 2D projections."""

    def __init__(self, limit=_CACHE_BYTES):
        self.limit = int(limit)
        self._items = OrderedDict()
        self.bytes = 0

    def get(self, key):
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key, value):
        if value.nbytes > self.limit:
            return
        old = self._items.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes
        self._items[key] = value
        self.bytes += value.nbytes
        while self.bytes > self.limit:
            _, evicted = self._items.popitem(last=False)
            self.bytes -= evicted.nbytes

    def invalidate_source(self, source_id):
        for key in list(self._items):
            if key[0] == source_id:
                self.bytes -= self._items.pop(key).nbytes


class ProjectionService:
    """One active read and at most one pending read per viewer.

    The newest request wins during rapid scrubbing.  A completed result is
    published only if its section generation still matches.  The GUI Qt
    signal is created lazily because headless benchmarks have no QApplication.
    """

    def __init__(self, asynchronous=False, trace=None):
        self.cache = ProjectionCache()
        self.trace = trace
        self.asynchronous = bool(asynchronous)
        self._generations = {}
        self._section_revisions = {}
        self._active = None
        self._pending = None
        self._batch_pending = []
        self._batch_mode = False
        self._pool = None
        self._bridge = None
        self._idle_callbacks = []
        self.last_error = None
        self._process_projector = None
        self._process_source = None
        self._process_failed = False
        self._closed = False
        if self.asynchronous:
            from qtpy.QtCore import QObject, Signal, Slot

            class _Bridge(QObject):
                finished = Signal(object)

                @Slot(object)
                def deliver(self, payload):
                    self.owner._complete(payload)

            self._bridge = _Bridge()
            self._bridge.owner = self
            self._bridge.finished.connect(self._bridge.deliver)
            self._pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="napari-section-read"
            )

    def invalidate_source(self, source_id):
        self.cache.invalidate_source(source_id)
        if self._active is not None and self._active[0][0] == source_id:
            self._active[3].set()
        if self._pending is not None and self._pending[0][0] == source_id:
            self._pending = None
        self._batch_pending = [
            request for request in self._batch_pending
            if request[0][0] != source_id
        ]

    def cancel_section(self, section_id):
        self.begin_section(section_id)
        self._generations[section_id] = self._generations.get(section_id, 0) + 1
        if self._active is not None and self._active[1] == section_id:
            self._active[3].set()
        if self._pending is not None and self._pending[1] == section_id:
            self._pending = None
        self._batch_pending = [
            request for request in self._batch_pending
            if request[1] != section_id
        ]

    def begin_batch(self):
        """Start one composite revision without cancelling sibling reads."""
        self._batch_pending.clear()
        if self._active is not None:
            self._active[3].set()
        self._batch_mode = True

    def end_batch(self):
        self._batch_mode = False

    def begin_section(self, section_id):
        """Invalidate callbacks from the previous application of a section."""
        revision = self._section_revisions.get(section_id, 0) + 1
        self._section_revisions[section_id] = revision
        return revision

    def section_is_current(self, section_id, revision):
        return (
            not self._closed
            and self._section_revisions.get(section_id) == revision
        )

    def retain_sections(self, section_ids):
        """Retire work for sections absent from the current rendered view."""
        for section_id in set(self._section_revisions) | set(self._generations):
            if section_id not in section_ids:
                self.cancel_section(section_id)

    def request(
        self, section_id, key, volume, axis, start, stop, projection, publish,
        on_error=None,
    ):
        if self._closed:
            return
        self.last_error = None
        generation = self._generations.get(section_id, 0) + 1
        self._generations[section_id] = generation
        cached = self.cache.get(key)
        if cached is not None:
            publish(cached)
            return
        if not self.asynchronous:
            result = project_slab(volume, axis, start, stop, projection)
            self.cache.put(key, result)
            publish(result)
            return
        if (
            self._active is not None and self._active[0] == key
            and self._active[1] == section_id
            and not self._active[3].is_set()
        ):
            # Same slab requested again while its read is in flight: keep
            # one producer and publish it through the newest callback.
            self._active[2] = generation
            self._active[9] = publish
            self._active[10] = on_error
            return
        if (
            self._pending is not None and self._pending[0] == key
            and self._pending[1] == section_id
        ):
            self._pending[2] = generation
            self._pending[9] = publish
            self._pending[10] = on_error
            return
        request = [
            key, section_id, generation, threading.Event(), volume, axis,
            start, stop, projection, publish, on_error,
        ]
        if self._active is None:
            self._start(request)
        elif self._batch_mode:
            self._batch_pending.append(request)
        else:
            # h5py cannot interrupt a read in flight, but no older queued
            # interaction can build an unbounded backlog.
            self._active[3].set()
            self._pending = request

    def _start(self, request):
        self._active = request

        def compute():
            try:
                if self.trace is None:
                    return self._project(request)
                with self.trace.timer("section_projection"):
                    return self._project(request)
            finally:
                if self._closed and self._process_projector is not None:
                    self._process_projector.shutdown()
                    self._process_projector = None

        future = self._pool.submit(compute)

        def finished(done):
            try:
                payload = (request, done.result(), None)
            except BaseException as error:  # pass to GUI for logging
                payload = (request, None, error)
            self._bridge.finished.emit(payload)

        future.add_done_callback(finished)

    def _project(self, request):
        """Use parallel HDF5 lanes only for an eligible direct source."""
        volume, axis, start, stop, projection, cancel = (
            request[4], request[5], request[6], request[7], request[8],
            request[3],
        )
        from .h5_process import (
            H5ProcessProjector, describe_h5_volume,
            direct_uncompressed_read, process_lanes, project_h5_direct,
        )

        descriptor = describe_h5_volume(volume)
        lanes = (
            process_lanes(descriptor, axis, start, stop)
            if descriptor is not None and not self._process_failed else 0
        )
        if lanes >= 2:
            source = (request[0][:3], descriptor, lanes)
            try:
                if self._process_source != source:
                    if self._process_projector is not None:
                        self._process_projector.shutdown()
                    self._process_projector = H5ProcessProjector(
                        descriptor, lanes
                    )
                    self._process_source = source
                return self._process_projector.project(
                    axis, start, stop, projection, cancel
                )
            except CancelledProjection:
                raise
            except Exception as error:  # spawn/pickle/reader fallback
                logger.warning(
                    "Parallel HDF5 projection unavailable; using one "
                    "reader: %s", error,
                )
                self._process_failed = True
                if self._process_projector is not None:
                    self._process_projector.shutdown()
                    self._process_projector = None
                    self._process_source = None
        else:
            if self._process_projector is not None:
                self._process_projector.shutdown()
                self._process_projector = None
                self._process_source = None
        if descriptor is not None and direct_uncompressed_read(
            volume, descriptor, axis
        ):
            try:
                return project_h5_direct(
                    descriptor, axis, start, stop, projection, cancel
                )
            except CancelledProjection:
                raise
            except Exception as error:
                logger.warning(
                    "Direct HDF5 projection unavailable; using Dask: %s",
                    error,
                )
        return project_slab(volume, axis, start, stop, projection, cancel)

    def _complete(self, payload):
        request, result, error = payload
        if self._active is request:
            self._active = None
        key, section_id, generation = request[:3]
        if error is not None and not isinstance(error, CancelledProjection):
            logger.warning("Section projection failed: %s", error)
            if self._generations.get(section_id) == generation:
                self.last_error = error
                if request[10] is not None:
                    request[10](error)
        if (
            not self._closed and error is None and not request[3].is_set()
            and self._generations.get(section_id) == generation
        ):
            self.cache.put(key, result)
            request[9](result)
        # Publish queued requests that are already cached and start the first
        # one that needs a read. Stopping after a cache hit would strand the
        # rest of the queue, since no read would remain to complete it.
        while self._active is None:
            pending = (
                self._batch_pending.pop(0) if self._batch_pending
                else self._pending
            )
            if pending is None:
                break
            if pending is self._pending:
                self._pending = None
            cached = self.cache.get(pending[0])
            if cached is None:
                self._start(pending)
                break
            if self._generations.get(pending[1]) == pending[2]:
                pending[9](cached)
        if self._active is None and self._pending is None and not self._batch_pending:
            callbacks, self._idle_callbacks = self._idle_callbacks, []
            for callback in callbacks:
                callback()

    def when_idle(self, callback):
        if self._active is None and self._pending is None and not self._batch_pending:
            callback()
        else:
            self._idle_callbacks.append(callback)

    def shutdown(self):
        self._closed = True
        if self._active is not None:
            self._active[3].set()
        self._pending = None
        self._batch_pending.clear()
        self._idle_callbacks.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        if self._active is None and self._process_projector is not None:
            self._process_projector.shutdown()
            self._process_projector = None


def service_for_viewer(viewer):
    service = getattr(viewer, "_napari_animation_projection_service", None)
    if service is None:
        try:
            from qtpy.QtWidgets import QApplication

            asynchronous = QApplication.instance() is not None and hasattr(viewer, "window")
        except ImportError:
            asynchronous = False
        service = ProjectionService(
            asynchronous=asynchronous,
            trace=getattr(viewer, "_napari_animation_trace", None),
        )
        viewer._napari_animation_projection_service = service
    return service


@contextmanager
def exact_render(viewer):
    """Render script screenshots synchronously at the authored resolution.

    Retire interactive work before applying a frame. The background worker
    may finish its read, but cannot publish into this render or its successor.
    Napari's screenshot method waits separately for its native layer slicer.
    """
    service = service_for_viewer(viewer)
    asynchronous = service.asynchronous
    previous = getattr(viewer, "_napari_animation_export_exact", False)
    service.retain_sections(set())
    service.asynchronous = False
    viewer._napari_animation_export_exact = True
    try:
        yield
    finally:
        service.asynchronous = asynchronous
        viewer._napari_animation_export_exact = previous
