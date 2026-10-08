"""Background prefetching of upcoming frames' data slices.

The dominant cost when rendering an animation over lazily-loaded data (e.g. an
Imaris/HDF5 file surfaced as dask arrays) is reading each displayed slice from
disk -- this shows up as the ``apply`` phase in the performance log. The
:class:`SlicePrefetcher` reads the slices that *upcoming* frames will display on
background threads, so that by the time the main thread applies a frame the read
hits the OS/dask cache instead of disk, and several reads overlap.

Prefetching only *warms caches*; it never affects what is rendered, so any error
encountered while prefetching is swallowed. Reads go through ``np.asarray`` on
the layer data, which for dask-backed sources uses dask's scheduler (handling
locking for thread-safe backends such as h5py). If a data source is not safe for
concurrent reads, prefetching can be disabled with ``prefetch=0``.
Fully materialized volumes already displayed in 3D are excluded: there is no
upcoming slice to warm, and re-reading their Dask/HDF5 chunks wastes I/O.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import numpy as np

from .layer_registry import OBJECT_ID_METADATA, SOURCE_ID_METADATA

logger = logging.getLogger("napari_animation")

#: Cap for the "auto" dask cache size.
_AUTO_CACHE_CAP = 2 * 1024**3  # 2 GiB
#: Fraction of available RAM to use for the "auto" dask cache.
_AUTO_CACHE_FRACTION = 0.25
_INTERACTIVE_CACHE_LOCK = Lock()
_INTERACTIVE_CACHE = None
_INTERACTIVE_CACHE_USERS = 0


def acquire_interactive_cache():
    """Keep decompressed Dask chunks warm for a GUI viewer's lifetime.

    One global cache is shared by all Animation docks in a process.  The
    default 512 MiB budget can be changed with
    ``NAPARI_ANIMATION_INTERACTIVE_CACHE_MB``; zero disables it.
    """
    global _INTERACTIVE_CACHE, _INTERACTIVE_CACHE_USERS
    try:
        size_mb = max(0, int(os.environ.get(
            "NAPARI_ANIMATION_INTERACTIVE_CACHE_MB", "512"
        )))
    except ValueError:
        size_mb = 512
    if size_mb == 0:
        return False
    with _INTERACTIVE_CACHE_LOCK:
        if _INTERACTIVE_CACHE is None:
            try:
                from ._dask_cache import ConcurrentCache as Cache

                cache = Cache(size_mb * 1024**2)
                cache.register()
                _INTERACTIVE_CACHE = cache
            except Exception as error:
                logger.warning("Interactive Dask cache unavailable: %s", error)
                return False
        _INTERACTIVE_CACHE_USERS += 1
    return True


def release_interactive_cache():
    """Release a GUI viewer's share of the persistent Dask cache."""
    global _INTERACTIVE_CACHE, _INTERACTIVE_CACHE_USERS
    with _INTERACTIVE_CACHE_LOCK:
        if _INTERACTIVE_CACHE_USERS == 0:
            return
        _INTERACTIVE_CACHE_USERS -= 1
        if _INTERACTIVE_CACHE_USERS == 0 and _INTERACTIVE_CACHE is not None:
            _INTERACTIVE_CACHE.unregister()
            _INTERACTIVE_CACHE = None


def resolve_cache_bytes(spec) -> int:
    """Resolve a dask-cache size spec to a byte count (0 disables it).

    ``spec`` may be ``None``/``False`` (disabled), ``"auto"`` (a fraction of
    available RAM, capped), or an integer number of bytes.
    """
    if not spec:
        return 0
    if spec == "auto":
        try:
            import psutil

            available = psutil.virtual_memory().available
        except Exception:  # noqa: BLE001 - psutil optional/!available
            available = 4 * 1024**3
        return int(min(_AUTO_CACHE_CAP, _AUTO_CACHE_FRACTION * available))
    return int(spec)


@contextlib.contextmanager
def dask_cache_context(spec):
    """Register a dask opportunistic cache for the duration of the block.

    The cache lets prefetched (decompressed) chunks be reused by the main
    thread's slice instead of being recomputed, and makes frames that share a
    chunk free. Yields the cache size in bytes (0 if disabled/unavailable).
    """
    nbytes = resolve_cache_bytes(spec)
    if nbytes <= 0:
        yield 0
        return
    with _INTERACTIVE_CACHE_LOCK:
        shared = _INTERACTIVE_CACHE
    if shared is not None:
        # ``total_bytes`` is occupancy, not the configured cache budget.
        yield int(shared.cache.available_bytes)
        return
    try:
        from ._dask_cache import ConcurrentCache as Cache
    except Exception as err:  # noqa: BLE001 - cachey may be missing
        logger.warning(
            "dask cache unavailable (%s); continuing without it", err
        )
        yield 0
        return
    with Cache(nbytes):
        yield nbytes


def _layer_level_array(layer):
    """Return the array a layer would read from (handles multiscale)."""
    data = getattr(layer, "data", None)
    if data is None:
        return None
    if getattr(layer, "multiscale", False) and isinstance(data, Sequence):
        if not data:
            return None
        level = getattr(layer, "data_level", 0)
        level = min(max(0, int(level)), len(data) - 1)
        return data[level]
    return data


def _axis_index(point, ranges, axis, size):
    """Map a world position along ``axis`` to an index into an array of size."""
    axis_range = ranges[axis]
    start, stop = float(axis_range[0]), float(axis_range[1])
    pt = float(point[axis])
    frac = 0.0 if stop == start else (pt - start) / (stop - start)
    idx = int(round(frac * (size - 1)))
    return max(0, min(size - 1, idx))


def _resident_full_view(layer, array):
    """Whether napari already holds this entire displayed array in memory."""
    if not getattr(layer, "loaded", False):
        return False
    image = getattr(getattr(layer, "_slice", None), "image", None)
    view = getattr(image, "view", None)
    return isinstance(view, np.ndarray) and view.shape == array.shape


def _all_sources_resident(viewer, frames):
    """Prove every frame uses already materialized full-volume sources."""
    if not frames:
        return False
    from .frame_sequence import FrameSequence, resolve_interpolation
    from .interpolation import Interpolation

    sources = [
        layer for layer in viewer.layers
        if not getattr(layer, "metadata", {}).get(OBJECT_ID_METADATA)
    ]
    if not sources:
        return False
    for layer in sources:
        array = _layer_level_array(layer)
        if array is None or not _resident_full_view(layer, array):
            return False
        if getattr(layer, "multiscale", False) and getattr(
            layer, "locked_data_level", None
        ) is None:
            return False
        source_id = getattr(layer, "metadata", {}).get(SOURCE_ID_METADATA)
        if source_id is None:
            return False

    # Only inspect saved keyframes when the effective rules prove that the
    # relevant fields cannot change between them. Unknown/custom sequences
    # retain normal prefetching rather than being evaluated eagerly here.
    if isinstance(frames, FrameSequence):
        paths = ["dims.ndisplay"] + [
            f"layers.{layer.metadata[SOURCE_ID_METADATA]}.locked_data_level"
            for layer in sources
        ]
        if not all(
            resolve_interpolation(frames.state_interpolation_map, path)
            is Interpolation.STEP_END for path in paths
        ):
            return False
        states = tuple(keyframe.viewer_state for keyframe in frames._key_frames)
    elif type(frames) in (list, tuple):
        states = frames  # Already materialized; do not copy or expand them.
    else:
        return False

    for layer in sources:
        source_id = layer.metadata[SOURCE_ID_METADATA]
        for frame in states:
            ndisplay = frame.dims.get("ndisplay")
            if ndisplay is None or int(ndisplay) < layer.ndim:
                return False
            layer_states = getattr(frame, "layers", None)
            state = layer_states.get(source_id) if layer_states is not None else None
            if state is None or state.get(
                "locked_data_level", getattr(layer, "locked_data_level", None)
            ) != getattr(layer, "locked_data_level", None):
                return False
    return True


def read_keys(viewer, dims):
    """Yield ``(array, index)`` pairs to materialize each layer's slice.

    ``index`` is a tuple with ``slice(None)`` for displayed axes and an integer
    for sliced axes, computed from the (interpolated) ``dims`` of a frame.
    """
    ndim = dims["ndim"]
    ndisplay = dims["ndisplay"]
    order = list(dims["order"])
    point = dims["point"]
    ranges = dims["range"]
    displayed = set(order[-ndisplay:])

    keys = []
    for layer in viewer.layers:
        try:
            if getattr(layer, "metadata", {}).get(OBJECT_ID_METADATA):
                # A managed section reads from its source image. Warming both
                # requests the same data twice and can prefetch stale output.
                continue
            if not getattr(layer, "visible", True):
                continue
            arr = _layer_level_array(layer)
            if arr is None:
                continue
            layer_ndim = getattr(layer, "ndim", arr.ndim)
            offset = ndim - layer_ndim
            if (
                all(axis + offset in displayed for axis in range(layer_ndim))
                and _resident_full_view(layer, arr)
            ):
                # A static 3D volume in a 3D view has already been read in
                # full for napari. Re-reading every HDF5/Dask chunk cannot
                # warm a smaller future slice and only consumes I/O and RAM.
                continue
            data_point = None
            if callable(getattr(layer, "world_to_data", None)):
                world_point = np.asarray(point[offset:offset + layer_ndim], float)
                data_point = np.asarray(layer.world_to_data(world_point), float)
                if getattr(layer, "multiscale", False):
                    factors = np.asarray(
                        layer.downsample_factors[layer.data_level], float
                    )
                    data_point = data_point / factors
            index = []
            for ax in range(layer_ndim):
                global_ax = ax + offset
                if global_ax in displayed:
                    index.append(slice(None))
                else:
                    index.append(
                        int(np.clip(round(data_point[ax]), 0, arr.shape[ax] - 1))
                        if data_point is not None else
                        _axis_index(point, ranges, global_ax, arr.shape[ax])
                    )
            keys.append((arr, tuple(index)))
        except Exception:  # noqa: BLE001 - best effort per layer
            continue
    return keys


def _read_signature(array, index):
    """Identify the Dask chunks warmed by a slice, or an exact native read.

    Adjacent scroll frames often select different Z planes from the same Dask
    block.  A registered Dask cache needs only one producer for that block;
    concurrent prefetches otherwise repeat the same HDF5 read/decompression.
    Native array-like sources use exact indices because their caches may not
    retain the whole chunk after a thin-slice read.
    """
    try:
        import dask.array as da

        is_dask = isinstance(array, da.Array)
    except ImportError:
        is_dask = False
    if not is_dask:
        return id(array), tuple(
            ("slice", part.start, part.stop, part.step)
            if isinstance(part, slice) else int(part)
            for part in index
        )
    coordinates = []
    for parts, selected in zip(array.chunks, index):
        if isinstance(selected, slice):
            if selected != slice(None):
                # read_keys currently uses full slices; retain exact identity
                # if a future caller supplies a partial or stepped slice.
                coordinates.append((
                    "slice", selected.start, selected.stop, selected.step
                ))
            else:
                coordinates.append("all")
        else:
            chunk = int(np.searchsorted(
                np.cumsum(parts), int(selected), side="right"
            ))
            coordinates.append(chunk)
    return id(array), tuple(coordinates)


class SlicePrefetcher:
    """Warm the data cache for frames ``current+1 .. current+depth`` ahead."""

    def __init__(self, viewer, frames, depth=2, workers=2):
        self.viewer = viewer
        self.frames = frames
        self.depth = max(0, int(depth))
        try:
            resident = _all_sources_resident(viewer, frames) if self.depth else False
        except Exception as error:  # noqa: BLE001 - read-ahead is best effort
            logger.warning("Could not assess resident sources: %s", error)
            resident = False
        self.enabled = self.depth > 0 and not resident
        self.skip_reason = "resident_full_volume" if resident and self.depth else None
        self._submitted = set()
        self._futures = {}
        self._lock = Lock()
        self._pool = (
            ThreadPoolExecutor(
                max_workers=max(1, int(workers)),
                thread_name_prefix="napari-anim-prefetch",
            )
            if self.enabled
            else None
        )

    def advance(self, current_index):
        """Submit prefetch reads for the look-ahead window past ``current``."""
        if not self.enabled:
            return
        wanted = range(
            current_index + 1,
            min(len(self.frames), current_index + self.depth + 1),
        )
        # Freeze viewer/layer state on the GUI thread, then group frames that
        # need precisely the same Dask chunks into one background task.
        frames = {}
        unique = {}
        for index in wanted:
            try:
                keys = read_keys(self.viewer, self.frames[index].dims)
            except Exception:  # noqa: BLE001 - prefetch is best effort
                continue
            signature = tuple(_read_signature(arr, key) for arr, key in keys)
            frames[index] = signature
            if signature:
                unique.setdefault(signature, keys)
        with self._lock:
            for signature, (future, cancel) in list(self._futures.items()):
                if signature not in unique:
                    cancel.set()
                    future.cancel()
                    self._futures.pop(signature, None)
            for signature, keys in unique.items():
                if signature not in self._futures and len(self._futures) < self.depth:
                    cancel = Event()
                    self._futures[signature] = (
                        self._pool.submit(self._warm_keys, keys, cancel), cancel
                    )
            self._submitted = {
                index for index, signature in frames.items()
                if not signature or signature in self._futures
            }

    def _warm(self, dims):
        self._warm_keys(read_keys(self.viewer, dims))

    @staticmethod
    def _warm_keys(keys, cancel=None):
        seen = set()
        for arr, index in keys:
            if cancel is not None and cancel.is_set():
                return
            signature = _read_signature(arr, index)
            if signature in seen:
                continue
            seen.add(signature)
            try:
                np.asarray(arr[index])
            except Exception as err:  # noqa: BLE001 - best effort per source
                logger.warning("Prefetch read failed: %s", err)

    def shutdown(self):
        if self._pool is not None:
            with self._lock:
                for future, cancel in self._futures.values():
                    cancel.set()
                    future.cancel()
                self._futures.clear()
                self._submitted.clear()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
