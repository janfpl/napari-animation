"""Independent read-only HDF5 lanes for large interactive section projections.

h5py serializes threads around the HDF5 library.  Separate spawned processes
can decompress disjoint stripes concurrently without sharing a file handle.
The parent retains only one in-flight section request; children receive file
names and small indices, never napari layers or Dask graphs.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .._h5_worker import StripeVolume as _StripeVolume  # noqa: F401 (tests)
from .._h5_worker import open_worker as _open_worker
from .._h5_worker import project_stripe as _project_stripe

_MIN_PHYSICAL_BYTES = 64 * 1024**2


def _dataset_descriptor(dataset):
    return (
        os.path.abspath(dataset.file.filename),
        dataset.name,
        tuple(int(size) for size in dataset.shape),
        dataset.dtype.str,
        (
            tuple(int(size) for size in dataset.chunks)
            if dataset.chunks
            else None
        ),
        dataset.compression,
    )


def _plain_getter_layer(layer):
    """Allow only Dask's unchanged getter without expanding all chunk tasks.

    Unknown graph forms conservatively keep their original reader.
    """
    from dask.array.core import getter, getter_nofancy
    from dask.blockwise import Blockwise

    return isinstance(layer, Blockwise) and getattr(
        getattr(layer, "task", None), "func", None
    ) in (getter, getter_nofancy)


def _imaris_reader(volume):
    """Recognize the reference loader's exact singleton T/C selection.

    A graph with extra operations could flip, scale, or otherwise alter voxels;
    it must continue through Dask rather than read the HDF5 dataset directly.
    """
    layers = volume.dask.layers
    originals = [name for name in layers if name.startswith("original-")]
    if len(layers) != 3 or len(originals) != 1 or volume.name not in layers:
        return None
    original = originals[0]
    reader_layers = set(layers) - {original, volume.name}
    if len(reader_layers) != 1 or not _plain_getter_layer(
        layers[next(iter(reader_layers))]
    ):
        return None
    reader = layers[original].get(original)
    for _ in range(2):
        module = type(reader).__module__
        name = type(reader).__name__
        if (module, name) not in (
            ("napari_imaris_loader._cache", "CachingReader"),
            ("napari_imaris_loader._logging", "TimedReader"),
        ):
            break
        reader = reader._reader
    if (
        type(reader).__module__ != "imaris_ims_file_reader.ims"
        or type(reader).__name__ != "ims_reader"
        or reader.squeeze_output is not False
        or reader.write is not False
        or not isinstance(reader.ResolutionLevelLock, int)
        or reader.ResolutionLevelLock < 0
        or tuple(reader.shape[:2]) != (1, 1)
        or tuple(reader.shape[2:]) != tuple(volume.shape)
        or np.dtype(reader.dtype) != volume.dtype
    ):
        return None
    full_slice = slice(None, None, None)
    expected = (0, 0, full_slice, full_slice, full_slice)
    for task in layers[volume.name].values():
        function = getattr(task, "func", None)
        args = getattr(task, "args", ())
        if (
            getattr(function, "__module__", None) != "dask.array.chunk"
            or getattr(function, "__name__", None) != "getitem"
            or len(args) != 2
            or args[1] != expected
        ):
            return None
    return os.fspath(reader.filePathComplete), reader.ResolutionLevelLock


def _imaris_descriptor(volume, h5py):
    source = _imaris_reader(volume)
    if source is None:
        return None
    path, level = source
    with h5py.File(path, "r") as file:
        dataset = file.get(
            f"DataSet/ResolutionLevel {level}/TimePoint 0/Channel 0/Data"
        )
        if (
            not isinstance(dataset, h5py.Dataset)
            or dataset.shape != volume.shape
            or dataset.dtype != volume.dtype
        ):
            return None
        return _dataset_descriptor(dataset)


def describe_h5_volume(volume):
    """Return an HDF5 descriptor only when graph voxels are unchanged."""
    try:
        import h5py

        dataset = volume if isinstance(volume, h5py.Dataset) else None
        if dataset is None:
            import dask.array as da

            if not isinstance(volume, da.Array) or volume.ndim != 3:
                return None
            original = f"original-{volume.name}"
            if set(volume.dask.layers) != {original, volume.name}:
                return _imaris_descriptor(volume, h5py)
            if not _plain_getter_layer(volume.dask.layers[volume.name]):
                return None
            dataset = volume.dask.get(original)
            if not isinstance(dataset, h5py.Dataset):
                return None
            if dataset.shape != volume.shape or dataset.dtype != volume.dtype:
                return None
        if not dataset.id.valid or dataset.ndim != 3:
            return None
        return _dataset_descriptor(dataset)
    except (
        ImportError,
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None


def process_lanes(descriptor, axis, start, stop):
    """Choose a small pool only when expected decompression is substantial."""
    setting = os.environ.get("NAPARI_ANIMATION_H5_PROCESS_LANES", "auto")
    if setting.lower() == "auto":
        if descriptor[5] is None or descriptor[4] is None:
            return 0
        chunks = descriptor[4]
        touched = 1
        for dimension, (length, chunk) in enumerate(
            zip(descriptor[2], chunks)
        ):
            if dimension == axis:
                touched *= (stop - 1) // chunk - start // chunk + 1
            else:
                touched *= (length + chunk - 1) // chunk
        physical_bytes = (
            touched * int(np.prod(chunks)) * np.dtype(descriptor[3]).itemsize
        )
        if physical_bytes < _MIN_PHYSICAL_BYTES:
            return 0
        requested = 4
        try:
            import psutil

            available = psutil.virtual_memory().available
            if available < 2 * 1024**3:
                return 0
            if available < 4 * 1024**3:
                requested = 2
        except ImportError:
            pass
    else:
        try:
            requested = int(setting)
        except ValueError:
            return 0
    chosen = min(8, max(0, requested), os.cpu_count() or 1)
    return chosen if chosen >= 2 else 0


def direct_uncompressed_read(volume, descriptor, axis):
    """Bypass Dask scheduling for many native chunks and unchanged voxels."""
    if descriptor[5] is not None:
        return False
    try:
        import dask.array as da

        return isinstance(volume, da.Array) and (
            int(
                np.prod(
                    [
                        len(volume.chunks[dimension])
                        for dimension in range(3)
                        if dimension != axis
                    ]
                )
            )
            >= 32
        )
    except (ImportError, AttributeError, TypeError):
        return False


def project_h5_direct(descriptor, axis, start, stop, projection, cancel):
    """Read the verified dataset from an independent read-only file handle."""
    import h5py

    from .processing import project_slab

    with h5py.File(descriptor[0], "r") as file:
        return project_slab(
            file[descriptor[1]], axis, start, stop, projection, cancel
        )


def _tasks(descriptor, axis, start, stop, projection, lanes):
    output_axis = next(index for index in range(3) if index != axis)
    length = descriptor[2][output_axis]
    chunk = (descriptor[4] or (64, 64, 64))[output_axis]
    count = (length + chunk - 1) // chunk
    for index in range(min(lanes, count)):
        total = min(lanes, count)
        first = min(length, round(index * count / total) * chunk)
        last = min(length, round((index + 1) * count / total) * chunk)
        if first < last:
            yield axis, start, stop, projection, first, last


class H5ProcessProjector:
    """Persistent process pool bound to one file, dataset and source revision."""

    def __init__(self, descriptor, lanes):
        self.descriptor = descriptor
        self.lanes = int(lanes)
        self.pool = ProcessPoolExecutor(
            max_workers=self.lanes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_open_worker,
            initargs=descriptor[:2],
        )

    def project(self, axis, start, stop, projection, cancel=None):
        from .processing import CancelledProjection

        if cancel is not None and cancel.is_set():
            raise CancelledProjection()
        tasks = list(
            _tasks(self.descriptor, axis, start, stop, projection, self.lanes)
        )
        parts = list(self.pool.map(_project_stripe, tasks))
        if cancel is not None and cancel.is_set():
            raise CancelledProjection()
        return np.concatenate(parts, axis=0)

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
