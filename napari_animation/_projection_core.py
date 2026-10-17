"""Pure, bounded slab reduction shared by GUI and HDF5 reader processes."""

from __future__ import annotations

import numpy as np


def project_slab_core(
    volume,
    axis,
    start,
    stop,
    projection,
    cancel=None,
    target_read_bytes=64 * 1024**2,
    cancel_error=RuntimeError,
):
    """Reduce a 3D slab through chunk-aligned basic ROI reads."""
    shape = tuple(int(s) for s in volume.shape)
    dtype = np.dtype(volume.dtype)
    output_axes = [index for index in range(3) if index != axis]
    output_shape = tuple(shape[index] for index in output_axes)
    depth = stop - start
    if depth <= 0:
        raise ValueError("Projection interval must contain at least one voxel")
    if projection not in {"max", "min", "mean"}:
        raise ValueError(f"Unknown projection {projection!r}")
    output_dtype = (
        np.dtype(np.float64)
        if projection == "mean" and dtype.kind in "biu"
        else dtype
    )
    result = np.empty(output_shape, dtype=output_dtype)

    chunks = getattr(volume, "chunks", None)
    if chunks is None:
        chunk_shape = (1, 1, 1)
    else:
        chunk_shape = tuple(
            int(part[0] if isinstance(part, tuple) else part)
            for part in chunks
        )
    chunk0, chunk1 = (chunk_shape[index] for index in output_axes)
    max_pixels = max(1, target_read_bytes // max(1, depth * dtype.itemsize))
    if output_shape[0] * output_shape[1] <= max_pixels:
        tile0, tile1 = output_shape
    else:
        side = max(1, int(np.sqrt(max_pixels)))
        tile0 = max(1, min(output_shape[0], (side // chunk0) * chunk0 or side))
        remaining = max(1, max_pixels // tile0)
        tile1 = max(
            1,
            min(output_shape[1], (remaining // chunk1) * chunk1 or remaining),
        )
    for low0 in range(0, output_shape[0], tile0):
        high0 = min(output_shape[0], low0 + tile0)
        for low1 in range(0, output_shape[1], tile1):
            if cancel is not None and cancel.is_set():
                raise cancel_error()
            high1 = min(output_shape[1], low1 + tile1)
            index = [slice(None)] * 3
            index[axis] = slice(start, stop)
            index[output_axes[0]] = slice(low0, high0)
            index[output_axes[1]] = slice(low1, high1)
            block = np.asarray(volume[tuple(index)])
            if projection == "mean":
                reduced = np.mean(block, axis=axis)
            elif projection == "max":
                reduced = np.max(block, axis=axis)
            else:
                reduced = np.min(block, axis=axis)
            result[low0:high0, low1:high1] = reduced
    return result
