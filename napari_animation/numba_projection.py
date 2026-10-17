"""Optional multi-core projection kernel for materialized 3D slabs.

Import this module only when benchmarking or after a background warm-up: the
first call for each dtype compiles a kernel and must not run in a Qt callback.
HDF5/Dask reading is deliberately outside this kernel.
"""

from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(parallel=True, nogil=True, cache=True)
def _project(slab, axis, mode, out):
    """Reduce one spatial axis; independent output pixels run on CPU cores."""
    ny, nx = out.shape
    length = slab.shape[axis]
    for flat in numba.prange(ny * nx):
        row = flat // nx
        col = flat - row * nx
        if axis == 0:
            z, y, x = 0, row, col
        elif axis == 1:
            z, y, x = row, 0, col
        else:
            z, y, x = row, col, 0

        if mode == 2:
            total = 0.0
            for k in range(length):
                if axis == 0:
                    total += slab[k, y, x]
                elif axis == 1:
                    total += slab[z, k, x]
                else:
                    total += slab[z, y, k]
            out[row, col] = total / length
        else:
            best = slab[z, y, x]
            for k in range(1, length):
                if axis == 0:
                    value = slab[k, y, x]
                elif axis == 1:
                    value = slab[z, k, x]
                else:
                    value = slab[z, y, k]
                if math.isnan(float(value)):
                    best = value
                    break
                if mode == 0 and value > best:
                    best = value
                elif mode == 1 and value < best:
                    best = value
            out[row, col] = best


def project(slab: np.ndarray, axis: int, projection: str) -> np.ndarray:
    """Return a max/min/mean projection using the warmed Numba backend."""
    if slab.ndim != 3 or axis not in (0, 1, 2):
        raise ValueError("Expected a 3D slab and spatial axis 0, 1, or 2")
    try:
        mode = {"max": 0, "min": 1, "mean": 2}[projection]
    except KeyError as error:
        raise ValueError(f"Unknown projection {projection!r}") from error
    if slab.shape[axis] < 1:
        raise ValueError("Cannot project an empty slab")
    if slab.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise TypeError(f"Unsupported projection dtype {slab.dtype}")

    shape = tuple(size for i, size in enumerate(slab.shape) if i != axis)
    dtype = (
        np.float64
        if projection == "mean" and np.issubdtype(slab.dtype, np.integer)
        else slab.dtype
    )
    result = np.empty(shape, dtype=dtype)
    _project(slab, axis, mode, result)
    return result


def warm() -> None:
    """Compile supported dtype/return combinations off the GUI thread."""
    for dtype in (np.uint8, np.uint16, np.float32, np.float64):
        sample = np.zeros((2, 2, 2), dtype=dtype)
        project(sample, 0, "max")
        project(sample, 0, "mean")
