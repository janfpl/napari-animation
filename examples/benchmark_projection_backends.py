"""Compare warmed NumPy and Numba reductions on bounded real HDF5 slabs."""

import json
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import h5py
import numba
import numpy as np
import platformdirs

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
SINGLE = PROJECT.parents[1] / "single tile.ims"
EMBRYO = (
    PROJECT.parents[1] / "Embryo8b" / "processed"
    / "20260526-153600_Task_3_Descrip_my_task" / "main_st-1.ims"
)


def measure(fn, repeats=4):
    times = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - start)
    return value, round(statistics.median(times), 6)


def run():
    from napari_animation.numba_projection import project, warm

    numba.set_num_threads(min(8, numba.config.NUMBA_NUM_THREADS))
    start = time.perf_counter()
    warm()
    compile_seconds = round(time.perf_counter() - start, 3)
    cases = []
    for file_path, path, planes in (
        (SINGLE, "DataSet/ResolutionLevel 2/TimePoint 0/Channel 0/Data", 32),
        (EMBRYO, "DataSet/ResolutionLevel 1/TimePoint 0/Channel 0/Data", 16),
    ):
        with h5py.File(file_path, "r") as f:
            ds = f[path]
            start = time.perf_counter()
            slab = ds[ds.shape[0] // 4:ds.shape[0] // 4 + planes]
            read_seconds = round(time.perf_counter() - start, 4)
        timings = {}
        for mode in ("max", "min", "mean"):
            expected, numpy_seconds = measure(
                lambda: getattr(np, mode)(slab, axis=0)
            )
            actual, numba_seconds = measure(
                lambda: project(slab, 0, mode)
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
            timings[mode] = {
                "numpy_median_s": numpy_seconds,
                "numba_median_s": numba_seconds,
                "speedup": round(numpy_seconds / numba_seconds, 2),
            }
        cases.append({
            "file": str(file_path), "dataset": path,
            "slab_shape": slab.shape, "slab_bytes": int(slab.nbytes),
            "read_seconds": read_seconds, "compute": timings,
        })
    return {
        "scope": "bounded slab read plus warmed compute-only timings; no GUI or GPU",
        "numba_threads": numba.get_num_threads(),
        "first_warm_seconds": compile_seconds,
        "cases": cases,
    }


if __name__ == "__main__":
    cache = PROJECT / ".performance-audit"
    cache.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for kind in ("cache", "config", "data", "state", "log"):
            stack.enter_context(patch.object(
                platformdirs, f"user_{kind}_dir", return_value=str(cache)
            ))
        print(json.dumps(run(), indent=2))
