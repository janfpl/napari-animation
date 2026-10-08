"""Compare a bounded section read through Dask and native HDF5.

Run each mode in a fresh process to compare peak RSS fairly. The same slab and
projection are used in both modes; this measures Python process memory, not
the operating system's file cache or a napari OpenGL canvas.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import dask.array as da
import h5py
import platformdirs
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("--mode", choices=("dask", "native"), required=True)
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--stop", type=int, default=124)
    args = parser.parse_args()

    cache = Path(__file__).resolve().parents[1] / ".performance-audit"
    cache.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for kind in ("cache", "config", "data", "state", "log"):
            stack.enter_context(patch.object(
                platformdirs, f"user_{kind}_dir", return_value=str(cache)
            ))
        from napari_animation.scene import processing

        with h5py.File(args.path, "r") as file:
            dataset = file[args.dataset]
            array = da.from_array(
                dataset, chunks=(min(64, dataset.shape[0]), -1, -1),
                asarray=False, name=False,
            )
            process = psutil.Process()
            baseline = process.memory_info().rss
            peak = [baseline]
            stop_sampling = threading.Event()

            def sample():
                while not stop_sampling.wait(0.002):
                    peak[0] = max(peak[0], process.memory_info().rss)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            started = time.perf_counter()
            output = processing.project_slab(
                array if args.mode == "dask" else dataset,
                args.axis, args.start, args.stop, "max",
            )
            elapsed = time.perf_counter() - started
            stop_sampling.set()
            sampler.join()
            peak[0] = max(peak[0], process.memory_info().rss)
            print(json.dumps({
                "file": str(args.path), "dataset": args.dataset,
                "shape": dataset.shape, "h5_chunks": dataset.chunks,
                "dask_chunks": array.chunks, "mode": args.mode,
                "axis": args.axis, "interval": [args.start, args.stop],
                "seconds": round(elapsed, 4),
                "baseline_rss_mib": round(baseline / 2**20, 1),
                "peak_rss_mib": round(peak[0] / 2**20, 1),
                "added_peak_rss_mib": round((peak[0] - baseline) / 2**20, 1),
                "result_shape": output.shape,
                "result_sum": int(output.sum(dtype="uint64")),
            }))


if __name__ == "__main__":
    main()
