"""Count HDF5 chunk reads for concurrent versus deduplicated scroll prefetch."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import dask.array as da
import h5py
import numpy as np
import platformdirs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2].parent
EMBRYO = (ROOT / "Embryo8b" / "processed"
          / "20260526-153600_Task_3_Descrip_my_task")
CASES = (
    (ROOT / "single tile.ims",
     "DataSet/ResolutionLevel 3/TimePoint 0/Channel 0/Data"),
    (EMBRYO / "uni_tp-0_ch-0_st-1_obj-left_cam-left_etc.lux.h5",
     "Data_4_4_4"),
)


class CountingSource:
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset.shape
        self.dtype = dataset.dtype
        self.ndim = dataset.ndim
        self._lock = Lock()
        self.reads = 0

    def __getitem__(self, key):
        with self._lock:
            self.reads += 1
        return self.dataset[key]


def run_case(path, dataset_name):
    from napari_animation.prefetch import SlicePrefetcher, dask_cache_context

    with h5py.File(path, "r") as file:
        dataset = file[dataset_name]
        source = CountingSource(dataset)
        array = da.from_array(
            source, chunks=dataset.chunks, asarray=False, name=False
        )
        frames = [SimpleNamespace(dims={
            "ndim": 3, "ndisplay": 2, "order": [0, 1, 2],
            "point": [z, 0, 0],
            "range": [(0, dataset.shape[i] - 1, 1) for i in range(3)],
        }) for z in range(4)]
        viewer = SimpleNamespace(layers=[SimpleNamespace(
            data=array, ndim=3, multiscale=False, visible=True,
        )])

        # Model the old frame-based prefetch: all three workers request a
        # different plane from the same native Z chunk at once.
        source.reads = 0
        with dask_cache_context(128 * 1024**2):
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda z: np.asarray(array[z]), (1, 2, 3)))
        concurrent_reads = source.reads

        source.reads = 0
        prefetcher = SlicePrefetcher(viewer, frames, depth=3, workers=3)
        try:
            with dask_cache_context(128 * 1024**2):
                prefetcher.advance(0)
                scheduled = len(prefetcher._futures)
                for future, _ in prefetcher._futures.values():
                    future.result(timeout=30)
                deduplicated_reads = source.reads
        finally:
            prefetcher.shutdown()
        return {
            "file": str(path), "dataset": dataset_name,
            "shape": dataset.shape, "native_chunks": dataset.chunks,
            "lookahead_frames": 3,
            "concurrent_frame_reads": concurrent_reads,
            "deduplicated_reads": deduplicated_reads,
            "scheduled_tasks": scheduled,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cache = Path(__file__).resolve().parents[1] / ".performance-audit"
    cache.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for kind in ("cache", "config", "data", "state", "log"):
            stack.enter_context(patch.object(
                platformdirs, f"user_{kind}_dir", return_value=str(cache)
            ))
        report = json.dumps({"cases": [
            run_case(*case) for case in CASES
        ]}, indent=2)
        if args.output:
            args.output.write_text(report + "\n", encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
