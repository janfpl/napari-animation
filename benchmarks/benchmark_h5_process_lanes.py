"""Test independent read-only HDF5 processes on one bounded IMS/Lux slab.

The first request includes Windows worker startup; the second uses the same
pool. File-system cache state is uncontrolled, so run multiple times and
compare warmed requests before choosing a production reader backend.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

_FILE = None
_DATASET = None


def _open_reader(path, dataset):
    global _FILE, _DATASET
    _FILE = h5py.File(path, "r")
    _DATASET = _FILE[dataset]


def _read_stripe(task):
    axis, start, stop, first, last = task
    output_axis = next(index for index in range(3) if index != axis)
    selection = [slice(None)] * 3
    selection[axis] = slice(start, stop)
    selection[output_axis] = slice(first, last)
    block = np.asarray(_DATASET[tuple(selection)])
    return np.max(block, axis=axis)


def _tasks(shape, chunks, axis, start, stop, lanes):
    output_axis = next(index for index in range(3) if index != axis)
    length = shape[output_axis]
    chunk = chunks[output_axis] if chunks else 64
    count = (length + chunk - 1) // chunk
    for index in range(lanes):
        first = min(length, round(index * count / lanes) * chunk)
        last = min(length, round((index + 1) * count / lanes) * chunk)
        if first < last:
            yield axis, start, stop, first, last


def _result(parts, started):
    result = np.concatenate(parts, axis=0)
    return {
        "seconds": round(time.perf_counter() - started, 4),
        "shape": result.shape,
        "sum_uint64": int(result.sum(dtype=np.uint64)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lanes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with h5py.File(args.path, "r") as file:
        dataset = file[args.dataset]
        shape, chunks = dataset.shape, dataset.chunks
        selection = [slice(None)] * 3
        selection[args.axis] = slice(args.start, args.start + args.depth)
        direct = []
        for start in (args.start, args.start + 1):
            selection[args.axis] = slice(start, start + args.depth)
            begun = time.perf_counter()
            direct.append(_result([
                np.max(dataset[tuple(selection)], axis=args.axis)
            ], begun))

    runs = []
    context = multiprocessing.get_context("spawn")
    for lanes in args.lanes:
        with ProcessPoolExecutor(
            max_workers=lanes, mp_context=context,
            initializer=_open_reader, initargs=(str(args.path), args.dataset),
        ) as pool:
            requests = []
            for start in (args.start, args.start + 1):
                tasks = list(_tasks(
                    shape, chunks, args.axis, start, start + args.depth,
                    lanes,
                ))
                begun = time.perf_counter()
                requests.append(_result(list(pool.map(_read_stripe, tasks)), begun))
            runs.append({"lanes": lanes, "requests": requests})

    report = {
        "file": str(args.path), "dataset": args.dataset,
        "shape": shape, "chunks": chunks, "axis": args.axis,
        "intervals": [[args.start, args.start + args.depth],
                      [args.start + 1, args.start + args.depth + 1]],
        "direct": direct, "process_pools": runs,
    }
    expected = [result["sum_uint64"] for result in direct]
    if any(
        [result["sum_uint64"] for result in run["requests"]] != expected
        for run in runs
    ):
        raise AssertionError("A process result differed from direct HDF5")
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
