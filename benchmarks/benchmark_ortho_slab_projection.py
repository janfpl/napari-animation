"""Compare CPU and GPU backends for thick ortho-slab projections.

usage: python benchmarks/benchmark_ortho_slab_projection.py volume.tif
           [--thickness 150] [--repeat 5] [--json out.json]

Reads one XY, XZ and YZ slab of ``--thickness`` voxels through the centre
of a 3D TIFF into RAM (timed separately), then times the max projection with
NumPy, the Numba CPU kernel and a Numba CUDA kernel. CUDA timings include
the host-to-device copy of the slab and the copy back, since slabs start in
host memory. Every backend's result is compared with NumPy's.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np


def _cuda_kernel():
    from numba import cuda

    @cuda.jit
    def project_max(slab, axis, out):
        row, col = cuda.grid(2)
        if row >= out.shape[0] or col >= out.shape[1]:
            return
        if axis == 0:
            best = slab[0, row, col]
            for k in range(1, slab.shape[0]):
                value = slab[k, row, col]
                if value > best:
                    best = value
        elif axis == 1:
            best = slab[row, 0, col]
            for k in range(1, slab.shape[1]):
                value = slab[row, k, col]
                if value > best:
                    best = value
        else:
            best = slab[row, col, 0]
            for k in range(1, slab.shape[2]):
                value = slab[row, col, k]
                if value > best:
                    best = value
        out[row, col] = best

    def project(slab, axis):
        shape = tuple(s for i, s in enumerate(slab.shape) if i != axis)
        device_slab = cuda.to_device(np.ascontiguousarray(slab))
        device_out = cuda.device_array(shape, dtype=slab.dtype)
        threads = (16, 16)
        blocks = (
            math.ceil(shape[0] / threads[0]),
            math.ceil(shape[1] / threads[1]),
        )
        project_max[blocks, threads](device_slab, axis, device_out)
        result = device_out.copy_to_host()
        cuda.synchronize()
        return result

    return project


def _time(function, repeat):
    function()  # warm up / compile
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume")
    parser.add_argument("--thickness", type=int, default=150)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json")
    args = parser.parse_args()

    import tifffile

    from napari_animation.numba_projection import project as numba_project

    volume = tifffile.memmap(args.volume)
    half = args.thickness // 2
    report = {
        "volume": args.volume,
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "thickness": args.thickness,
        "cases": [],
    }
    try:
        cuda_project = _cuda_kernel()
    except Exception as error:  # noqa: BLE001 - no CUDA on this machine
        cuda_project = None
        report["cuda_error"] = str(error)
    for view, axis in (("XY", 0), ("XZ", 1), ("YZ", 2)):
        centre = volume.shape[axis] // 2
        index = [slice(None)] * 3
        index[axis] = slice(centre - half, centre + half)
        start = time.perf_counter()
        slab = np.ascontiguousarray(volume[tuple(index)])
        # Kernels need native byte order; TIFFs are often big-endian.
        slab = slab.astype(slab.dtype.newbyteorder("="), copy=False)
        read = time.perf_counter() - start
        reference = slab.max(axis=axis)
        case = {
            "view": view,
            "slab_shape": list(slab.shape),
            "slab_gb": slab.nbytes / 1e9,
            "read_s": read,
        }
        backends = {
            "numpy": lambda: slab.max(axis=axis),
            "numba_cpu": lambda: numba_project(slab, axis, "max"),
        }
        if cuda_project is not None:
            backends["numba_cuda"] = lambda: cuda_project(slab, axis)
        for name, function in backends.items():
            try:
                seconds, result = _time(function, args.repeat)
            except Exception as error:  # noqa: BLE001 - e.g. no libNVVM
                case[f"{name}_error"] = str(error).splitlines()[0]
                continue
            case[f"{name}_s"] = seconds
            case[f"{name}_matches"] = bool(np.array_equal(result, reference))
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
