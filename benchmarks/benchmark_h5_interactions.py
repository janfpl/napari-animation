"""Bounded, repeatable GUI-controller benchmarks on real HDF5 files.

Run with the workspace .venv Python. This measures the production Animation
controller with a headless ViewerModel; it does not measure Qt paint latency.
The selected pyramid levels each fit in 32 MiB. Full-resolution probes read
only tiny native-chunk-aligned ROIs, never whole volumes.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
SINGLE = ROOT.parent / "single tile.ims"
EMBRYO = ROOT.parent / "Embryo8b" / "processed" / "20260526-153600_Task_3_Descrip_my_task"


def run_case(label, file_path, dataset_path):
    import dask.array as da
    import h5py
    from napari.components import ViewerModel
    from napari_animation import Animation, ViewerState
    from napari_animation.scene import OrthoSlice
    from napari_animation.scene.renderer import SceneRenderer

    timings = {}
    counts = {}

    def record(action, name):
        counts.clear()
        start = time.perf_counter()
        result = action()
        timings[name] = {
            "seconds": round(time.perf_counter() - start, 4),
            "calls": dict(counts),
        }
        return result

    def wrapper(original, name):
        def wrapped(*args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            return original(*args, **kwargs)
        return wrapped

    with h5py.File(file_path, "r") as f:
        ds = f[dataset_path]
        arr = da.from_array(
            ds, chunks=ds.chunks or "auto", name=False, asarray=False
        )
        viewer = ViewerModel(ndisplay=3)
        viewer.add_image(arr, name="volume", contrast_limits=(0, 65535))
        animation = Animation(viewer)
        section = OrthoSlice(
            name="section", source="volume",
            position=tuple((s - 1) / 2 for s in ds.shape),
            thickness=4,
        )
        animation.add_scene_object(section)
        with ExitStack() as stack:
            for cls, method in (
                (OrthoSlice, "ensure_layer"),
                (OrthoSlice, "_apply_2d"),
                (ViewerState, "apply"),
                (SceneRenderer, "apply_working"),
            ):
                stack.enter_context(patch.object(
                    cls, method, wrapper(getattr(cls, method), method)
                ))
            def screenshot(*args, **kwargs):
                counts["screenshot"] = counts.get("screenshot", 0) + 1
                return np.zeros((32, 32, 4), dtype=np.uint8)
            stack.enter_context(patch.object(
                ViewerModel, "screenshot", screenshot, create=True
            ))
            record(lambda: animation.snap_2d("XY", capture=False), "snap_XY")
            record(
                lambda: animation.add_scroll_through(
                    axis=0, start=ds.shape[0] // 4,
                    stop=ds.shape[0] // 2, steps=5,
                ),
                "add_scroll_through",
            )
            record(
                lambda: animation.add_slice_sweep(
                    section, start=ds.shape[0] // 4,
                    stop=ds.shape[0] // 2, steps=5,
                ),
                "add_slice_sweep",
            )
            record(lambda: animation.set_to_keyframe(0), "select_keyframe")
        viewer.layers.clear()
        return {
            "file": str(file_path), "dataset": dataset_path,
            "shape": ds.shape, "chunks": ds.chunks,
            "compression": ds.compression, "dtype": str(ds.dtype),
            "controller": timings,
        }


def run_io_probes():
    import h5py
    results = []
    specs = (
        (SINGLE, "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"),
        (EMBRYO / "main_st-1.ims", "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"),
        (EMBRYO / "uni_tp-0_ch-0_st-1_obj-left_cam-left_etc.lux.h5", "Data"),
        (ROOT.parent / "Embryo8b" / "raw" / "stack_1_channel_0-561_obj_left"
         / "Cam_left_00000.lux.h5", "Data"),
    )
    for file_path, dataset_path in specs:
        with h5py.File(file_path, "r") as f:
            ds = f[dataset_path]
            # Small XY region and a 4-plane slab; both deliberately bounded.
            z, y, x = (min(s // 2, s - 1) for s in ds.shape)
            key = (slice(z, min(z + 4, ds.shape[0])),
                   slice(y, min(y + 64, ds.shape[1])),
                   slice(x, min(x + 64, ds.shape[2])))
            start = time.perf_counter()
            chunk = ds[key]
            results.append({
                "file": str(file_path), "dataset": dataset_path,
                "shape": ds.shape, "chunks": ds.chunks,
                "compression": ds.compression,
                "sample_bytes": int(chunk.nbytes),
                "sample_seconds": round(time.perf_counter() - start, 4),
            })
    return results


def main():
    import platformdirs

    cache = Path(__file__).resolve().parents[1] / ".performance-audit"
    cache.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for kind in ("cache", "config", "data", "state", "log"):
            stack.enter_context(patch.object(
                platformdirs, f"user_{kind}_dir", return_value=str(cache)
            ))
        from napari.settings import get_settings
        get_settings().experimental.async_ = False
        output = {
            "scope": "headless controller and bounded HDF5 read; no Qt/OpenGL timing",
            "io_probes": run_io_probes(),
            "cases": [
                run_case(
                    "single tile level 3", SINGLE,
                    "DataSet/ResolutionLevel 3/TimePoint 0/Channel 0/Data",
                ),
                run_case(
                    "Embryo8b level 2", EMBRYO / "main_st-1.ims",
                    "DataSet/ResolutionLevel 2/TimePoint 0/Channel 0/Data",
                ),
                run_case(
                    "Embryo8b direct Luxendo level 2",
                    EMBRYO / "uni_tp-0_ch-0_st-1_obj-left_cam-left_etc.lux.h5",
                    "Data_4_4_4",
                ),
            ],
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
