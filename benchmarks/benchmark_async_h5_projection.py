"""Measure Qt heartbeat while real IMS/Luxendo slabs are read in a worker.

The benchmark uses bounded pyramid levels and one four-plane full-resolution
ROI.  It does not create an OpenGL canvas or claim to measure GPU paint time.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import dask.array as da
import h5py
from qtpy.QtCore import QCoreApplication, QEventLoop, QTimer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2].parent
SINGLE = ROOT / "single tile.ims"
EMBRYO = ROOT / "Embryo8b" / "processed" / "20260526-153600_Task_3_Descrip_my_task"


def run_case(path, dataset, axis, start, stop):
    from napari_animation.scene.processing import ProjectionService

    with h5py.File(path, "r") as file:
        ds = file[dataset]
        volume = da.from_array(
            ds, chunks=ds.chunks or "auto", name=False, asarray=False
        )
        service = ProjectionService(asynchronous=True)
        loop = QEventLoop()
        ticks = []
        results = []
        timer = QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
        timer.start()
        started = time.perf_counter()
        key = (str(path), dataset, axis, start, stop, "max")
        service.request(
            "section", key, volume, axis, start, stop, "max",
            lambda result: results.append((time.perf_counter(), result.shape)),
        )
        service.when_idle(loop.quit)
        QTimer.singleShot(30000, loop.quit)
        loop.exec_()
        timer.stop()
        service.shutdown()
        if not results:
            raise RuntimeError(f"Projection timed out: {path} {dataset}")
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        return {
            "file": str(path), "dataset": dataset, "shape": ds.shape,
            "chunks": ds.chunks, "compression": ds.compression,
            "axis": axis, "interval": [start, stop],
            "result_shape": results[0][1],
            "exact_seconds": round(results[0][0] - started, 4),
            "qt_heartbeat_max_ms": round(max(gaps, default=0) * 1000, 2),
            "qt_heartbeat_p95_ms": round(
                sorted(gaps)[int(.95 * (len(gaps) - 1))] * 1000, 2
            ) if gaps else None,
            "heartbeat_samples": len(gaps),
        }


def main():
    import platformdirs

    cache = Path(__file__).resolve().parents[1] / ".performance-audit"
    cache.mkdir(exist_ok=True)
    cases = [
        (SINGLE, "DataSet/ResolutionLevel 3/TimePoint 0/Channel 0/Data", 0, 64, 96),
        (EMBRYO / "main_st-1.ims", "DataSet/ResolutionLevel 2/TimePoint 0/Channel 0/Data", 1, 192, 224),
        (EMBRYO / "uni_tp-0_ch-0_st-1_obj-left_cam-left_etc.lux.h5", "Data_4_4_4", 2, 192, 224),
        (EMBRYO / "uni_tp-0_ch-0_st-1_obj-left_cam-left_etc.lux.h5", "Data", 0, 120, 124),
    ]
    with ExitStack() as stack:
        for kind in ("cache", "config", "data", "state", "log"):
            stack.enter_context(patch.object(
                platformdirs, f"user_{kind}_dir", return_value=str(cache)
            ))
        app = QCoreApplication.instance() or QCoreApplication([])
        print(json.dumps({"scope": "Qt worker heartbeat; no OpenGL paint", "cases": [
            run_case(*case) for case in cases
        ]}, indent=2))
    assert app is not None


if __name__ == "__main__":
    main()
