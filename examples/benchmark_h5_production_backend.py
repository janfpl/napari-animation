"""Exercise the production Qt projection service on IMS and Luxendo HDF5."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import dask.array as da
import h5py
import numpy as np
import platformdirs
import psutil
from qtpy.QtCore import QCoreApplication, QEventLoop, QTimer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[2].parent
EMBRYO = (ROOT / "Embryo8b" / "processed"
          / "20260526-153600_Task_3_Descrip_my_task")


class ims_reader:
    """Minimal source with the graph structure of napari-imaris-loader."""

    def __init__(self, path, dataset, dataset_name):
        self.filePathComplete = str(path)
        self.data_path = dataset.file.filename
        self.dataset_name = dataset.name
        self.ResolutionLevelLock = int(
            dataset_name.split("ResolutionLevel ", 1)[1].split("/", 1)[0]
        )
        self.squeeze_output = False
        self.write = False
        self.shape = (1, 1, *dataset.shape)
        self.chunks = (1, 1, *dataset.chunks)
        self.ndim = 5
        self.dtype = dataset.dtype

    def __getitem__(self, key):
        with h5py.File(self.data_path, "r") as file:
            data = np.asarray(file[self.dataset_name][key[2:]])
        return data[np.newaxis, np.newaxis]


ims_reader.__module__ = "imaris_ims_file_reader.ims"


def run_case(path, dataset_name, start, reader_graph=False):
    from napari_animation.scene.processing import ProjectionService

    with h5py.File(path, "r") as file:
        dataset = file[dataset_name]
        if reader_graph:
            source = ims_reader(path, dataset, dataset_name)
            volume = da.from_array(
                source, chunks=source.chunks, fancy=False
            )[0, 0]
        else:
            volume = da.from_array(
                dataset, chunks=dataset.chunks, name=False, asarray=False
            )
        service = ProjectionService(asynchronous=True)
        cases = []
        try:
            for position in (start, start + 1, start + 2):
                loop = QEventLoop()
                results = []
                ticks = []
                timer = QTimer()
                timer.setInterval(10)
                timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
                timer.start()
                begun = time.perf_counter()
                pool_existed_at_start = service._process_projector is not None
                key = (str(path), 0, id(volume), position, 0, position,
                       position + 4, "max")

                def submit():
                    service.request(
                        "section", key, volume, 0, position, position + 4,
                        "max", results.append,
                    )
                    service.when_idle(loop.quit)

                QTimer.singleShot(0, submit)
                deadline = QTimer()
                deadline.setSingleShot(True)
                deadline.timeout.connect(loop.quit)
                deadline.start(30000)
                loop.exec_()
                deadline.stop()
                timer.stop()
                if not results:
                    raise RuntimeError("Section projection timed out")
                result = results[0]
                elapsed = time.perf_counter() - begun
                expected = np.max(
                    dataset[position:position + 4, :, :], axis=0
                )
                np.testing.assert_array_equal(result, expected)
                gaps = [b - a for a, b in zip(ticks, ticks[1:])]
                cases.append({
                    "interval": [position, position + 4],
                    "pool_existed_at_start": pool_existed_at_start,
                    "exact_seconds": round(elapsed, 4),
                    "sum_uint64": int(result.sum(dtype=np.uint64)),
                    "qt_heartbeat_max_ms": round(
                        max(gaps, default=0) * 1000, 2
                    ),
                })
        finally:
            used_processes = service._process_projector is not None
            lanes = (
                service._process_projector.lanes if used_processes else 0
            )
            workers = (
                list(service._process_projector.pool._processes)
                if used_processes else []
            )
            worker_processes = [psutil.Process(pid) for pid in workers]
            worker_rss_mib = round(sum(
                process.memory_info().rss for process in worker_processes
            ) / 2**20, 1)
            process_failed = service._process_failed
            service.shutdown()
            _, remaining = psutil.wait_procs(
                worker_processes, timeout=3
            )
        return {
            "file": str(path), "dataset": dataset_name,
            "reader_graph": reader_graph,
            "shape": dataset.shape, "chunks": dataset.chunks,
            "compression": dataset.compression,
            "process_lanes": lanes, "process_failed": process_failed,
            "worker_rss_mib": worker_rss_mib,
            "workers_after_close": len(remaining),
            "cases": cases,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lux-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    cache = Path(__file__).resolve().parents[1] / ".performance-audit"
    cache.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for kind in ("cache", "config", "data", "state", "log"):
            stack.enter_context(patch.object(
                platformdirs, f"user_{kind}_dir", return_value=str(cache)
            ))
        app = QCoreApplication.instance() or QCoreApplication([])
        specs = [
            (
                ROOT / "single tile.ims",
                "DataSet/ResolutionLevel 1/TimePoint 0/Channel 0/Data",
                120, False,
            ),
            (
                ROOT / "single tile.ims",
                "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
                120, False,
            ),
            (
                EMBRYO / "uni_tp-0_ch-0_st-1_obj-left_cam-left_etc.lux.h5",
                "Data", 120, False,
            ),
            (
                ROOT / "single tile.ims",
                "DataSet/ResolutionLevel 1/TimePoint 0/Channel 0/Data",
                120, True,
            ),
            (
                EMBRYO / "main_st-1.ims",
                "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
                120, True,
            ),
        ]
        if args.lux_only:
            specs = [specs[2], specs[4]]
        report = {"cases": [run_case(*spec) for spec in specs]}
        output = Path(__file__).resolve().parents[1] / "docs"
        output /= args.output or (
            "performance-lux-forced-processes.json" if args.lux_only
            else "performance-h5-production-processes.json"
        )
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        assert app is not None


if __name__ == "__main__":
    main()
