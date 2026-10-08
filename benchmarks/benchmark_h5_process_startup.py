"""Compare immediate HDF5 process startup with the existing first local read."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from napari_animation.scene.h5_process import (
        H5ProcessProjector, describe_h5_volume,
    )

    path = Path(r"D:\081926\single tile.ims")
    results = []
    with h5py.File(path, "r") as file:
        for level in (1, 0):
            dataset = file[
                f"DataSet/ResolutionLevel {level}/TimePoint 0/Channel 0/Data"
            ]
            descriptor = describe_h5_volume(dataset)
            projector = H5ProcessProjector(descriptor, 4)
            requests = []
            try:
                for start in (120, 121):
                    begun = time.perf_counter()
                    output = projector.project(0, start, start + 4, "max")
                    requests.append({
                        "start": start,
                        "seconds": round(time.perf_counter() - begun, 4),
                        "sum_uint64": int(output.sum(dtype=np.uint64)),
                    })
            finally:
                projector.shutdown()
            results.append({"level": level, "requests": requests})
    print(json.dumps({"file": str(path), "lanes": 4, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
