"""Small headless audit probes; no HDF5 or GPU performance claims.

Run with the project's Python. Napari's disposable settings/theme directories
are redirected to .performance-audit so the user's settings are untouched.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run():
    import dask.array as da
    from napari.components import ViewerModel
    from napari.settings import get_settings
    from napari_animation.prefetch import _layer_level_array, read_keys
    from napari_animation.scene.ortho_slice import OrthoSlice

    get_settings().experimental.async_ = False
    viewer = ViewerModel(ndisplay=3)
    arr = da.from_array(
        np.arange(16**3, dtype=np.uint16).reshape((16,) * 3),
        chunks=(4, 8, 8),
    )
    viewer.add_image(arr, name="source", contrast_limits=(0, 4095))
    section = OrthoSlice(
        name="section", source="source", position=(8, 8, 8), thickness=4
    )
    backing = section.ensure_layer(viewer)
    events = []
    backing.events.data.connect(lambda event: events.append(1))
    for _ in range(3):
        section.ensure_layer(viewer)

    second = ViewerModel(ndisplay=2)
    multi = second.add_image(
        [arr, arr[::2, ::2, ::2]], multiscale=True,
        contrast_limits=(0, 4095),
    )
    dims = second.dims.model_dump()
    read_error = None
    try:
        for data, index in read_keys(second, dims):
            np.asarray(data[index])
    except Exception as error:
        read_error = f"{type(error).__name__}: {error}"
    results = {
        "scope": "synthetic array/model checks; no Qt canvas, disk or GPU timing",
        "unchanged_ensure_layer_calls": 3,
        "backing_data_events": len(events),
        "multiscale_container": type(multi.data).__name__,
        "prefetch_returns_container_instead_of_level":
            _layer_level_array(multi) is multi.data,
        "multiscale_prefetch_read_key_count":
            len(read_keys(second, dims)),
        "multiscale_prefetch_read_error": read_error,
    }

    class ArrayProtocolSource:
        shape = (16, 16, 16)
        ndim = 3
        dtype = np.dtype("uint16")

        def __init__(self):
            self.full_materializations = 0
            self.slice_requests = 0

        def __array__(self, dtype=None, copy=None):
            self.full_materializations += 1
            return np.zeros(self.shape, dtype=self.dtype)

        def __getitem__(self, key):
            self.slice_requests += 1
            return np.zeros(self.shape, dtype=self.dtype)[key]

    source = ArrayProtocolSource()
    np.asarray(np.take(source, np.arange(4, 8), axis=0))
    results["array_protocol_take_full_materializations"] = source.full_materializations
    source = ArrayProtocolSource()
    np.asarray(source[4:8, :, :])
    results["basic_slice_full_materializations"] = source.full_materializations
    results["basic_slice_requests"] = source.slice_requests

    # Exercise the exact controller methods used by the reported buttons.
    # Screenshots are stubbed: these are call counts, never render timings.
    from contextlib import ExitStack
    from napari_animation import Animation, ViewerState
    from napari_animation.scene.renderer import SceneRenderer
    third = ViewerModel(ndisplay=3)
    third.add_image(arr, name="source", contrast_limits=(0, 4095))
    animation = Animation(third)
    section = OrthoSlice(
        name="section", source="source", position=(8, 8, 8), thickness=4
    )
    animation.add_scene_object(section)

    def count_action(action):
        counts = {}
        def wrapper(original, label):
            def wrapped(*args, **kwargs):
                counts[label] = counts.get(label, 0) + 1
                return original(*args, **kwargs)
            return wrapped
        def screenshot(*args, **kwargs):
            counts["screenshot"] = counts.get("screenshot", 0) + 1
            return np.zeros((32, 32, 4), dtype=np.uint8)
        with ExitStack() as stack:
            for cls, method in (
                (OrthoSlice, "ensure_layer"), (OrthoSlice, "_apply_2d"),
                (ViewerState, "apply"), (SceneRenderer, "apply_working"),
            ):
                stack.enter_context(patch.object(
                    cls, method, wrapper(getattr(cls, method), method)
                ))
            stack.enter_context(patch.object(
                ViewerModel, "screenshot", screenshot, create=True
            ))
            action()
        return counts

    results["controller_call_counts_without_qt_canvas"] = {
        "snap_XY_capture_false": count_action(
            lambda: animation.snap_2d("XY", capture=False)
        ),
        "add_scroll_through": count_action(
            lambda: animation.add_scroll_through(axis=0, start=2, stop=12)
        ),
        "add_slice_sweep": count_action(
            lambda: animation.add_slice_sweep(section, start=2, stop=12)
        ),
        "select_first_keyframe": count_action(
            lambda: animation.set_to_keyframe(0)
        ),
    }
    third.layers.clear()
    viewer.layers.clear()
    second.layers.clear()
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    temp = root / ".performance-audit"
    temp.mkdir(exist_ok=True)
    # Patch only this process, before napari imports platformdirs aliases.
    from contextlib import ExitStack
    with ExitStack() as stack:
        for kind in ("cache", "config", "data", "state", "log"):
            stack.enter_context(patch(
                f"platformdirs.user_{kind}_dir", return_value=str(temp)
            ))
        print(json.dumps(run(), indent=2))
