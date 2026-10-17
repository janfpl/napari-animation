"""Export work that cannot change a rendered frame is skipped."""

from types import SimpleNamespace

import numpy as np
from napari.components import ViewerModel

from napari_animation import ViewerState
from napari_animation.render_capture import ExportCaptureSession


def _viewer_3d():
    viewer = ViewerModel()
    layer = viewer.add_image(np.random.random((8, 16, 16)))
    viewer.dims.ndisplay = 3
    return viewer, layer


def _count_slices(layer):
    calls = []
    layer.events.set_data.connect(lambda event: calls.append(1))
    return calls


def test_displayed_point_move_keeps_the_3d_slice():
    viewer, layer = _viewer_3d()
    state = ViewerState.from_viewer(viewer)
    point = list(state.dims["point"])
    point[1] = 3.0
    point[2] = 12.0
    moved = ViewerState(
        camera=state.camera,
        dims={**state.dims, "point": tuple(point)},
        layers=state.layers,
    )
    slices = _count_slices(layer)
    moved.apply(viewer)
    assert slices == []
    np.testing.assert_allclose(viewer.dims.point, point)
    # The dims model and the layer keep emitting their own events.
    state.apply(viewer)
    assert slices == []


def test_point_on_a_sliced_axis_still_reslices():
    viewer, layer = _viewer_3d()
    viewer.dims.ndisplay = 2
    state = ViewerState.from_viewer(viewer)
    point = list(state.dims["point"])
    point[0] = 5.0
    moved = ViewerState(
        camera=state.camera,
        dims={**state.dims, "point": tuple(point)},
        layers=state.layers,
    )
    slices = _count_slices(layer)
    moved.apply(viewer)
    assert slices
    assert layer._slice_input.world_slice.point[0] == 5.0


def test_display_switch_with_point_move_still_reslices():
    viewer, layer = _viewer_3d()
    state = ViewerState.from_viewer(viewer)
    point = list(state.dims["point"])
    point[2] = 12.0
    switched = ViewerState(
        camera=state.camera,
        dims={**state.dims, "point": tuple(point), "ndisplay": 2},
        layers=state.layers,
    )
    slices = _count_slices(layer)
    switched.apply(viewer)
    assert slices
    assert viewer.dims.ndisplay == 2


class _Native:
    def __init__(self):
        self.enabled = True
        self.updates = 0

    def width(self):
        return 800

    def height(self):
        return 600

    def devicePixelRatioF(self):
        return 1.0

    def resize(self, *size):
        pass

    def updatesEnabled(self):
        return self.enabled

    def setUpdatesEnabled(self, enabled):
        self.enabled = enabled

    def update(self):
        self.updates += 1


def _viewer_with_native(native):
    return SimpleNamespace(
        camera=SimpleNamespace(zoom=1.0, center=(0, 0, 0)),
        window=SimpleNamespace(
            _qt_viewer=SimpleNamespace(
                canvas=SimpleNamespace(
                    native=native, on_draw=lambda event: None
                )
            )
        ),
    )


def test_canvas_only_session_pauses_and_resumes_repaints():
    for render_size in (None, (720, 1280)):
        native = _Native()
        with ExportCaptureSession(_viewer_with_native(native), render_size):
            assert not native.enabled
        assert native.enabled
        assert native.updates == 1


def test_full_window_session_keeps_repaints():
    native = _Native()
    viewer = _viewer_with_native(native)
    with ExportCaptureSession(viewer, None, canvas_only=False):
        assert native.enabled
    assert native.updates == 0


def test_session_leaves_already_paused_canvas_paused():
    native = _Native()
    native.enabled = False
    with ExportCaptureSession(_viewer_with_native(native), None):
        pass
    assert not native.enabled
