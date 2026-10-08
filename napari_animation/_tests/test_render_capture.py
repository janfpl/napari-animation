"""Preset captures must rerender at a higher pixel scale."""

from types import SimpleNamespace

import numpy as np
import pytest

from napari_animation.render_capture import capture_export_frame


class Native:
    def __init__(self, ratio=1):
        self.size = (800, 600)
        self.ratio = ratio

    def width(self):
        return self.size[0]

    def height(self):
        return self.size[1]

    def devicePixelRatioF(self):
        return self.ratio

    def resize(self, width, height):
        self.size = (width, height)


def viewer_with_canvas(screenshot, ratio=1):
    return SimpleNamespace(
        camera=SimpleNamespace(zoom=2.0, center=(0, 50, 50)),
        window=SimpleNamespace(
            _qt_viewer=SimpleNamespace(canvas=SimpleNamespace(
                native=Native(ratio), on_draw=lambda event: None,
            ))
        ),
        screenshot=screenshot,
    )


def test_preset_scales_camera_during_render_and_restores_it():
    observed = []

    def screenshot(**kwargs):
        observed.append((viewer.camera.zoom, kwargs))
        native = viewer.window._qt_viewer.canvas.native
        return np.zeros((native.height(), native.width(), 4), dtype=np.uint8)

    viewer = viewer_with_canvas(screenshot)
    frame = capture_export_frame(viewer, render_size=(1200, 1600))
    assert frame.shape == (1200, 1600, 4)
    assert observed == [(4.0, {
        "canvas_only": True, "flash": False,
    })]
    assert viewer.camera.zoom == 2.0
    assert viewer.window._qt_viewer.canvas.native.size == (800, 600)


def test_changed_aspect_ratio_fits_old_view_and_errors_restore_zoom():
    def screenshot(**kwargs):
        assert viewer.camera.zoom == 3.6  # min(1920/800, 1080/600)
        raise RuntimeError("capture failed")

    viewer = viewer_with_canvas(screenshot)
    with pytest.raises(RuntimeError, match="capture failed"):
        capture_export_frame(viewer, render_size=(1080, 1920))
    assert viewer.camera.zoom == 2.0
    assert viewer.window._qt_viewer.canvas.native.size == (800, 600)


@pytest.mark.parametrize("ratio", [1.0, 1.25, 1.5, 1.75, 2.0])
def test_fractional_scaling_preserves_dimensions_and_pixels(ratio):
    captured = []

    def screenshot(**kwargs):
        assert "size" not in kwargs
        native = viewer.window._qt_viewer.canvas.native
        height = int(native.height() * ratio + 0.5)
        width = int(native.width() * ratio + 0.5)
        frame = np.arange(height * width * 4, dtype=np.uint8).reshape(height, width, 4)
        captured.append(frame)
        return frame

    viewer = viewer_with_canvas(screenshot, ratio)
    frame = capture_export_frame(viewer, render_size=(720, 1280))
    assert frame.shape == (720, 1280, 4)
    original = captured[0]
    top, left = (original.shape[0] - 720) // 2, (original.shape[1] - 1280) // 2
    np.testing.assert_array_equal(frame, original[top:top + 720, left:left + 1280])
    assert viewer.window._qt_viewer.canvas.native.size == (800, 600)
    assert viewer.camera.zoom == 2.0


def test_wrong_frame_size_fails_and_restores_view():
    viewer = viewer_with_canvas(lambda **kwargs: np.zeros((600, 800, 4)))
    with pytest.raises(RuntimeError, match="canvas returned"):
        capture_export_frame(viewer, render_size=(1080, 1920))
    assert viewer.window._qt_viewer.canvas.native.size == (800, 600)
    assert viewer.camera.zoom == 2.0


@pytest.mark.parametrize("ratio", [1.0, 1.75])
def test_export_session_resizes_once_and_preserves_each_authored_zoom(ratio):
    from math import ceil
    from napari_animation.render_capture import ExportCaptureSession
    observed = []
    def screenshot(**kwargs):
        observed.append(viewer.camera.zoom)
        native = viewer.window._qt_viewer.canvas.native
        return np.zeros((ceil(native.height() * ratio), ceil(native.width() * ratio), 4), np.uint8)
    viewer = viewer_with_canvas(screenshot, ratio)
    native = viewer.window._qt_viewer.canvas.native
    resize = native.resize
    sizes = []
    def resized(*size):
        sizes.append(size)
        resize(*size)
    native.resize = resized
    with ExportCaptureSession(viewer, (720, 1280)):
        for zoom in (2.0, 3.0, 4.0):
            viewer.camera.zoom = zoom
            frame = capture_export_frame(viewer, render_size=(720, 1280))
            assert frame.shape == (720, 1280, 4)
            assert viewer.camera.zoom == zoom
        assert sizes == [(ceil(1280 / ratio), ceil(720 / ratio))]
    assert sizes[-1] == (800, 600)
    assert not hasattr(viewer, "_napari_animation_capture_session")
    np.testing.assert_allclose(observed, np.array([2, 3, 4]) * min(1280 / (800 * ratio), 720 / (600 * ratio)))


def test_export_session_failure_and_scaling_change_restore_view():
    from napari_animation.render_capture import ExportCaptureSession
    def fail(**kwargs):
        raise RuntimeError("failed readback")
    viewer = viewer_with_canvas(fail)
    native = viewer.window._qt_viewer.canvas.native
    for scaling_changed in (False, True):
        native.ratio = 1
        with pytest.raises(RuntimeError):
            with ExportCaptureSession(viewer, (720, 1280)):
                if scaling_changed:
                    native.ratio = 1.75
                capture_export_frame(viewer, render_size=(720, 1280))
        assert native.size == (800, 600)
        assert viewer.camera.zoom == 2
        assert not hasattr(viewer, "_napari_animation_capture_session")
