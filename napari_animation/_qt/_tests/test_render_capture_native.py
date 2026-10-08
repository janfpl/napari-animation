"""Real Qt/OpenGL regressions; run also with QT_SCALE_FACTOR=1.75."""

import os
import time

import dask.array as da
import napari
import numpy as np
import pytest
from napari.settings import get_settings
from qtpy.QtWidgets import QApplication

from napari_animation.render_capture import capture_export_frame

pytestmark = pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") in {"offscreen", "minimal"},
    reason="Requires a native Qt OpenGL context, including actual device scaling.",
)


@pytest.fixture
def native_viewer(qtbot):
    settings = get_settings()
    previous_async = settings.experimental.async_
    settings.experimental.async_ = False
    viewer = napari.Viewer(show=True)
    QApplication.processEvents()
    try:
        yield viewer
    finally:
        viewer.close()
        settings.experimental.async_ = previous_async


@pytest.mark.parametrize("ndisplay", [2, 3])
def test_native_presets_keep_exact_size_and_rerender_detail(
    native_viewer, ndisplay
):
    viewer = native_viewer
    viewer.dims.ndisplay = ndisplay
    shape = (100, 100) if ndisplay == 2 else (32, 32, 32)
    viewer.add_image(
        np.full(shape, 220, dtype=np.uint8), contrast_limits=(0, 255)
    )
    QApplication.processEvents()
    viewer.camera.zoom = 1
    canvas = viewer.window._qt_viewer.canvas
    native = canvas.native
    original_size = (native.width(), native.height())
    original_center = viewer.camera.center
    spans = []
    for size in [(720, 1280), (1440, 2560), (1080, 1920), (1536, 2048)]:
        frame = capture_export_frame(viewer, render_size=size)
        assert frame.shape == (*size, 4)
        rows, columns = np.where(frame[..., :3].max(axis=2) > 128)
        assert rows.size > 0
        spans.append((np.ptp(rows) + 1, np.ptp(columns) + 1))
        assert (native.width(), native.height()) == original_size
        assert viewer.camera.zoom == 1
        assert viewer.camera.center == original_center
    # The image occupies twice as many rendered pixels at twice the size.
    np.testing.assert_allclose(spans[1], np.asarray(spans[0]) * 2, atol=3)


def test_async_auto_multiscale_matches_fresh_synchronous_pixels(
    native_viewer,
    qtbot,
    monkeypatch,
):
    viewer = native_viewer
    get_settings().experimental.async_ = True

    def slow_read(block):
        time.sleep(0.05)
        return block

    levels = [
        da.full((2, n, n), value, chunks=(1, n, n), dtype=np.uint8).map_blocks(
            slow_read, dtype=np.uint8
        )
        for n, value in [(512, 220), (256, 150), (128, 80), (64, 20)]
    ]
    layer = viewer.add_image(levels, multiscale=True, contrast_limits=(0, 255))
    hidden = viewer.add_image(
        levels,
        multiscale=True,
        visible=False,
        contrast_limits=(0, 255),
    )

    def unexpected_hidden_read():
        raise AssertionError("Export must not materialize a hidden layer")

    monkeypatch.setattr(
        hidden._slicing_state, "set_view_slice", unexpected_hidden_read
    )
    viewer.camera.zoom = 0.2
    qtbot.waitUntil(
        lambda: layer.loaded and layer.data_level > 0, timeout=5000
    )
    callbacks = list(layer.events.reload.callbacks)
    first = capture_export_frame(viewer, render_size=(2160, 3840))
    assert not viewer._layer_slicer._force_sync
    assert list(layer.events.reload.callbacks) == callbacks
    get_settings().experimental.async_ = False
    reference = capture_export_frame(viewer, render_size=(2160, 3840))
    np.testing.assert_array_equal(first, reference)
    # The original coarse level contains 80 (or 20), never these fine values.
    assert int(first[1080, 1920, 0]) in (150, 220)


def test_capture_failure_restores_canvas_camera_and_slice_callbacks(
    native_viewer,
    qtbot,
    monkeypatch,
):
    viewer = native_viewer
    layer = viewer.add_image(
        np.full((100, 100), 220, dtype=np.uint8), contrast_limits=(0, 255)
    )
    QApplication.processEvents()
    get_settings().experimental.async_ = True
    native = viewer.window._qt_viewer.canvas.native
    original = (
        (native.width(), native.height()),
        viewer.camera.zoom,
        viewer.camera.center,
        list(layer.events.reload.callbacks),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("capture failed")

    with monkeypatch.context() as patch:
        patch.setattr(napari.Viewer, "screenshot", fail)
        with pytest.raises(RuntimeError, match="capture failed"):
            capture_export_frame(viewer, render_size=(720, 1280))
    assert (
        (native.width(), native.height()),
        viewer.camera.zoom,
        viewer.camera.center,
        list(layer.events.reload.callbacks),
    ) == original
    assert not viewer._layer_slicer._force_sync
    layer.refresh()
    qtbot.waitUntil(lambda: layer.loaded, timeout=5000)
    assert viewer.screenshot(canvas_only=True)[..., :3].max() > 128


def test_native_export_session_matches_individual_frames(native_viewer):
    from napari_animation.render_capture import ExportCaptureSession

    viewer = native_viewer
    viewer.dims.ndisplay = 3
    data = np.zeros((24, 48, 48), dtype=np.uint16)
    data[4:20, 10:35, 15:38] = 2400
    viewer.add_image(data, contrast_limits=(0, 3000), colormap="magenta")
    QApplication.processEvents()
    native = viewer.window._qt_viewer.canvas.native
    before = (native.width(), native.height())
    refs = []
    for zoom in (2.0, 3.0, 4.0):
        viewer.camera.zoom = zoom
        refs.append(capture_export_frame(viewer, render_size=(720, 1280)))
    with ExportCaptureSession(viewer, (720, 1280)):
        for zoom, reference in zip((2.0, 3.0, 4.0), refs):
            viewer.camera.zoom = zoom
            frame = capture_export_frame(viewer, render_size=(720, 1280))
            np.testing.assert_array_equal(frame, reference)
    assert (native.width(), native.height()) == before


def test_native_qt_session_cancel_restores_viewport(
    native_viewer, monkeypatch, tmp_path
):
    import importlib

    from qtpy.QtCore import QEventLoop, QTimer

    from napari_animation import Animation
    from napari_animation._qt.export_controller import QtExportController

    module = importlib.import_module("napari_animation._qt.export_controller")
    viewer = native_viewer
    viewer.add_image(
        np.full((16, 32, 32), 200, np.uint8), contrast_limits=(0, 255)
    )
    animation = Animation(viewer)
    animation.capture_keyframe(steps=1)
    animation.capture_keyframe(steps=4)
    native = viewer.window._qt_viewer.canvas.native
    before = (native.width(), native.height())
    controller = QtExportController(
        animation,
        dict(
            path=str(tmp_path / "frames"),
            render_size=(720, 1280),
            prefetch=0,
            dask_cache=None,
            diagnose=False,
        ),
    )
    original = module.capture_export_frame
    frames = []

    def capture(*args, **kwargs):
        frame = original(*args, **kwargs)
        frames.append(frame.shape)
        QTimer.singleShot(0, controller.cancel)
        return frame

    monkeypatch.setattr(module, "capture_export_frame", capture)
    loop = QEventLoop()
    errors = []
    controller.finished.connect(
        lambda path, error: (errors.append(error), loop.quit())
    )
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    deadline.start(5000)
    try:
        controller.start()
        loop.exec_()
        assert frames == [(720, 1280, 4)]
        assert len(errors) == 1 and str(errors[0]) == "Export cancelled"
        assert (native.width(), native.height()) == before
        assert not hasattr(viewer, "_napari_animation_capture_session")
    finally:
        deadline.stop()
        animation.close()


def test_scaled_clip_slab_selects_authored_world_position(native_viewer):
    from napari_animation.native_clipping import release_native_clipping
    from napari_animation.scene import OrthoSlice, Scene

    viewer = native_viewer
    viewer.dims.ndisplay = 3
    data = np.zeros((20, 40, 40), np.uint16)
    data[5, 10:30, 10:30] = 5000
    layer = viewer.add_image(
        data, name="scaled", scale=(2, 1, 1), contrast_limits=(0, 5000)
    )
    scene = Scene()
    section = OrthoSlice(
        source="scaled", render_mode="clip", position=(10, 20, 20), thickness=2
    )
    scene.append(section)
    scene.apply(viewer)
    QApplication.processEvents()
    frame = capture_export_frame(viewer, render_size=(720, 1280))
    assert frame[..., :3].max() > 128
    # Public napari model remains in data coordinates for editing/saving.
    np.testing.assert_allclose(
        layer.experimental_clipping_planes[0].position, (4.5, 20, 20)
    )
    # A native transform change must also update scene-space shader planes.
    layer.translate = (10, 0, 0)
    node = viewer.window._qt_viewer.canvas.layer_to_visual[layer].node
    if node.clipping_planes_coord_system == "scene":
        np.testing.assert_allclose(node.clipping_planes[0, 0], (20, 20, 19))
    section.position = (34, 20, 20)  # z=12 in the translated data: empty.
    scene.apply(viewer)
    frame = capture_export_frame(viewer, render_size=(720, 1280))
    assert frame[..., :3].max() == 0
    release_native_clipping(viewer)
    assert not hasattr(viewer, "_napari_animation_clip_bindings")


def test_clip_export_converts_nanometers_to_world_micrometers(native_viewer):
    from napari_animation.native_clipping import release_native_clipping

    viewer = native_viewer
    viewer.dims.ndisplay = 3
    layer = viewer.add_image(
        np.full((20, 20, 20), 200, np.uint8),
        scale=(1000, 1000, 1000),
        units=("nm",) * 3,
        contrast_limits=(0, 255),
    )
    viewer.layers.units = ("um",) * 3
    QApplication.processEvents()
    viewer.reset_view()
    layer.experimental_clipping_planes = [
        dict(position=(8, 0, 0), normal=(1, 0, 0)),
        dict(position=(12, 0, 0), normal=(-1, 0, 0)),
    ]
    original = layer.experimental_clipping_planes.as_array().copy()
    try:
        frame = capture_export_frame(viewer, render_size=(720, 1280))
        assert frame.shape == (720, 1280, 4)
        assert frame[..., :3].max() > 128
        node = viewer.window._qt_viewer.canvas.layer_to_visual[layer].node
        np.testing.assert_allclose(node.clipping_planes[:, 0, 2], (8, 12))
        # Change units without pumping Qt: capture must handle the pending
        # canvas unit update as well as already-updated visuals.
        viewer.layers.units = ("nm",) * 3
        viewer.camera.center = tuple(np.asarray(viewer.camera.center) * 1000)
        viewer.camera.zoom /= 1000
        frame = capture_export_frame(viewer, render_size=(720, 1280))
        assert frame[..., :3].max() > 128
        np.testing.assert_allclose(
            node.clipping_planes[:, 0, 2], (8000, 12000)
        )
        np.testing.assert_array_equal(
            layer.experimental_clipping_planes.as_array(), original
        )
    finally:
        release_native_clipping(viewer)
