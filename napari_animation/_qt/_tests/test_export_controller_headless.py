"""Qt event-loop export without an OpenGL canvas."""

from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication

from napari_animation import Animation
from napari_animation._qt.export_controller import QtExportController


class HeadlessViewer(ViewerModel):
    def screenshot(self, *args, **kwargs):
        value = (int(round(self.dims.point[0])) * 60) % 256
        frame = np.full((16, 16, 4), value, dtype=np.uint8)
        frame[..., 3] = 255
        return frame


def test_export_pump_keeps_qt_alive_and_orders_frames():
    app = QApplication.instance() or QApplication([])
    viewer = HeadlessViewer(ndisplay=2)
    viewer.add_image(np.zeros((4, 16, 16), dtype=np.uint8), name="volume")
    animation = Animation(viewer)
    viewer.dims.set_point(0, 0)
    animation.capture_keyframe(steps=1)
    viewer.dims.set_point(0, 3)
    animation.capture_keyframe(steps=3)
    before = tuple(viewer.dims.point)

    output = Path(".performance-audit/export-pump-test").resolve()
    controller = QtExportController(
        animation,
        {"path": str(output), "prefetch": 0, "dask_cache": None,
         "diagnose": False},
    )
    loop = QEventLoop()
    result = []
    heartbeats = []
    pulse = QTimer()
    pulse.setInterval(5)
    pulse.timeout.connect(lambda: heartbeats.append(1))
    pulse.start()
    controller.finished.connect(lambda path, error: (result.append((path, error)), loop.quit()))
    QTimer.singleShot(5000, loop.quit)
    controller.start()
    loop.exec_()
    pulse.stop()

    assert result and result[0][1] is None
    files = sorted(output.glob("*.png"))
    assert len(files) == 4
    assert int(iio.imread(files[0])[0, 0, 0]) == 0
    assert int(iio.imread(files[-1])[0, 0, 0]) == 180
    assert tuple(viewer.dims.point) == before
    assert viewer._napari_animation_export_exact is False
    import json
    performance = json.loads((output / "render_perf.json").read_text())
    assert performance["metadata"]["status"] == "complete"
    assert performance["counters"]["frames_written"] == 4
    assert performance["phases"]["screenshot"]["count"] == 4
    assert {event["frame"] for event in performance["events"]
            if event["phase"] == "encode/write"} == {0, 1, 2, 3}
    assert heartbeats
    assert app is not None


def test_full_writer_queue_does_not_block_qt(monkeypatch):
    import time

    app = QApplication.instance() or QApplication([])
    original = iio.imwrite

    def slow_write(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(iio, "imwrite", slow_write)
    viewer = HeadlessViewer(ndisplay=2)
    viewer.add_image(np.zeros((10, 16, 16), dtype=np.uint8))
    animation = Animation(viewer)
    animation.capture_keyframe(steps=1)
    viewer.dims.set_point(0, 9)
    animation.capture_keyframe(steps=9)
    output = Path(".performance-audit/export-pump-slow-test").resolve()
    controller = QtExportController(
        animation, {"path": str(output), "prefetch": 0,
                    "dask_cache": None, "diagnose": False},
    )
    loop = QEventLoop()
    ticks = []
    result = []
    pulse = QTimer()
    pulse.setInterval(10)
    pulse.timeout.connect(lambda: ticks.append(time.perf_counter()))
    pulse.start()
    controller.finished.connect(lambda path, error: (result.append(error), loop.quit()))
    QTimer.singleShot(5000, loop.quit)
    controller.start()
    loop.exec_()
    pulse.stop()
    assert result == [None]
    assert len(list(output.glob("*.png"))) == 10
    assert len(ticks) >= 20
    assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.1
    assert app is not None


def test_video_export_pads_frames_without_resampling(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])

    class OddSizedViewer(HeadlessViewer):
        def screenshot(self, *args, **kwargs):
            frame = np.full((18, 17, 4), 73, dtype=np.uint8)
            frame[..., 3] = 255
            return frame

    viewer = OddSizedViewer(ndisplay=2)
    viewer.add_image(np.zeros((2, 18, 17), dtype=np.uint8))
    animation = Animation(viewer)
    animation.capture_keyframe(steps=1)
    animation.capture_keyframe(steps=1)
    controller = QtExportController(animation, {
        "path": str(tmp_path / "padded.mp4"),
        "prefetch": 0, "dask_cache": None, "diagnose": False,
    })
    captured = []

    class Writer:
        def append_data(self, frame):
            captured.append(frame.copy())

        def close(self):
            pass

    monkeypatch.setattr(controller, "_open_writer", Writer)
    results = []
    loop = QEventLoop()
    controller.finished.connect(
        lambda _path, error: (results.append(error), loop.quit())
    )
    QTimer.singleShot(5000, loop.quit)
    try:
        controller.start()
        loop.exec_()
        assert results == [None]
        assert len(captured) == 2
        for frame in captured:
            assert frame.shape == (32, 32, 4)
            assert np.all(frame[:18, :17, :3] == 73)
            assert np.all(frame[18:, :, :3] == 0)
            assert np.all(frame[:, 17:, :3] == 0)
        assert app is not None
    finally:
        animation.close()


def test_preset_renders_at_requested_size_and_preserves_video_dimensions(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    screenshots = []

    class SizedViewer(HeadlessViewer):
        def screenshot(self, *args, **kwargs):
            if "size" in kwargs:
                screenshots.append(kwargs)
            height, width = kwargs.get("size", (16, 16))
            frame = np.full((height, width, 4), 91, dtype=np.uint8)
            frame[..., 3] = 255
            return frame

    viewer = SizedViewer(ndisplay=2)
    viewer.add_image(np.zeros((2, 18, 30), dtype=np.uint8))
    animation = Animation(viewer)
    animation.capture_keyframe(steps=1)
    animation.capture_keyframe(steps=1)
    controller = QtExportController(animation, {
        "path": str(Path(".performance-audit/preset-dimensions.mp4").resolve()),
        "render_size": (18, 30),
        "prefetch": 0, "dask_cache": None, "diagnose": False,
    })
    frames = []

    class Writer:
        def append_data(self, frame):
            frames.append(frame.copy())

        def close(self):
            pass

    monkeypatch.setattr(controller, "_open_writer", Writer)
    results = []
    loop = QEventLoop()
    controller.finished.connect(
        lambda _path, error: (results.append(error), loop.quit())
    )
    QTimer.singleShot(5000, loop.quit)
    try:
        controller.start()
        loop.exec_()
        assert results == [None]
        assert screenshots == [
            {"canvas_only": True, "size": (18, 30), "flash": False},
            {"canvas_only": True, "size": (18, 30), "flash": False},
        ]
        assert len(frames) == 2
        assert all(frame.shape == (18, 30, 4) for frame in frames)
        assert all(np.all(frame[..., :3] == 91) for frame in frames)
        assert app is not None
    finally:
        animation.close()


def test_unavailable_video_writer_falls_back_to_png(monkeypatch):
    app = QApplication.instance() or QApplication([])
    viewer = HeadlessViewer(ndisplay=2)
    viewer.add_image(np.zeros((3, 16, 16), dtype=np.uint8))
    animation = Animation(viewer)
    animation.capture_keyframe(steps=1)
    viewer.dims.set_point(0, 2)
    animation.capture_keyframe(steps=2)
    path = Path(".performance-audit/export-pump-fallback.mp4").resolve()
    controller = QtExportController(
        animation, {"path": str(path), "prefetch": 0,
                    "dask_cache": None, "diagnose": False},
    )
    monkeypatch.setattr(
        controller, "_open_writer",
        lambda: (_ for _ in ()).throw(RuntimeError("encoder unavailable")),
    )
    loop = QEventLoop()
    results = []
    controller.finished.connect(lambda output, error: (results.append((output, error)), loop.quit()))
    QTimer.singleShot(5000, loop.quit)
    controller.start()
    loop.exec_()
    assert results and results[0][1] is None
    assert results[0][0] == path.with_suffix("")
    assert len(list(results[0][0].glob("*.png"))) == 3
    assert app is not None


def test_video_verification_runs_off_qt_thread(monkeypatch):
    import threading
    import time

    app = QApplication.instance() or QApplication([])
    viewer = HeadlessViewer(ndisplay=2)
    viewer.add_image(np.zeros((2, 16, 16), dtype=np.uint8))
    animation = Animation(viewer)
    animation.capture_keyframe(steps=1)
    viewer.dims.set_point(0, 1)
    animation.capture_keyframe(steps=1)

    class Writer:
        def append_data(self, frame):
            pass

        def close(self):
            pass

    controller = QtExportController(animation, {
        "path": str(Path(".performance-audit/verify-thread.mp4").resolve()),
        "prefetch": 0, "dask_cache": None,
    })
    monkeypatch.setattr(controller, "_open_writer", Writer)
    monkeypatch.setattr(controller._diagnostics, "describe_environment", lambda: None)
    monkeypatch.setattr(controller._diagnostics, "report", lambda: "verified")
    verify_threads = []

    def verify(*_args):
        verify_threads.append(threading.get_ident())
        time.sleep(0.2)

    monkeypatch.setattr(controller._diagnostics, "verify_output", verify)
    gui_thread = threading.get_ident()
    ticks = []
    pulse = QTimer()
    pulse.setInterval(10)
    pulse.timeout.connect(lambda: ticks.append(time.perf_counter()))
    pulse.start()
    loop = QEventLoop()
    results = []
    controller.finished.connect(
        lambda _path, error: (results.append(error), loop.quit())
    )
    QTimer.singleShot(5000, loop.quit)
    controller.start()
    loop.exec_()
    pulse.stop()
    assert results == [None]
    assert len(verify_threads) == 1
    assert verify_threads[0] != gui_thread
    assert len(ticks) >= 10
    assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.1
    assert app is not None


def test_export_restores_frame_after_edits_and_uses_frozen_timeline(monkeypatch):
    app = QApplication.instance() or QApplication([])
    viewer = HeadlessViewer(ndisplay=2)
    viewer.add_image(np.zeros((4, 16, 16), dtype=np.uint8), name="volume")
    animation = Animation(viewer)
    viewer.dims.set_point(0, 0)
    animation.capture_keyframe(steps=1)
    viewer.dims.set_point(0, 3)
    animation.capture_keyframe(steps=3)
    before = tuple(viewer.dims.point)
    controller = QtExportController(animation, {
        "path": str(Path(".performance-audit/review-frozen-export.mp4").resolve()),
        "prefetch": 0, "dask_cache": None, "diagnose": False,
    })
    captured = []

    class Writer:
        def append_data(self, frame):
            captured.append(int(frame[0, 0, 0]))

        def close(self):
            pass

    monkeypatch.setattr(controller, "_open_writer", Writer)
    original_wait = controller._await_capture
    edited = []

    def edit_before_capture(generation):
        if not edited:
            edited.append(True)
            viewer.dims.set_point(0, 3)
            viewer.camera.zoom *= 2
            animation.set_keyframe_timing(1, steps=1)
        original_wait(generation)

    monkeypatch.setattr(controller, "_await_capture", edit_before_capture)
    loop = QEventLoop()
    results = []
    controller.finished.connect(lambda path, error: (results.append(error), loop.quit()))
    QTimer.singleShot(5000, loop.quit)
    try:
        controller.start()
        loop.exec_()
        assert results == [None]
        assert captured == [0, 60, 120, 180]
        assert tuple(viewer.dims.point) == before
        assert len(animation._frames) == 2
        assert app is not None
    finally:
        animation.close()


def test_export_progress_blocks_viewer_edits_and_cancel_restores():
    from types import SimpleNamespace
    from typing import ClassVar
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QWidget

    app = QApplication.instance() or QApplication([])
    qt_window = QWidget()
    stopped = []

    class WindowViewer(HeadlessViewer):
        window: ClassVar = SimpleNamespace(
            _qt_window=qt_window,
            _qt_viewer=SimpleNamespace(dims=SimpleNamespace(stop=lambda: stopped.append(True))),
        )

    viewer = WindowViewer(ndisplay=2)
    viewer.add_image(np.zeros((2, 16, 16), np.uint8))
    animation = Animation(viewer)
    animation.capture_keyframe(steps=1)
    animation.capture_keyframe(steps=1)
    controller = QtExportController(animation, {
        "path": str(Path(".performance-audit/review-progress-export").resolve()),
        "prefetch": 0, "dask_cache": None, "diagnose": False,
    })
    loop = QEventLoop()
    results = []
    controller.finished.connect(lambda path, error: (results.append(error), loop.quit()))
    QTimer.singleShot(5000, loop.quit)
    try:
        controller.start()
        progress = controller._progress
        assert progress.windowModality() == Qt.WindowModal
        assert stopped == [True]
        progress.canceled.emit()
        loop.exec_()
        assert len(results) == 1 and str(results[0]) == "Export cancelled"
        assert controller._progress is None
        assert not viewer._napari_animation_export_exact
        assert app is not None
    finally:
        animation.close()
        qt_window.close()


@pytest.mark.parametrize("edit", ("add", "remove", "replace"))
def test_export_reports_source_changes_before_capture(monkeypatch, edit):
    app = QApplication.instance() or QApplication([])
    viewer = HeadlessViewer(ndisplay=2)
    source = viewer.add_image(np.zeros((4, 16, 16), dtype=np.uint8))
    animation = Animation(viewer)
    animation.capture_keyframe(steps=1)
    viewer.dims.set_point(0, 3)
    animation.capture_keyframe(steps=3)
    controller = QtExportController(animation, {
        "path": str(Path(".performance-audit/review-source-edit.mp4").resolve()),
        "prefetch": 0, "dask_cache": None, "diagnose": False,
    })
    captured = []

    class Writer:
        def append_data(self, frame):
            captured.append(frame)

        def close(self):
            pass

    monkeypatch.setattr(controller, "_open_writer", Writer)
    original_wait = controller._await_capture

    def edit_before_capture(generation):
        if edit == "add":
            viewer.add_image(np.ones((4, 16, 16), np.uint8))
        elif edit == "remove":
            viewer.layers.remove(source)
        else:
            source.data = np.ones((4, 16, 16), np.uint8)
        original_wait(generation)

    monkeypatch.setattr(controller, "_await_capture", edit_before_capture)
    results = []
    loop = QEventLoop()
    controller.finished.connect(lambda path, error: (results.append(error), loop.quit()))
    QTimer.singleShot(5000, loop.quit)
    try:
        controller.start()
        loop.exec_()
        assert len(results) == 1
        assert "Source layers changed" in str(results[0])
        assert not captured
        assert not viewer._napari_animation_export_exact
        assert app is not None
    finally:
        animation.close()
