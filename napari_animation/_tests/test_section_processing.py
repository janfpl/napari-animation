"""Numerical and cancellation checks for interactive section processing."""

import time

import numpy as np
import pytest

from napari_animation.scene.processing import ProjectionService, project_slab


class CountingVolume:
    def __init__(self, data, delay=0):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.reads = 0
        self.delay = delay

    def __getitem__(self, key):
        self.reads += 1
        if self.delay:
            time.sleep(self.delay)
        return self.data[key]


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("projection", ("max", "min", "mean"))
@pytest.mark.parametrize("dtype", (np.uint16, np.float32))
def test_bounded_projection_matches_numpy(
    axis, projection, dtype, monkeypatch
):
    import napari_animation.scene.processing as processing

    monkeypatch.setattr(processing, "_TARGET_READ_BYTES", 100)
    data = np.arange(5 * 6 * 7, dtype=dtype).reshape(5, 6, 7)
    data = data + np.arange(5, dtype=dtype)[:, None, None]
    volume = CountingVolume(data)
    actual = project_slab(volume, axis, 1, 4, projection)
    key = [slice(None)] * 3
    key[axis] = slice(1, 4)
    expected = getattr(np, projection)(data[tuple(key)], axis=axis)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
    assert volume.reads >= 2


def test_cached_projection_is_reused_and_invalidated():
    volume = CountingVolume(np.arange(5 * 6 * 7).reshape(5, 6, 7))
    service = ProjectionService()
    values = []
    key = ("source", 0, id(volume), 0, (), 0, 1, 3, "max")
    for _ in range(2):
        service.request("section", key, volume, 0, 1, 3, "max", values.append)
    assert volume.reads == 1
    assert values[0] is values[1]
    service.invalidate_source("source")
    service.request("section", key, volume, 0, 1, 3, "max", values.append)
    assert volume.reads == 2


def test_async_scrub_only_publishes_latest():
    from qtpy.QtCore import QEventLoop, QTimer
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    volume = CountingVolume(np.arange(6 * 7 * 8).reshape(6, 7, 8), delay=0.03)
    service = ProjectionService(asynchronous=True)
    published = []
    loop = QEventLoop()
    for request_index in range(100):
        start = request_index % 5
        key = (
            "source",
            0,
            id(volume),
            request_index,
            (),
            0,
            start,
            start + 1,
            "max",
        )
        service.request(
            "section",
            key,
            volume,
            0,
            start,
            start + 1,
            "max",
            lambda result, i=request_index: published.append((i, result)),
        )
    service.when_idle(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    loop.exec_()
    assert [index for index, _ in published] == [99]
    np.testing.assert_array_equal(published[0][1], volume.data[4])
    service.shutdown()
    assert app is not None


def test_async_identical_requests_share_one_read():
    from qtpy.QtCore import QEventLoop, QTimer
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    volume = CountingVolume(np.ones((4, 5, 6)), delay=0.03)
    service = ProjectionService(asynchronous=True)
    published = []
    key = ("source", 0, id(volume), 0, (), 0, 1, 2, "max")
    service.request("section", key, volume, 0, 1, 2, "max", published.append)
    service.request("section", key, volume, 0, 1, 2, "max", published.append)
    loop = QEventLoop()
    service.when_idle(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    loop.exec_()
    assert len(published) == 1
    assert volume.reads == 1
    service.shutdown()
    assert app is not None


def test_process_start_failure_falls_back_to_worker(monkeypatch):
    from qtpy.QtCore import QEventLoop, QTimer
    from qtpy.QtWidgets import QApplication

    import napari_animation.scene.h5_process as backend

    app = QApplication.instance() or QApplication([])
    volume = CountingVolume(np.arange(4 * 5 * 6).reshape(4, 5, 6))
    monkeypatch.setattr(
        backend,
        "describe_h5_volume",
        lambda _volume: (
            "missing.ims",
            "/Data",
            volume.shape,
            volume.dtype.str,
            (2, 5, 6),
            "gzip",
        ),
    )
    monkeypatch.setattr(backend, "process_lanes", lambda *_args: 4)

    class FailingProjector:
        def __init__(self, *_args):
            raise PermissionError("process spawn unavailable")

    monkeypatch.setattr(backend, "H5ProcessProjector", FailingProjector)
    service = ProjectionService(asynchronous=True)
    published = []
    key = ("source", 0, id(volume), 0, (), 0, 1, 2, "max")
    loop = QEventLoop()

    def submit():
        service.request(
            "section", key, volume, 0, 1, 2, "max", published.append
        )
        service.when_idle(loop.quit)

    QTimer.singleShot(0, submit)
    QTimer.singleShot(5000, loop.quit)
    loop.exec_()
    np.testing.assert_array_equal(published[0], volume.data[1])
    assert service._process_failed
    assert volume.reads == 1
    service.shutdown()
    assert app is not None


def test_process_readers_close_on_source_switch_and_shutdown(monkeypatch):
    import threading

    import napari_animation.scene.h5_process as backend

    created = []

    class Projector:
        def __init__(self, descriptor, lanes):
            self.descriptor = descriptor
            self.closed = False
            created.append(self)

        def project(self, *_args):
            return np.zeros((5, 6), dtype=np.uint16)

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr(backend, "H5ProcessProjector", Projector)
    monkeypatch.setattr(backend, "process_lanes", lambda *_args: 4)
    monkeypatch.setattr(
        backend,
        "describe_h5_volume",
        lambda volume: (
            volume.source,
            "/Data",
            volume.shape,
            volume.dtype.str,
            (2, 5, 6),
            "gzip",
        ),
    )
    service = ProjectionService()
    for index in range(2):
        volume = CountingVolume(np.zeros((4, 5, 6), dtype=np.uint16))
        volume.source = f"source-{index}.ims"
        request = [
            (volume.source, 0, id(volume)),
            "section",
            index,
            threading.Event(),
            volume,
            0,
            1,
            2,
            "max",
            None,
        ]
        assert service._project(request).shape == (5, 6)
    assert len(created) == 2
    assert created[0].closed
    assert not created[1].closed
    service.shutdown()
    assert created[1].closed


def test_removed_section_cannot_publish_late_result():
    from qtpy.QtCore import QEventLoop, QTimer
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    volume = CountingVolume(np.ones((4, 5, 6)), delay=0.03)
    service = ProjectionService(asynchronous=True)
    published = []
    key = ("source", 0, id(volume), 0, (), 0, 1, 2, "max")
    service.request("section", key, volume, 0, 1, 2, "max", published.append)
    service.cancel_section("section")
    loop = QEventLoop()
    service.when_idle(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    loop.exec_()
    assert not published
    service.shutdown()
    assert app is not None


def test_multiscale_section_previews_then_refines():
    from typing import ClassVar

    from napari.components import ViewerModel
    from qtpy.QtCore import QEventLoop, QTimer
    from qtpy.QtWidgets import QApplication

    from napari_animation.layer_registry import LayerRegistry
    from napari_animation.scene import OrthoSlice, Scene

    class FauxGuiViewer(ViewerModel):
        window: ClassVar[bool] = True  # same request path as a Qt viewer

    app = QApplication.instance() or QApplication([])
    viewer = FauxGuiViewer(ndisplay=2)
    full = np.arange(8 * 16 * 16, dtype=np.uint16).reshape(8, 16, 16)
    coarse = full[::2, ::2, ::2]
    viewer.add_image([full, coarse], name="volume", multiscale=True)
    section = OrthoSlice(
        name="section",
        source="volume",
        source_level=0,
        position=(4, 0, 0),
        thickness=2,
    )
    scene = Scene()
    scene.append(section)
    scene.apply(viewer)
    observed = []
    loop = QEventLoop()

    def remember():
        layer = LayerRegistry(viewer).find_managed(section.id, "section_2d")
        observed.append(
            None if layer is None else tuple(np.squeeze(layer.data).shape)
        )

    QTimer.singleShot(100, remember)
    QTimer.singleShot(800, loop.quit)
    loop.exec_()
    remember()
    assert observed == [(8, 8), (16, 16)]
    service = viewer._napari_animation_projection_service
    service.shutdown()
    assert app is not None


def test_reopening_animation_creates_fresh_projection_service():
    from napari.components import ViewerModel

    from napari_animation import Animation
    from napari_animation.scene.processing import service_for_viewer

    viewer = ViewerModel()
    first = Animation(viewer)
    old_service = service_for_viewer(viewer)
    first.close()
    second = Animation(viewer)
    assert service_for_viewer(viewer) is not old_service
    second.close()


def test_cached_queued_request_does_not_strand_the_queue():
    """A queued read answered from the cache must not stall reads behind it."""
    from qtpy.QtCore import QEventLoop, QTimer
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    volume = CountingVolume(np.arange(4 * 5 * 6).reshape(4, 5, 6), delay=0.03)
    service = ProjectionService(asynchronous=True)
    published = []
    shared = ("source", 0, id(volume), 0, (), 0, 1, 2, "max")
    other = ("source", 0, id(volume), 0, (), 0, 2, 3, "max")
    service.begin_batch()
    service.request(
        "first",
        shared,
        volume,
        0,
        1,
        2,
        "max",
        lambda _result: published.append("first"),
    )
    # another section showing the same slab is answered from the cache once
    # the first read completes; the read queued behind it must still run
    service.request(
        "second",
        shared,
        volume,
        0,
        1,
        2,
        "max",
        lambda _result: published.append("second"),
    )
    service.request(
        "third",
        other,
        volume,
        0,
        2,
        3,
        "max",
        lambda _result: published.append("third"),
    )
    service.end_batch()
    loop = QEventLoop()
    service.when_idle(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    loop.exec_()
    assert published == ["first", "second", "third"]
    assert volume.reads == 2
    service.shutdown()
    assert app is not None
