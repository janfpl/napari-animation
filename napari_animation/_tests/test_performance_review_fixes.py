"""Regressions reproduced in the performance-change review."""

from copy import deepcopy
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication

from napari_animation import Animation, KeyFrame, ViewerState
from napari_animation.layer_registry import LayerRegistry
from napari_animation.scene import OrthoSlice, Scene, SceneRenderer
from napari_animation.scene.processing import service_for_viewer, project_slab


class Viewer(ViewerModel):
    def screenshot(self, *args, **kwargs):
        layers = [layer for layer in self.layers if layer.visible]
        if layers:
            return np.asarray(layers[-1].data).squeeze().copy()
        return np.zeros((16, 16), dtype=np.uint16)


class GuiViewer(Viewer):
    window: ClassVar[bool] = True


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def wait_for_projection(viewer):
    service = service_for_viewer(viewer)
    if service._active is None and service._pending is None:
        return
    loop = QEventLoop()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(5000)
    service.when_idle(loop.quit)
    loop.exec_()
    assert service._active is None and service._pending is None
    assert service.last_error is None


def setup_section(gui=False, multiscale=False):
    viewer = (GuiViewer if gui else Viewer)(ndisplay=2)
    data = np.arange(8 * 16 * 16, dtype=np.uint16).reshape(8, 16, 16)
    viewer.add_image(
        [data, data[::2, ::2, ::2]] if multiscale else data,
        name="volume", multiscale=multiscale,
    )
    section = OrthoSlice(
        name="section", source="volume", source_level=0,
        position=(4, 0, 0), thickness=1,
    )
    scene = Scene()
    scene.append(section)
    renderer = SceneRenderer.for_viewer(viewer, scene)
    renderer.active_section_id = section.id
    return viewer, section, scene, renderer, data


def test_cached_keyframe_switch_publishes_selected_section(monkeypatch):
    viewer, first, _scene, _renderer, _data = setup_section()
    monkeypatch.setattr(Viewer, "screenshot", lambda *a, **k: np.zeros((30, 30, 4), np.uint8))
    animation = Animation(viewer)
    animation.add_scene_object(first)
    second = animation.add_scene_object(OrthoSlice(
        name="second", source="volume", position=(6, 0, 0), thickness=1,
    ))
    try:
        for section in (first, second):
            animation.set_active_section(section.id)
            animation.capture_keyframe(steps=1)
        for index, selected in ((0, first), (1, second), (0, first)):
            animation.set_to_keyframe(index)
            registry = LayerRegistry(viewer)
            assert registry.find_managed(selected.id, "section_2d").visible
            other = second if selected is first else first
            assert not registry.find_managed(other.id, "section_2d").visible
    finally:
        animation.close()


def test_scroll_through_moves_active_section_pixels():
    viewer, section, _scene, _renderer, data = setup_section()
    animation = Animation(viewer)
    animation.add_scene_object(section)
    try:
        animation.add_scroll_through(axis=0, start=0, stop=7, steps=7)
        for index in (0, 3, 7):
            animation.set_movie_frame_index(index)
            output = LayerRegistry(viewer).find_managed(section.id, "section_2d")
            assert viewer.dims.point[0] == index
            np.testing.assert_array_equal(np.squeeze(output.data), data[index])
    finally:
        animation.close()


@pytest.mark.parametrize("action", ("disable", "remove", "empty", "close"))
def test_refinement_cannot_revive_retired_sections(app, monkeypatch, action):
    viewer, section, scene, renderer, _data = setup_section(gui=True, multiscale=True)
    delayed = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, callback: delayed.append(callback))
    renderer.apply_working()
    wait_for_projection(viewer)
    assert delayed
    service = service_for_viewer(viewer)
    try:
        if action == "disable":
            section.enabled = False
            renderer.apply_working()
        elif action == "remove":
            section.remove_layer(viewer)
            scene.remove(section)
            renderer.active_section_id = None
            renderer.apply_working()
        elif action == "empty":
            renderer.apply_snapshot({}, active_section_id=None)
        else:
            service.shutdown()
        for callback in delayed:
            callback()
        wait_for_projection(viewer)
        output = LayerRegistry(viewer).find_managed(section.id, "section_2d")
        if action != "close":
            assert output is None or not output.visible
        # No delayed exact read was started by the retired timer.
        assert not any(key[3] == 0 for key in service.cache._items)
    finally:
        service.shutdown()


def test_cached_snapshot_invalidates_previous_refinement(app, monkeypatch):
    viewer, section, scene, _renderer, data = setup_section(gui=True, multiscale=True)
    delayed = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, callback: delayed.append(callback))
    state_a = ViewerState.from_viewer(
        viewer, scene=scene.to_dict(), view={"active_section_id": section.id},
    )
    scene_b = deepcopy(state_a.scene)
    scene_b[section.id]["position"] = (6, 0, 0)
    state_b = replace(state_a, scene=scene_b)
    service = service_for_viewer(viewer)
    try:
        viewer._napari_animation_export_exact = True
        state_b.apply(viewer)
        wait_for_projection(viewer)
        viewer._napari_animation_export_exact = False
        state_a.apply(viewer)
        state_b.apply(viewer)
        for callback in delayed:
            callback()
        wait_for_projection(viewer)
        output = LayerRegistry(viewer).find_managed(section.id, "section_2d")
        assert viewer.dims.point[0] == 6
        np.testing.assert_array_equal(np.squeeze(output.data), data[6])
    finally:
        service.shutdown()


def test_render_waits_for_exact_multiscale_projection(app, monkeypatch):
    import time
    import napari_animation.scene.processing as processing

    viewer, section, scene, _renderer, data = setup_section(gui=True, multiscale=True)
    state = ViewerState.from_viewer(
        viewer, scene=scene.to_dict(), view={"active_section_id": section.id},
    )
    original = processing.project_slab

    def slow(*args, **kwargs):
        time.sleep(0.02)
        return original(*args, **kwargs)

    monkeypatch.setattr(processing, "project_slab", slow)
    service = service_for_viewer(viewer)
    try:
        actual = state.render(viewer)
        np.testing.assert_array_equal(actual, data[4])
        assert service.asynchronous
        assert not viewer._napari_animation_export_exact
        monkeypatch.setattr(processing, "project_slab", lambda *a, **k: (_ for _ in ()).throw(OSError("read failed")))
        service.cache.invalidate_source(LayerRegistry(viewer).source_id(viewer.layers["volume"]))
        with pytest.raises(OSError, match="read failed"):
            state.render(viewer)
        assert service.asynchronous
        assert not viewer._napari_animation_export_exact
    finally:
        service.shutdown()


def test_animate_uses_exact_projections_with_gui_service(app, monkeypatch):
    import napari_animation.animation as animation_module

    viewer, section, _scene, _renderer, data = setup_section(gui=True, multiscale=True)
    animation = Animation(viewer)
    animation.scene.append(section)
    animation.active_section_id = section.id
    animation.renderer.active_section_id = section.id
    for z in (2, 6):
        section.position = (z, 0, 0)
        animation.key_frames.append(KeyFrame(
            ViewerState.from_viewer(viewer, scene=animation.scene.to_dict(), view={"active_section_id": section.id}),
            np.zeros((30, 30, 4), np.uint8), steps=1,
        ))
    captured = []

    class Writer:
        def append_data(self, frame):
            captured.append(frame.copy())

        def close(self):
            pass

    monkeypatch.setattr(animation_module.imageio, "get_writer", lambda *a, **k: Writer())
    try:
        animation.animate("unused-review-test.mp4", prefetch=0, dask_cache=None, diagnose=False, perf_log=False)
        assert len(captured) == 2
        for frame, z in zip(captured, (2, 6)):
            np.testing.assert_array_equal(frame, data[z])
    finally:
        animation.close()


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_boolean_mean_preserves_fractional_values(axis):
    data = (np.arange(24).reshape(2, 3, 4) % 3) == 0
    actual = project_slab(data, axis, 0, data.shape[axis], "mean")
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.mean(data, axis=axis))


def test_gui_export_matches_exact_opengl_frames(make_napari_viewer, monkeypatch):
    from napari_animation._qt.export_controller import QtExportController
    from napari_animation.video_padding import pad_video_frame

    viewer = make_napari_viewer(ndisplay=2)
    data = np.arange(8 * 16 * 16, dtype=np.uint16).reshape(8, 16, 16)
    viewer.add_image([data, data[::2, ::2, ::2]], name="volume", multiscale=True)
    animation = Animation(viewer)
    section = OrthoSlice(
        name="section", source="volume", source_level=0,
        position=(2, 0, 0), thickness=1,
    )
    animation.scene.append(section)
    animation.active_section_id = section.id
    animation.renderer.active_section_id = section.id
    for z in (2, 6):
        section.position = (z, 0, 0)
        animation.key_frames.append(KeyFrame(
            ViewerState.from_viewer(viewer, scene=animation.scene.to_dict(), view={"active_section_id": section.id}),
            np.zeros((30, 30, 4), np.uint8), steps=1,
        ))
    expected = [state.render(viewer) for state in animation._frames]
    captured = []

    class Writer:
        def append_data(self, frame):
            captured.append(frame.copy())

        def close(self):
            pass

    controller = QtExportController(animation, {
        "path": "unused-opengl-review-test.mp4", "prefetch": 0,
        "dask_cache": None, "diagnose": False,
    })
    monkeypatch.setattr(controller, "_open_writer", Writer)
    loop = QEventLoop()
    results = []
    controller.finished.connect(lambda path, error: (results.append(error), loop.quit()))
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(10000)
    try:
        controller.start()
        loop.exec_()
        assert results == [None]
        assert len(captured) == len(expected) == 2
        for actual, reference in zip(captured, expected):
            np.testing.assert_array_equal(actual, pad_video_frame(reference))
    finally:
        animation.close()
