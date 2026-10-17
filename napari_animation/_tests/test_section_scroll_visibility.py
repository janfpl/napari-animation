"""Scrolling an active 2D section must not hide it during projection reads."""

import threading

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation
from napari_animation.scene import OrthoSlice
from napari_animation.scene.processing import (
    exact_render,
    project_slab,
    service_for_viewer,
)


def setup_section(viewer, axis=0):
    data = np.arange(8 * 10 * 12, dtype=np.uint16).reshape(8, 10, 12)
    source = viewer.add_image(data, name="volume", scale=(0.7, 0.3, 0.2))
    animation = Animation(viewer)
    position = np.asarray((2, 2, 2)) * source.scale
    normal = np.eye(3)[axis]
    with exact_render(viewer):
        section = animation.add_scene_object(
            OrthoSlice(
                source=source.name,
                position=tuple(position),
                normal=tuple(normal),
                thickness=float(source.scale[axis]),
            )
        )
        animation.set_active_section(section.id)
        animation.snap_2d(capture=False)
    output = animation.layer_registry.find_managed(section.id, "section_2d")
    return animation, section, source, output


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_slider_keeps_last_projection_visible_until_replacement(
    axis, monkeypatch
):
    viewer = ViewerModel(ndisplay=2)
    animation, section, source, output = setup_section(viewer, axis)
    service = service_for_viewer(viewer)
    previous = np.asarray(output._data_view).copy()
    requests, visibility = [], []
    output.events.visible.connect(
        lambda event: visibility.append(output.visible)
    )
    monkeypatch.setattr(
        service, "request", lambda *args, **kwargs: requests.append(args)
    )
    try:
        for step in (3, 5, 4):
            viewer.dims.set_point(axis, step * source.scale[axis])
            assert output.visible
            assert not output._slice.empty
            np.testing.assert_array_equal(output._data_view, previous)
            assert section.enabled
            assert not source.visible
        assert visibility == []
        # Completing an obsolete read must neither reveal it nor move the slider.
        old, latest = requests[0], requests[-1]
        old[-1](project_slab(*old[2:7]))
        assert viewer.dims.point[axis] == pytest.approx(4 * source.scale[axis])
        np.testing.assert_array_equal(output._data_view, previous)
        latest[-1](project_slab(*latest[2:7]))
        np.testing.assert_array_equal(
            output._data_view, section.projection_oracle(viewer)
        )
        assert output.visible and visibility == []
    finally:
        animation.close()


def test_cached_scroll_never_emits_visibility_toggles():
    viewer = ViewerModel(ndisplay=2)
    animation, section, source, output = setup_section(viewer)
    visibility = []
    output.events.visible.connect(
        lambda event: visibility.append(output.visible)
    )
    try:
        for step in (3, 2, 3, 2):
            viewer.dims.set_point(0, step * source.scale[0])
            np.testing.assert_array_equal(
                output._data_view, section.projection_oracle(viewer)
            )
        assert visibility == []
    finally:
        animation.close()


@pytest.mark.parametrize("transition", ["disable", "3d", "other", "outside"])
def test_section_still_hides_when_it_should(transition):
    viewer = ViewerModel(ndisplay=2)
    animation, section, source, output = setup_section(viewer)
    try:
        if transition == "disable":
            section.enabled = False
        elif transition == "3d":
            viewer.dims.ndisplay = 3
        elif transition == "other":
            other = animation.add_scene_object(
                OrthoSlice(
                    source=source.name,
                    position=section.position,
                )
            )
            animation.set_active_section(other.id)
        else:
            section.position = (100, 0, 0)
        animation.apply_working_state()
        assert not output.visible
    finally:
        animation.close()


def test_native_gui_preserves_section_while_read_is_blocked(
    make_napari_viewer, qtbot, monkeypatch
):
    viewer = make_napari_viewer(ndisplay=2)
    animation, section, source, output = setup_section(viewer)
    service = service_for_viewer(viewer)
    started, release = threading.Event(), threading.Event()
    original = service._project
    previous = np.asarray(output._data_view).copy()
    visibility = []
    output.events.visible.connect(
        lambda event: visibility.append(output.visible)
    )

    def delayed(request):
        started.set()
        assert release.wait(5)
        return original(request)

    monkeypatch.setattr(service, "_project", delayed)
    try:
        viewer.dims.set_current_step(0, 4)  # Built-in napari slider operation.
        qtbot.waitUntil(started.is_set, timeout=5000)
        qtbot.wait(50)  # Allow painting while the worker is still blocked.
        assert output.visible and not output._slice.empty
        np.testing.assert_array_equal(output._data_view, previous)
        assert visibility == []
        release.set()
        expected = section.projection_oracle(viewer)
        qtbot.waitUntil(
            lambda: np.array_equal(output._data_view, expected), timeout=5000
        )
        assert output.visible and visibility == []
    finally:
        release.set()
        animation.close()
