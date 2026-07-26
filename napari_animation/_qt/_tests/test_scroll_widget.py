"""Tests for the view-snapping and scroll-through panel."""

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation
from napari_animation._qt.scroll_widget import ScrollThroughWidget
from napari_animation.scene import VIEW_NORMALS, ClipPlane, OrthoSlice


class _HeadlessViewer(ViewerModel):
    def screenshot(self, *args, **kwargs):
        return np.zeros((60, 60, 4), dtype=np.uint8)


class _Host:
    def __init__(self, animation):
        self.animation = animation


@pytest.fixture
def animation():
    viewer = _HeadlessViewer()
    viewer.dims.ndisplay = 3
    viewer.add_image(
        np.random.random((10, 20, 30)), name="img", scale=(3, 1, 1)
    )
    return Animation(viewer)


@pytest.fixture
def widget(animation, qtbot, monkeypatch):
    panel = ScrollThroughWidget.__new__(ScrollThroughWidget)
    monkeypatch.setattr(
        ScrollThroughWidget, "parentWidget", lambda self: _Host(animation)
    )
    ScrollThroughWidget.__init__(panel)
    qtbot.addWidget(panel)
    return panel


# ------------------------------------------------------------------ snapping


def test_snap_buttons_change_the_view(widget, animation):
    widget.viewButtons["XZ"].click()

    assert animation.viewer.dims.ndisplay == 2
    assert animation.viewer.dims.order == (1, 0, 2)

    widget.view3dButton.click()
    assert animation.viewer.dims.ndisplay == 3


def test_snapping_does_not_capture_a_keyframe(widget, animation):
    """The buttons set up a view; capturing stays an explicit action."""
    widget.viewButtons["XY"].click()

    assert len(animation.key_frames) == 0


def test_axis_choice_follows_the_snapped_view(widget, animation):
    widget.viewButtons["YZ"].click()

    # YZ shows Z/Y, so X (axis 2) becomes the slider and the default to scroll
    assert widget.axisComboBox.currentData() == 2


# ------------------------------------------------------------ range loading


def test_range_is_prefilled_from_the_data_extent(widget, animation):
    widget.viewButtons["XY"].click()

    # 10 planes at a z scale of 3, measured centre to centre
    assert widget.startSpinBox.value() == pytest.approx(0.0)
    assert widget.stopSpinBox.value() == pytest.approx(27.0)


def test_changing_axis_reloads_the_range(widget, animation):
    index = widget.axisComboBox.findData(2)
    widget.axisComboBox.setCurrentIndex(index)

    # x spans 30 samples at unit scale
    assert widget.stopSpinBox.value() == pytest.approx(29.0)


def test_an_explicitly_chosen_axis_is_not_overridden(widget, animation):
    """Following the view is a default, not something that fights the user."""
    index = widget.axisComboBox.findData(2)
    widget.axisComboBox.setCurrentIndex(index)
    widget.axisComboBox.activated.emit(index)  # as a real click would

    widget.viewButtons["XY"].click()

    assert widget.axisComboBox.currentData() == 2


# ------------------------------------------------------------ scroll-through


def test_scroll_button_adds_two_keyframes(widget, animation):
    widget.viewButtons["XY"].click()
    widget.stepsSpinBox.setValue(12)

    widget.scrollButton.click()

    assert len(animation.key_frames) == 2
    assert animation.key_frames[1].steps == 12
    points = [
        animation._frames[i].dims["point"][0]
        for i in range(len(animation._frames))
    ]
    assert points[0] == pytest.approx(0.0)
    assert points[-1] == pytest.approx(27.0)


def test_scroll_uses_the_edited_range(widget, animation):
    widget.viewButtons["XY"].click()
    widget.startSpinBox.setValue(9.0)
    widget.stopSpinBox.setValue(18.0)

    widget.scrollButton.click()

    points = [
        animation._frames[i].dims["point"][0]
        for i in range(len(animation._frames))
    ]
    assert points[0] == pytest.approx(9.0)
    assert points[-1] == pytest.approx(18.0)


# ------------------------------------------------------------- slice sweeps


def test_sweep_is_disabled_without_a_slice(widget):
    assert widget.sweepButton.isEnabled() is False


def test_slice_list_only_offers_ortho_slices(widget, animation):
    animation.add_scene_object(ClipPlane(name="cut"))
    animation.add_scene_object(OrthoSlice(name="Ortho", source="img"))
    widget.refresh()

    names = [
        widget.sliceComboBox.itemText(i)
        for i in range(widget.sliceComboBox.count())
    ]
    assert names == ["Ortho"]
    assert widget.sweepButton.isEnabled() is True


def test_sweep_button_moves_the_slice_across_the_volume(widget, animation):
    slice_ = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img", normal=VIEW_NORMALS["XY"])
    )
    widget.refresh()
    widget.stepsSpinBox.setValue(8)

    widget.sweepButton.click()

    assert len(animation.key_frames) == 2
    positions = [
        animation._frames[i].scene[slice_.id]["position"][0]
        for i in range(len(animation._frames))
    ]
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(27.0)
    assert positions == sorted(positions)


def test_new_slices_appear_in_the_list(widget, animation):
    assert widget.sliceComboBox.count() == 0

    animation.add_scene_object(OrthoSlice(name="Later", source="img"))
    widget.refresh()

    assert widget.sliceComboBox.count() == 1
