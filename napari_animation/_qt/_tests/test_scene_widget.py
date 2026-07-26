"""Tests for the scene and voxel-size panels.

These drive the widgets the way a user would -- click add, change a spin box,
untick a checkbox -- and assert the effect reaches the viewer. They run
offscreen against a ViewerModel, so no OpenGL context is needed.
"""

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtCore import Qt

from napari_animation import Animation
from napari_animation._qt.scene_widget import ALL_LAYERS, SceneWidget
from napari_animation._qt.voxel_size_widget import VoxelSizeWidget


class _HeadlessViewer(ViewerModel):
    def screenshot(self, *args, **kwargs):
        return np.zeros((60, 60, 4), dtype=np.uint8)


class _Host:
    """Stands in for AnimationWidget, which the panels read `animation` from."""

    def __init__(self, animation):
        self.animation = animation


@pytest.fixture
def animation():
    viewer = _HeadlessViewer()
    viewer.add_image(
        np.random.random((10, 20, 20)), name="img", scale=(1, 1, 1)
    )
    return Animation(viewer)


@pytest.fixture
def scene_widget(animation, qtbot, monkeypatch):
    # the panels look up their animation via parentWidget()
    widget = SceneWidget.__new__(SceneWidget)
    monkeypatch.setattr(
        SceneWidget, "parentWidget", lambda self: _Host(animation)
    )
    SceneWidget.__init__(widget)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def voxel_widget(animation, qtbot, monkeypatch):
    widget = VoxelSizeWidget.__new__(VoxelSizeWidget)
    monkeypatch.setattr(
        VoxelSizeWidget, "parentWidget", lambda self: _Host(animation)
    )
    VoxelSizeWidget.__init__(widget)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------- scene list


def test_add_clipping_plane_cuts_the_volume(scene_widget, animation):
    scene_widget.add_clip_plane()

    assert len(animation.scene) == 1
    assert scene_widget.listWidget.count() == 1
    planes = animation.viewer.layers["img"].experimental_clipping_planes
    assert len(planes) == 1


def test_added_planes_get_unique_names(scene_widget, animation):
    scene_widget.add_clip_plane()
    scene_widget.add_clip_plane()

    names = [obj.name for obj in animation.scene]
    assert names == ["Clipping plane", "Clipping plane 2"]


def test_add_ortho_slice_creates_a_backing_layer(scene_widget, animation):
    scene_widget.add_ortho_slice()

    assert "Ortho slice" in animation.viewer.layers
    assert animation.viewer.layers["Ortho slice"].depiction == "plane"


def test_new_objects_are_placed_at_the_data_centre(scene_widget, animation):
    scene_widget.add_clip_plane()

    # the image spans 0..10 in z at unit scale, so the centre is around 5
    assert 0.0 < animation.scene[0].position[0] < 10.0


def test_unticking_switches_an_object_off(scene_widget, animation):
    scene_widget.add_clip_plane()
    item = scene_widget.listWidget.item(0)

    item.setCheckState(Qt.Unchecked)

    assert animation.scene[0].enabled is False
    assert not animation.viewer.layers["img"].experimental_clipping_planes


def test_renaming_moves_the_backing_layer(scene_widget, animation):
    scene_widget.add_ortho_slice()
    item = scene_widget.listWidget.item(0)

    item.setText("Sagittal")

    assert animation.scene[0].name == "Sagittal"
    assert "Sagittal" in animation.viewer.layers
    assert "Ortho slice" not in animation.viewer.layers


def test_remove_deletes_the_object_and_its_layer(scene_widget, animation):
    scene_widget.add_ortho_slice()
    scene_widget.listWidget.setCurrentRow(0)

    scene_widget.remove_selected()

    assert len(animation.scene) == 0
    assert "Ortho slice" not in animation.viewer.layers
    assert scene_widget.listWidget.count() == 0


def test_selecting_shows_the_matching_editor(scene_widget, animation):
    scene_widget.add_clip_plane()
    scene_widget.add_ortho_slice()

    scene_widget.listWidget.setCurrentRow(0)
    assert scene_widget.editorStack.currentWidget() is (
        scene_widget.clipPlaneEditor
    )

    scene_widget.listWidget.setCurrentRow(1)
    assert scene_widget.editorStack.currentWidget() is (
        scene_widget.orthoSliceEditor
    )


# -------------------------------------------------------------- clip editor


def test_editing_position_moves_the_plane(scene_widget, animation):
    scene_widget.add_clip_plane()
    scene_widget.listWidget.setCurrentRow(0)
    editor = scene_widget.clipPlaneEditor

    editor.positionBoxes[0].setValue(7.0)

    assert animation.scene[0].position[0] == pytest.approx(7.0)
    planes = animation.viewer.layers["img"].experimental_clipping_planes
    assert planes[0].position[0] == pytest.approx(7.0)


def test_orientation_preset_sets_the_normal(scene_widget, animation):
    scene_widget.add_clip_plane()
    scene_widget.listWidget.setCurrentRow(0)

    scene_widget.clipPlaneEditor.orientationComboBox.setCurrentText("-Y")

    assert animation.scene[0].normal == pytest.approx((0.0, -1.0, 0.0))


def test_flip_reverses_the_cut(scene_widget, animation):
    scene_widget.add_clip_plane()
    scene_widget.listWidget.setCurrentRow(0)
    before = animation.scene[0].normal

    scene_widget.clipPlaneEditor.flipButton.click()

    assert animation.scene[0].normal == pytest.approx(
        tuple(-value for value in before)
    )


def test_target_layer_restricts_the_cut(scene_widget, animation):
    animation.viewer.add_image(np.zeros((10, 20, 20)), name="other")
    scene_widget.add_clip_plane()
    scene_widget.listWidget.setCurrentRow(0)

    scene_widget.clipPlaneEditor.targetComboBox.setCurrentText("img")

    assert animation.scene[0].targets == ("img",)
    assert animation.viewer.layers["img"].experimental_clipping_planes
    assert not animation.viewer.layers["other"].experimental_clipping_planes
    # and back to everything
    scene_widget.clipPlaneEditor.targetComboBox.setCurrentText(ALL_LAYERS)
    assert animation.scene[0].targets == ()


# ------------------------------------------------------------- slice editor


def test_editing_thickness_reaches_the_layer(scene_widget, animation):
    scene_widget.add_ortho_slice()
    scene_widget.listWidget.setCurrentRow(0)

    scene_widget.orthoSliceEditor.thicknessSpinBox.setValue(6.0)

    assert animation.scene[0].thickness == pytest.approx(6.0)
    assert animation.viewer.layers["Ortho slice"].plane.thickness == (
        pytest.approx(6.0)
    )


def test_projection_reaches_the_layer(scene_widget, animation):
    scene_widget.add_ortho_slice()
    scene_widget.listWidget.setCurrentRow(0)

    scene_widget.orthoSliceEditor.projectionComboBox.setCurrentText("mean")

    assert animation.viewer.layers["Ortho slice"].rendering == "average"


def test_view_preset_sets_the_slab_normal(scene_widget, animation):
    scene_widget.add_ortho_slice()
    scene_widget.listWidget.setCurrentRow(0)

    scene_widget.orthoSliceEditor.viewComboBox.setCurrentText("YZ")

    assert animation.scene[0].normal == pytest.approx((0.0, 0.0, 1.0))


def test_slice_source_excludes_its_own_layer(scene_widget, animation):
    scene_widget.add_ortho_slice()
    scene_widget.listWidget.setCurrentRow(0)

    sources = [
        scene_widget.orthoSliceEditor.sourceComboBox.itemText(i)
        for i in range(scene_widget.orthoSliceEditor.sourceComboBox.count())
    ]

    assert "Ortho slice" not in sources
    assert "img" in sources


def test_switching_to_clip_mode_removes_the_extra_layer(
    scene_widget, animation
):
    scene_widget.add_ortho_slice()
    scene_widget.listWidget.setCurrentRow(0)

    scene_widget.orthoSliceEditor.renderComboBox.setCurrentText("clip")

    # clip mode carves the source volume instead of drawing a slab layer
    assert animation.viewer.layers["Ortho slice"].visible is False
    assert (
        len(animation.viewer.layers["img"].experimental_clipping_planes) == 2
    )


def test_manipulator_button_reflects_availability(scene_widget):
    from napari_animation.scene._manipulators import is_available

    button = scene_widget.orthoSliceEditor.manipulatorButton
    assert button.isEnabled() == is_available()
    if not is_available():
        assert "napari-threedee" in button.toolTip()


# ---------------------------------------------------------------- voxel size


def test_voxel_widget_shows_the_current_spacing(voxel_widget, animation):
    animation.viewer.layers["img"].scale = (3.0, 0.5, 0.5)
    voxel_widget._load_from_layer()

    assert voxel_widget.spinBoxes["Z"].value() == pytest.approx(3.0)
    assert voxel_widget.spinBoxes["X"].value() == pytest.approx(0.5)


def test_applying_voxel_size_corrects_the_layer(voxel_widget, animation):
    voxel_widget.spinBoxes["Z"].setValue(4.0)
    voxel_widget.spinBoxes["Y"].setValue(0.325)
    voxel_widget.spinBoxes["X"].setValue(0.325)

    voxel_widget._apply()

    assert tuple(animation.viewer.layers["img"].scale) == pytest.approx(
        (4.0, 0.325, 0.325)
    )


def test_voxel_size_is_corrected_in_existing_keyframes(
    voxel_widget, animation
):
    """A wrong Z step is a data problem, not something to keyframe."""
    animation.capture_keyframe()
    animation.capture_keyframe(steps=5)

    voxel_widget.spinBoxes["Z"].setValue(4.0)
    voxel_widget._apply()

    # replaying an old keyframe must not restore the wrong spacing
    animation.key_frames[0].viewer_state.apply(animation.viewer)
    assert animation.viewer.layers["img"].scale[0] == pytest.approx(4.0)


def test_apply_to_all_layers(voxel_widget, animation):
    animation.viewer.add_image(np.zeros((10, 20, 20)), name="ch2")
    voxel_widget._refresh_layers()
    voxel_widget.spinBoxes["Z"].setValue(2.5)

    voxel_widget.allLayersCheckBox.setChecked(True)
    voxel_widget._apply()

    assert animation.viewer.layers["img"].scale[0] == pytest.approx(2.5)
    assert animation.viewer.layers["ch2"].scale[0] == pytest.approx(2.5)


def test_apply_to_single_layer_only(voxel_widget, animation):
    animation.viewer.add_image(np.zeros((10, 20, 20)), name="ch2")
    voxel_widget._refresh_layers()
    voxel_widget.layerComboBox.setCurrentText("img")
    voxel_widget.spinBoxes["Z"].setValue(2.5)

    voxel_widget.allLayersCheckBox.setChecked(False)
    voxel_widget._apply()

    assert animation.viewer.layers["img"].scale[0] == pytest.approx(2.5)
    assert animation.viewer.layers["ch2"].scale[0] == pytest.approx(1.0)


def test_zero_voxel_size_is_rejected(animation):
    with pytest.raises(ValueError, match="non-zero"):
        animation.set_voxel_size("img", (0.0, 1.0, 1.0))


def test_voxel_size_updates_ortho_slice_thickness(scene_widget, animation):
    """Slab thickness is in world units, so it must track the voxel size."""
    scene_widget.add_ortho_slice()
    scene_widget.listWidget.setCurrentRow(0)
    scene_widget.orthoSliceEditor.thicknessSpinBox.setValue(6.0)
    assert animation.viewer.layers["Ortho slice"].plane.thickness == (
        pytest.approx(6.0)
    )

    # 6 world units is now only 2 planes at a Z spacing of 3
    animation.set_voxel_size("img", (3.0, 1.0, 1.0))
    animation.set_voxel_size("Ortho slice", (3.0, 1.0, 1.0))
    scene_widget.apply()

    assert animation.viewer.layers["Ortho slice"].plane.thickness == (
        pytest.approx(2.0)
    )
