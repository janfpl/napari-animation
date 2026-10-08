"""Resolution controls must agree with rendered pixels at every keyframe."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation
from napari_animation._qt import AnimationWidget
from napari_animation.scene import OrthoSlice


def pyramid(viewer, name="volume"):
    fine = np.arange(8 * 16 * 16, dtype=np.uint16).reshape(8, 16, 16)
    return viewer.add_image(
        [fine, np.full((4, 8, 8), 9000, dtype=np.uint16)],
        multiscale=True, name=name, contrast_limits=(0, 10000),
    )


def test_layer_controls_resolution_can_apply_to_all_layers(qtbot):
    from napari._qt.layer_controls import QtLayerControlsContainer

    viewer = ViewerModel()
    controls_container = QtLayerControlsContainer(viewer)
    qtbot.addWidget(controls_container)
    object.__setattr__(
        viewer, "window",
        SimpleNamespace(_qt_viewer=SimpleNamespace(_controls=controls_container)),
    )
    layers = [pyramid(viewer, f"channel {i}") for i in range(2)]
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)
    animation = widget.animation
    layers.append(pyramid(viewer, "channel 2"))
    animation.add_scroll_through()

    controls = controls_container.widgets
    checkboxes = [animation.resolution._checkboxes[layer] for layer in layers]
    combos = [controls[layer]._multiscale_level_control.level_combobox
              for layer in layers]
    assert all(box.isChecked() for box in checkboxes)

    combos[0].setCurrentIndex(2)
    assert [layer.locked_data_level for layer in layers] == [1, 1, 1]
    assert [combo.currentIndex() for combo in combos] == [2, 2, 2]
    for frame in animation.key_frames:
        assert all(
            frame.viewer_state.layers[animation.layer_registry.source_id(layer)][
                "locked_data_level"
            ] == 1
            for layer in layers
        )

    checkboxes[1].setChecked(False)
    assert not any(box.isChecked() for box in checkboxes)
    combos[1].setCurrentIndex(1)
    assert [layer.locked_data_level for layer in layers] == [1, 0, 1]

    checkboxes[2].setChecked(True)
    combos[2].setCurrentIndex(0)
    assert [layer.locked_data_level for layer in layers] == [None] * 3
    assert all(box.isChecked() for box in checkboxes)
    widget.close()


@pytest.mark.parametrize("ndisplay", [2, 3])
def test_native_resolution_updates_all_frames_and_actual_pixels(qtbot, ndisplay):
    from napari._qt.layer_controls.widgets.qt_multiscale_level_control import QtMultiscaleLevelControl

    viewer = ViewerModel(ndisplay=ndisplay)
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)
    layer = pyramid(viewer)  # Also checks layers imported after opening the dock.
    animation = widget.animation
    assert layer.locked_data_level == 0
    widget.scrollWidget._add_scroll_through()
    assert len(animation.key_frames) == 2
    controls = QtMultiscaleLevelControl(widget, layer)
    combo = controls.level_combobox
    source_id = animation.layer_registry.source_id(layer)
    # Warm the interpolation cache before changing resolution.
    animation._frames[3]
    layer.name = "renamed"
    combo.setCurrentIndex(2)  # Auto, finest, coarse.
    for frame in animation.key_frames:
        assert frame.viewer_state.layers[source_id]["locked_data_level"] == 1
    assert animation._frames[3].layers[source_id]["locked_data_level"] == 1
    animation.set_to_keyframe(0)
    animation.set_to_keyframe(1)
    assert combo.currentIndex() == 2
    assert np.asarray(layer._data_view).max() == 9000
    combo.setCurrentIndex(1)
    for index in (0, 1, 0):
        animation.set_to_keyframe(index)
        assert combo.currentIndex() == 1
        assert layer.locked_data_level == 0
        assert np.asarray(layer._data_view).shape == (
            (16, 16) if ndisplay == 2 else (8, 16, 16)
        )
        assert np.asarray(layer._data_view).max() < 9000
    combo.setCurrentIndex(0)
    assert all(frame.viewer_state.layers[source_id]["locked_data_level"] is None
               for frame in animation.key_frames)
    assert animation._frames[3].layers[source_id]["locked_data_level"] is None
    widget.close()


def test_native_resolution_save_load_and_listener_cleanup(tmp_path):
    viewer = ViewerModel()
    layer = pyramid(viewer)
    animation = Animation(viewer)  # Also checks already loaded layers.
    assert layer.locked_data_level == 0
    animation.add_scroll_through()
    layer.locked_data_level = 1
    path = animation.save_keyframes(tmp_path / "resolution.json")
    restored_viewer = ViewerModel()
    restored_layer = pyramid(restored_viewer)
    restored = Animation(restored_viewer)
    try:
        restored.load_keyframes(path, reload_layers=False)
        assert restored_layer.locked_data_level == 1
        restored.set_to_keyframe(1)
        assert restored_layer.locked_data_level == 1
        assert np.asarray(restored_layer._data_view).max() == 9000
        viewer.layers.remove(layer)
        assert layer not in animation.resolution._listeners
        animation.close()
        later = pyramid(viewer, "later")
        assert later.locked_data_level is None
    finally:
        animation.close()
        restored.close()


def test_section_resolution_default_global_edits_and_navigation(qtbot, tmp_path):
    viewer = ViewerModel(ndisplay=3)
    source = pyramid(viewer)
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)
    animation = widget.animation
    section = animation.add_scene_object(OrthoSlice(source=source.name))
    assert section.source_level == 0
    animation.snap_2d(capture=False)
    widget.sceneWidget.refresh()
    widget.scrollWidget._add_scroll_through()
    control = widget.sceneWidget.orthoSliceEditor.sourceLevelComboBox
    assert control.currentData() == 0
    original_points = [frame.viewer_state.dims["point"] for frame in animation.key_frames]
    animation._frames[3]
    control.setCurrentIndex(control.findData(1))
    assert all(frame.viewer_state.scene[section.id]["source_level"] == 1
               for frame in animation.key_frames)
    assert animation._frames[3].scene[section.id]["source_level"] == 1
    animation.set_to_keyframe(0)
    control.setCurrentIndex(control.findData(0))
    for index in (1, 0, 1):
        animation.set_to_keyframe(index)
        live = animation._active_section()
        assert control.currentData() == live.source_level == 0
        assert widget.sceneWidget.orthoSliceEditor.object is live
        output = animation.layer_registry.find_managed(section.id, "section_2d")
        assert np.asarray(output._data_view).shape == (16, 16)
        np.testing.assert_array_equal(output._data_view, live.projection_oracle(viewer))
    assert [frame.viewer_state.dims["point"] for frame in animation.key_frames] == original_points
    path = animation.save_keyframes(tmp_path / "s.json")
    animation.load_keyframes(path, reload_layers=False)
    assert control.currentData() == 0
    assert all(frame.viewer_state.scene[section.id]["source_level"] == 0
               for frame in animation.key_frames)
    widget.close()


def test_legacy_mixed_section_levels_refresh_controls_without_rewriting(qtbot):
    viewer = ViewerModel(ndisplay=3)
    source = pyramid(viewer)
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)
    animation = widget.animation
    section = animation.add_scene_object(OrthoSlice(source=source.name))
    animation.snap_2d(capture=False)
    animation.add_scroll_through()
    frame = animation.key_frames[1]
    scene = dict(frame.viewer_state.scene)
    scene[section.id] = dict(scene[section.id], source_level=1)
    frame.viewer_state = replace(frame.viewer_state, scene=scene)
    animation._frames._rebuild_frame_index()
    for index, level in [(0, 0), (1, 1), (0, 0)]:
        animation.set_to_keyframe(index)
        assert widget.sceneWidget.orthoSliceEditor.sourceLevelComboBox.currentData() == level
    assert frame.viewer_state.scene[section.id]["source_level"] == 1
    widget.close()


def test_gui_section_refines_both_scroll_endpoints(make_napari_viewer, qtbot):
    viewer = make_napari_viewer(ndisplay=2)
    source = pyramid(viewer)
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)
    animation = widget.animation
    section = animation.add_scene_object(OrthoSlice(source=source.name))
    widget.sceneWidget.refresh()
    widget.sceneWidget.listWidget.setCurrentRow(0)
    animation.add_scroll_through(start=2, stop=6)
    control = widget.sceneWidget.orthoSliceEditor.sourceLevelComboBox
    control.setCurrentIndex(control.findData(1))
    animation.set_to_keyframe(0)
    control.setCurrentIndex(control.findData(0))
    # Rapid navigation must not let a pending coarse result win.
    animation.set_to_keyframe(1)
    for index, z in [(0, 2), (1, 6)]:
        animation.set_to_keyframe(index)
        expected = np.asarray(source.data[0])[z]

        def exact_pixels():
            layer = animation.layer_registry.find_managed(section.id, "section_2d")
            return layer is not None and np.array_equal(layer._data_view, expected)

        qtbot.waitUntil(exact_pixels, timeout=10000)
        assert control.currentData() == 0
        assert animation._active_section().source_level == 0
    widget.close()
