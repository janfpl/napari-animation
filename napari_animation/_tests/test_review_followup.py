"""Independent regressions found while reviewing the renderer fixes."""

from dataclasses import replace

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation, ViewerState
from napari_animation.layer_registry import LayerRegistry
from napari_animation.scene import OrthoSlice


def setup_section(ndisplay=2, **kwargs):
    viewer = ViewerModel(ndisplay=ndisplay)
    source = viewer.add_image(
        np.arange(6 * 7 * 9, dtype=float).reshape(6, 7, 9), name="volume"
    )
    animation = Animation(viewer)
    section = animation.add_scene_object(
        OrthoSlice(name="section", source="volume", position=(2, 3, 4), **kwargs)
    )
    animation.set_active_section(section.id)
    return viewer, source, animation, section


def managed(viewer, section, role="section_2d"):
    return LayerRegistry(viewer).find_managed(section.id, role)


def test_unchanged_section_keeps_existing_data_binding():
    viewer, source, animation, section = setup_section()
    backing = managed(viewer, section, "section_3d")
    data_events = []
    backing.events.data.connect(lambda event: data_events.append(event))

    animation.apply_working_state()
    animation.apply_working_state()

    assert not data_events
    # A section first created in 2D keeps only a tiny editable plane layer;
    # the full source is bound when the user enters 3D.
    assert backing.data.shape == (1, 1, 1)
    animation.snap_3d(capture=False)
    assert backing.data is source.data


def test_2d_section_does_not_toggle_source_visibility_on_reapply():
    viewer, source, animation, section = setup_section()
    events = []
    source.events.visible.connect(lambda event: events.append(event))
    animation.apply_working_state()
    animation.apply_working_state()
    assert not source.visible
    assert not events


def test_unchanged_3d_section_emits_no_plane_event():
    viewer = ViewerModel(ndisplay=3)
    viewer.add_image(np.zeros((6, 7, 9)), name="volume")
    animation = Animation(viewer)
    section = animation.add_scene_object(OrthoSlice(source="volume"))
    backing = managed(viewer, section, "section_3d")
    events = []
    backing.events.plane.connect(lambda event: events.append(event))
    animation.apply_working_state()
    animation.apply_working_state()
    assert not events


def test_in_place_source_edit_can_invalidate_hidden_section():
    viewer, source, animation, section = setup_section()
    output = managed(viewer, section)
    before = np.asarray(output.data).copy()
    source.data[2, 3, 4] = 10000
    animation.invalidate_source("volume")
    output = managed(viewer, section)
    assert not np.array_equal(output.data, before)


def test_timepoint_binding_changes_only_when_time_changes():
    viewer = ViewerModel(ndisplay=3)
    source = viewer.add_image(np.zeros((2, 6, 7, 9)), name="volume")
    animation = Animation(viewer)
    section = animation.add_scene_object(OrthoSlice(source="volume"))
    backing = managed(viewer, section, "section_3d")
    data_events = []
    backing.events.data.connect(lambda event: data_events.append(event))

    animation.apply_working_state()
    assert not data_events
    viewer.dims.set_point(0, 1)
    assert len(data_events) == 1
    animation.apply_working_state()
    assert len(data_events) == 1


def test_keyframe_selection_applies_snapshot_once(monkeypatch):
    viewer, source, animation, section = setup_section()
    monkeypatch.setattr(
        type(viewer), "screenshot",
        lambda *args, **kwargs: np.zeros((32, 32, 4), dtype=np.uint8),
        raising=False,
    )
    animation.capture_keyframe()
    animation.update_scene_object(section.id, position=(4, 3, 4))
    animation.capture_keyframe()
    applied = []
    original = ViewerState.apply

    def counted(state, target):
        applied.append(state)
        return original(state, target)

    monkeypatch.setattr(ViewerState, "apply", counted)
    animation.set_to_keyframe(0)

    assert len(applied) == 1
    assert section.position == (2, 3, 4)
    assert managed(viewer, section, "section_3d")._napari_animation_plane_listener[0] is section


def test_initial_authored_position_is_not_replaced_by_dims():
    viewer, source, animation, section = setup_section()
    assert section.position == (2, 3, 4)
    assert animation._capture_state()[1][section.id]["position"] == (2, 3, 4)

    other = animation.add_scene_object(
        OrthoSlice(source="volume", position=(4.25, 3, 4))
    )
    animation.set_active_section(other.id)
    assert other.position == (4.25, 3, 4)
    assert animation._capture_state()[1][other.id]["position"] == (4.25, 3, 4)


def test_outside_position_survives_capture():
    viewer, source, animation, section = setup_section()
    animation.update_scene_object(section.id, position=(1000, 3, 4))
    assert animation._capture_state()[1][section.id]["position"] == (1000, 3, 4)


def test_2d_to_oblique_3d_snapshot_uses_destination_mode():
    viewer, source, animation, section = setup_section(3, normal=(1, 1, 0))
    state = ViewerState.from_viewer(
        viewer, scene=animation.scene.to_dict(), view={"active_section_id": section.id}
    )
    viewer.dims.ndisplay = 2
    state.apply(viewer)
    assert viewer.dims.ndisplay == 3
    assert managed(viewer, section, "section_3d").visible


def test_disabling_active_section_restores_source():
    viewer, source, animation, section = setup_section()
    animation.update_scene_object(section.id, enabled=False)
    assert not managed(viewer, section).visible
    assert source.visible
    assert animation.active_section_id == section.id


def test_clip_mode_has_visible_thick_projection_in_2d():
    viewer, source, animation, section = setup_section(render_mode="clip", thickness=2)
    output = managed(viewer, section)
    assert output is not None and output.visible
    np.testing.assert_array_equal(output._data_view, source.data[1:3].max(axis=0))


def test_2d_projection_uses_section_appearance():
    viewer, source, animation, section = setup_section(3)
    backing = managed(viewer, section, "section_3d")
    backing.colormap = "red"
    backing.gamma = 2
    backing.opacity = 0.25
    backing.contrast_limits = (10, 100)
    animation.snap_2d(capture=False)
    output = managed(viewer, section)
    assert output.colormap.name == "red"
    assert output.gamma == 2
    assert output.opacity == 0.25
    assert list(output.contrast_limits) == [10, 100]


@pytest.mark.parametrize("render_mode", ["plane", "clip"])
def test_first_snapshot_application_restores_section_appearance(render_mode):
    viewer, source, animation, section = setup_section(render_mode=render_mode)
    managed(viewer, section, "section_3d").colormap = "red"
    state = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    with animation.renderer.applying_state():
        section.remove_layer(viewer)
    state.apply(viewer)
    assert managed(viewer, section).colormap.name == "red"


def test_colliding_managed_name_does_not_change_each_render():
    viewer, source, animation, section = setup_section()
    viewer.add_image(np.zeros((2, 2, 2)), name="occupied")
    animation.update_scene_object(section.id, name="occupied")
    name = managed(viewer, section, "section_3d").name
    animation.apply_working_state()
    assert managed(viewer, section, "section_3d").name == name


def test_deleted_source_cannot_rebind_by_reused_name():
    viewer, source, animation, section = setup_section()
    state = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    viewer.layers.remove(source)
    replacement = viewer.add_image(np.zeros((6, 7, 9)), name="volume", opacity=0.3)
    assert section.source_layer(viewer) is None
    state.apply(viewer)
    assert replacement.opacity == 0.3
    assert not managed(viewer, section).visible


@pytest.mark.parametrize("scene", [None, {}])
def test_absent_scene_clears_outputs_and_restores_source(scene):
    viewer, source, animation, section = setup_section()
    state = ViewerState.from_viewer(viewer, scene=scene)
    state.apply(viewer)
    assert not managed(viewer, section).visible
    assert source.visible


def test_snapshot_dims_events_do_not_modify_live_scene():
    viewer, source, animation, section = setup_section()
    state = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    dims = dict(state.dims, current_step=(4, 3, 4))
    snapshot_scene = {section.id: dict(state.scene[section.id], position=(4, 3, 4))}
    replace(state, dims=dims, scene=snapshot_scene).apply(viewer)
    assert section.position == (2, 3, 4)


def test_partially_outside_slab_displays_expected_pixels():
    viewer, source, animation, section = setup_section(thickness=14)
    animation.update_scene_object(section.id, position=(-5, 3, 4))
    output = managed(viewer, section)
    assert output.visible
    np.testing.assert_array_equal(output._data_view, source.data[0:2].max(axis=0))


def test_boundary_spanning_slab_is_not_sliced_out_of_output():
    viewer, source, animation, section = setup_section(thickness=100)
    animation.update_scene_object(section.id, position=(0, 3, 4))
    output = managed(viewer, section)
    assert not output._slice.empty
    np.testing.assert_array_equal(output._data_view, source.data.max(axis=0))


def test_time_scroll_refreshes_3d_section():
    viewer = ViewerModel(ndisplay=3)
    data = np.stack([np.zeros((6, 7, 9)), np.ones((6, 7, 9))])
    viewer.add_image(data, name="volume")
    viewer.dims.set_point(0, 0)
    animation = Animation(viewer)
    section = animation.add_scene_object(OrthoSlice(source="volume"))
    viewer.dims.set_point(0, 1)
    np.testing.assert_array_equal(managed(viewer, section, "section_3d").data, data[1])


def test_invalid_edit_does_not_poison_scene():
    viewer, source, animation, section = setup_section()
    before = section.to_dict()
    with pytest.raises(ValueError):
        animation.update_scene_object(section.id, thickness=-1)
    assert section.to_dict() == before


def test_interpolated_2d_axis_switch_stays_renderable():
    from napari_animation.frame_sequence import FrameSequence
    from napari_animation.key_frame import KeyFrameList

    viewer, source, animation, section = setup_section()
    start = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    end = replace(
        start, dims=dict(start.dims, order=(1, 0, 2)),
        scene={section.id: dict(start.scene[section.id], normal=(0, 1, 0))},
    )
    middle = FrameSequence(KeyFrameList())._interpolate_state(start, end, 0.5)
    middle.apply(viewer)
    assert tuple(middle.scene[section.id]["normal"]) == (1, 0, 0)


def test_interpolated_2d_source_rotation_stays_renderable():
    from napari_animation.frame_sequence import FrameSequence
    from napari_animation.key_frame import KeyFrameList

    viewer, source, animation, section = setup_section()
    start = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    layers = {key: dict(value) for key, value in start.layers.items()}
    layers[LayerRegistry.ensure_id(source)]["rotate"] = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float
    )
    end = replace(start, layers=layers)
    middle = FrameSequence(KeyFrameList())._interpolate_state(start, end, 0.5)
    middle.apply(viewer)
    np.testing.assert_allclose(source.rotate, np.eye(3))


def test_backfilled_section_is_bound_before_source_name_reuse():
    from napari_animation import KeyFrame

    viewer = ViewerModel()
    source = viewer.add_image(np.ones((6, 7, 9)), name="volume")
    animation = Animation(viewer)
    animation.key_frames.append(KeyFrame(
        viewer_state=ViewerState.from_viewer(viewer),
        thumbnail=np.zeros((30, 30, 4), dtype=np.uint8),
    ))
    section = animation.add_scene_object(OrthoSlice(source="volume"))
    params = animation.key_frames[0].viewer_state.scene[section.id]
    assert params["source_id"] == LayerRegistry.ensure_id(source)
    source.name = "renamed"
    viewer.add_image(np.zeros((6, 7, 9)), name="volume")
    animation.set_to_keyframe(0)
    assert animation.scene[0].source_layer(viewer) is source


def test_snapshot_validates_destination_source_transform():
    viewer, source, animation, section = setup_section()
    state = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    with animation.renderer.applying_state():
        source.rotate = (45, 0, 0)
    state.apply(viewer)
    np.testing.assert_allclose(source.rotate, np.eye(3))


def test_invalid_snapshot_transform_is_rejected_before_mutation():
    viewer, source, animation, section = setup_section()
    state = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    layers = {key: dict(value) for key, value in state.layers.items()}
    layers[LayerRegistry.ensure_id(source)]["rotate"] = (45, 0, 0)
    with pytest.raises(ValueError, match="require a 3D view"):
        replace(state, layers=layers).apply(viewer)
    np.testing.assert_allclose(source.rotate, np.eye(3))


def test_load_failure_restores_working_edit_and_bindings(tmp_path, monkeypatch):
    from napari_animation import KeyFrame

    viewer, source, animation, section = setup_section()
    animation.key_frames.append(KeyFrame(
        viewer_state=ViewerState.from_viewer(viewer, scene=animation.scene.to_dict()),
        thumbnail=np.zeros((30, 30, 4), dtype=np.uint8),
    ))
    path = animation.save_keyframes(tmp_path / "animation.json")
    fresh_viewer, fresh_source, fresh, fresh_section = setup_section()
    fresh.update_scene_object(fresh_section.id, position=(4, 3, 4))
    fresh_viewer.camera.zoom = 7.5
    before_scene = fresh.scene.to_dict()
    before_metadata = dict(fresh_source.metadata)
    before_layers = list(fresh_viewer.layers)
    real_select = fresh.set_to_keyframe

    def fail_after_selection(index):
        real_select(index)
        raise RuntimeError("injected selection failure")

    monkeypatch.setattr(fresh, "set_to_keyframe", fail_after_selection)
    with pytest.raises(RuntimeError, match="injected"):
        fresh.load_keyframes(path, reload_layers=False)
    assert len(fresh.key_frames) == 0
    assert fresh_source.metadata == before_metadata
    assert list(fresh_viewer.layers) == before_layers
    assert fresh.scene.to_dict() == before_scene
    assert fresh_viewer.camera.zoom == pytest.approx(7.5)


@pytest.mark.parametrize("ndisplay", [2, 3])
def test_four_dimensional_clip_section(ndisplay):
    viewer = ViewerModel(ndisplay=ndisplay)
    viewer.add_image(np.ones((2, 6, 7, 9)), name="volume")
    animation = Animation(viewer)
    section = animation.add_scene_object(OrthoSlice(source="volume", render_mode="clip"))
    assert len(viewer.layers["volume"].experimental_clipping_planes) == 2
    if ndisplay == 2:
        assert managed(viewer, section).visible


def test_invalid_add_leaves_scene_and_layers_unchanged():
    viewer, source, animation, section = setup_section()
    before_layers = list(viewer.layers)
    before_scene = animation.scene.to_dict()
    with pytest.raises(ValueError):
        animation.add_scene_object(OrthoSlice(source="volume", thickness=-1))
    assert animation.scene.to_dict() == before_scene
    assert list(viewer.layers) == before_layers


@pytest.mark.parametrize("centre,thickness", [(2, 2), (0, 100), (-5, 14)])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_section_projection_matches_reference_opengl_pixels(
    make_napari_viewer, centre, thickness, axis
):
    viewer = make_napari_viewer(show=True)
    data = np.arange(6 * 7 * 9, dtype=float).reshape(6, 7, 9)
    viewer.add_image(data, name="volume", contrast_limits=(0, 400))
    animation = Animation(viewer)
    position = [2, 3, 4]
    position[axis] = centre
    section = animation.add_scene_object(
        OrthoSlice(source="volume", position=tuple(position), thickness=thickness,
                   normal=tuple(np.eye(3)[axis]))
    )
    animation.set_active_section(section.id)
    backing = managed(viewer, section, "section_3d")
    backing.colormap = "red"
    animation.apply_working_state()
    viewer.window._qt_viewer.canvas.native.setFixedSize(400, 300)
    from qtpy.QtWidgets import QApplication
    QApplication.processEvents()
    viewer.camera.zoom = 20
    camera_center = (0, *(value for index, value in enumerate(position) if index != axis))
    viewer.camera.center = camera_center
    QApplication.processEvents()
    actual = viewer.screenshot(canvas_only=True, flash=False)

    reference = make_napari_viewer(show=True)
    indices = np.arange(data.shape[axis])
    included = (indices >= centre - thickness / 2) & (indices < centre + thickness / 2)
    expected_data = np.take(data, indices[included], axis=axis).max(axis=axis)
    reference.add_image(expected_data, contrast_limits=(0, 400), colormap="red")
    reference.window._qt_viewer.canvas.native.setFixedSize(400, 300)
    QApplication.processEvents()
    reference.camera.zoom = 20
    reference.camera.center = camera_center
    QApplication.processEvents()
    expected = reference.screenshot(canvas_only=True, flash=False)
    assert np.ptp(expected[..., 0]) > 30
    np.testing.assert_allclose(actual.astype(float), expected.astype(float), atol=1)
