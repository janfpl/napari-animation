"""Tests for the ortho slice -- a thick optical section through a volume.

The slice is backed by a real napari layer in ``depiction='plane'`` mode, which
is what makes it appear in napari's layer list with its own visibility
checkbox, and what lets an external tool reposition it.
"""

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation, ViewerState
from napari_animation.layer_registry import LayerRegistry
from napari_animation.scene import (
    VIEW_NORMALS,
    ClipPlane,
    OrthoSlice,
    Scene,
    SceneObject,
    _manipulators,
)


class HeadlessViewer(ViewerModel):
    def screenshot(self, *args, **kwargs):
        return np.zeros((60, 60, 4), dtype=np.uint8)


@pytest.fixture
def viewer():
    viewer = HeadlessViewer()
    # ViewerModel starts in 2D, where a plane-depiction slab layer is hidden
    # because it would just redraw its source; these tests are about 3D
    viewer.dims.ndisplay = 3
    viewer.add_image(
        np.random.random((10, 20, 20)), name="img", scale=(3, 1, 1)
    )
    return viewer


# --------------------------------------------------------------- the model


def test_ortho_slice_round_trips_through_a_dict():
    slice_ = OrthoSlice(
        name="XY", position=(1.0, 2.0, 3.0), thickness=7.0, projection="mean"
    )

    restored = SceneObject.from_dict(slice_.to_dict())

    assert isinstance(restored, OrthoSlice)
    assert restored == slice_


def test_named_views_map_to_axis_normals():
    # "XY" is what you see looking down Z, so it is sliced along Z
    assert VIEW_NORMALS["XY"] == (1.0, 0.0, 0.0)
    assert VIEW_NORMALS["YZ"] == (0.0, 0.0, 1.0)


# ------------------------------------------------------- the backing layer


def test_plane_mode_creates_a_real_layer(viewer):
    scene = Scene()
    scene.append(OrthoSlice(name="Ortho XY", source="img"))

    scene.apply(viewer)

    assert "Ortho XY" in viewer.layers
    layer = viewer.layers["Ortho XY"]
    assert layer.depiction == "plane"
    # sharing the source array rather than copying it
    assert layer.data is viewer.layers["img"].data


def test_managed_section_layers_keep_source_units(viewer):
    source = viewer.layers["img"]
    source.units = ("um", "um", "um")
    scene = Scene()
    section = OrthoSlice(name="Ortho", source="img")
    scene.append(section)

    scene.apply(viewer)
    backing = viewer.layers["Ortho"]
    assert tuple(backing.units) == tuple(source.units)
    assert viewer.layers.extent.units is not None

    viewer.dims.ndisplay = 2
    scene.apply(viewer)
    output = LayerRegistry(viewer).find_managed(section.id, "section_2d")
    assert output is not None
    assert tuple(output.units) == tuple(source.units)
    assert viewer.layers.extent.units is not None

    source.units = ("nm", "nm", "nm")
    scene.apply(viewer)
    assert tuple(backing.units) == tuple(source.units)
    assert tuple(output.units) == tuple(source.units)


def test_voxel_size_change_updates_existing_section_units(viewer):
    animation = Animation(viewer)
    section = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img")
    )
    backing = LayerRegistry(viewer).find_managed(section.id, "section_3d")

    animation.set_voxel_size("img", (2, 1, 1), units="um")

    assert tuple(backing.units) == tuple(viewer.layers["img"].units)
    assert viewer.layers.extent.units is not None


def test_voxel_size_change_moves_existing_section_with_its_source(viewer):
    animation = Animation(viewer)
    section = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img")
    )
    backing = LayerRegistry(viewer).find_managed(section.id, "section_3d")

    animation.set_voxel_size("img", (5, 1, 1))

    np.testing.assert_allclose(
        backing.extent.world, viewer.layers["img"].extent.world
    )


def test_projection_maps_onto_the_rendering_mode(viewer):
    scene = Scene()
    scene.append(OrthoSlice(name="Ortho", source="img", projection="mean"))

    scene.apply(viewer)

    assert viewer.layers["Ortho"].rendering == "average"


def test_2d_plane_slice_uses_managed_projection_layer(viewer):
    """A scene-owned slab remains visible as a thick 2D section."""
    viewer.dims.ndisplay = 2
    scene = Scene()
    scene.append(
        OrthoSlice(
            name="Ortho",
            source="img",
            position=(9.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
            thickness=6.0,
            projection="mean",
        )
    )

    scene.apply(viewer)

    # The 3D backing plane is hidden and a deterministic projected singleton
    # owns the 2D view. This avoids napari's inclusive-margin rounding.
    assert viewer.layers["Ortho"].visible is False
    output = LayerRegistry(viewer).find_managed(scene[0].id, "section_2d")
    assert output is not None and output.visible
    assert viewer.dims.margin_left[0] == pytest.approx(0.0)
    assert viewer.dims.margin_right[0] == pytest.approx(0.0)
    assert viewer.dims.point[0] == pytest.approx(9.0)
    assert scene[0].position[0] == pytest.approx(9.0)
    np.testing.assert_allclose(
        np.squeeze(output.data), scene[0].projection_oracle(viewer)
    )
    # Keep a native plane available for editors and a 3D transition.
    assert viewer.layers["Ortho"].plane.thickness == pytest.approx(2.0)


def test_plane_events_update_live_object(viewer):
    """Native napari edits are captured by the editable scene object."""
    scene = Scene()
    slice_ = OrthoSlice(name="Ortho", source="img")
    scene.append(slice_)
    scene.apply(viewer)

    layer = viewer.layers["Ortho"]
    layer.plane.position = (2.0, 0.0, 0.0)
    layer.plane.thickness = 4.0
    layer.visible = False
    layer.rendering = "average"

    assert slice_.position[0] == pytest.approx(6.0)
    assert slice_.thickness == pytest.approx(12.0)
    assert slice_.enabled is False
    assert slice_.projection == "mean"


def test_multiscale_source_level_rebinds_backing_layer(viewer):
    """A plane slice uses the selected pyramid level without copying data."""
    viewer.layers.remove(viewer.layers["img"])
    levels = [
        np.zeros((16, 32, 32), dtype=np.uint8),
        np.ones((8, 16, 16), dtype=np.uint8),
    ]
    viewer.add_image(
        levels, name="pyramid", multiscale=True, scale=(2.0, 3.0, 4.0)
    )
    scene = Scene()
    slice_ = OrthoSlice(name="Ortho", source="pyramid", source_level=1)
    scene.append(slice_)
    scene.apply(viewer)

    backing = viewer.layers["Ortho"]
    assert backing.data.shape == levels[1].shape
    np.testing.assert_allclose(
        backing.affine.affine_matrix[:3, :3], np.diag((4.0, 6.0, 8.0))
    )
    assert backing.data is viewer.layers["pyramid"].data[1]

    slice_.source_level = 0
    scene.apply(viewer)
    assert viewer.layers["Ortho"] is backing
    assert backing.data is viewer.layers["pyramid"].data[0]
    np.testing.assert_allclose(
        backing.affine.affine_matrix[:3, :3], np.diag((2.0, 3.0, 4.0))
    )


def test_source_id_survives_source_layer_rename(viewer):
    scene = Scene()
    slice_ = OrthoSlice(name="Ortho", source="img")
    scene.append(slice_)
    scene.apply(viewer)
    source_id = slice_.source_id

    viewer.layers["img"].name = "renamed volume"
    scene.apply(viewer)

    assert slice_.source_id == source_id
    assert slice_.source == "renamed volume"
    assert viewer.layers["Ortho"].data is viewer.layers["renamed volume"].data


def test_active_section_owns_the_2d_projection(viewer):
    viewer.dims.ndisplay = 2
    animation = Animation(viewer)
    first = animation.add_scene_object(
        OrthoSlice(
            name="XY",
            source="img",
            position=(3.0, 0.0, 0.0),
            thickness=2.0,
            projection="max",
        )
    )
    second = animation.add_scene_object(
        OrthoSlice(
            name="XZ",
            source="img",
            position=(7.0, 0.0, 0.0),
            thickness=6.0,
            projection="mean",
        )
    )

    animation.set_active_section(second.id)

    assert animation.active_section_id == second.id
    assert viewer.dims.point[0] == pytest.approx(7.0)
    assert viewer.dims.margin_left[0] == pytest.approx(0.0)
    # The other managed layer remains available for a later 3D transition but
    # does not own global 2D margins or projection settings.
    assert viewer.layers["XY"].visible is False
    assert viewer.layers["XZ"].visible is False
    assert (
        not LayerRegistry(viewer).find_managed(first.id, "section_2d").visible
    )
    assert LayerRegistry(viewer).find_managed(second.id, "section_2d").visible


def test_2d_projection_oracle_matches_coordinate_ramp():
    viewer = HeadlessViewer()
    data = np.arange(5 * 6 * 7, dtype=float).reshape(5, 6, 7)
    viewer.add_image(data, name="ramp", scale=(2.0, 1.0, 1.0))
    viewer.dims.ndisplay = 2
    slice_ = OrthoSlice(
        name="section",
        source="ramp",
        position=(4.0, 0.0, 0.0),
        normal=VIEW_NORMALS["XY"],
        thickness=6.0,
        projection="mean",
    )

    oracle = slice_.projection_oracle(viewer)

    expected = np.mean(data[1:4], axis=0)
    np.testing.assert_allclose(oracle, expected)


@pytest.mark.parametrize(
    "projection, reducer",
    [
        ("max", np.max),
        ("mean", np.mean),
        ("min", np.min),
    ],
)
def test_2d_native_pixels_match_oracle_for_exact_even_slab(
    projection, reducer
):
    """The real napari slice and the independent oracle use the same voxels."""
    viewer = HeadlessViewer()
    data = np.arange(5 * 4 * 3, dtype=float).reshape(5, 4, 3)
    viewer.add_image(data, name="ramp", scale=(2.0, 1.0, 1.0))
    viewer.dims.ndisplay = 2
    section = OrthoSlice(
        name="section",
        source="ramp",
        # data centre 2.5, thickness 2 data units -> exact interval [2, 4)
        position=(5.0, 0.0, 0.0),
        normal=VIEW_NORMALS["XY"],
        thickness=4.0,
        projection=projection,
    )
    scene = Scene()
    scene.append(section)
    scene.apply(viewer)

    expected = reducer(data[2:4], axis=0)
    np.testing.assert_allclose(section.projection_oracle(viewer), expected)
    output = LayerRegistry(viewer).find_managed(section.id, "section_2d")
    np.testing.assert_allclose(np.asarray(output._data_view), expected)


def test_oblique_section_is_rejected_before_2d_mutation(viewer):
    viewer.dims.ndisplay = 2
    viewer.dims.margin_left = (4.0, 0.0, 0.0)
    before = tuple(viewer.dims.margin_left)
    scene = Scene()
    scene.append(
        OrthoSlice(name="oblique", source="img", normal=(1.0, 1.0, 0.0))
    )

    with pytest.raises(ValueError, match="require a 3D view"):
        scene.apply(viewer)

    assert tuple(viewer.dims.margin_left) == before


def test_active_section_is_captured_and_restored(viewer):
    viewer.dims.ndisplay = 2
    animation = Animation(viewer)
    first = animation.add_scene_object(OrthoSlice(name="first", source="img"))
    second = animation.add_scene_object(
        OrthoSlice(name="second", source="img", position=(6.0, 0.0, 0.0))
    )
    animation.set_active_section(first.id)
    animation.capture_keyframe(steps=1)
    animation.set_active_section(second.id)
    animation.capture_keyframe(steps=1)

    assert animation.key_frames[1].viewer_state.view["active_section_id"] == (
        second.id
    )
    animation.set_movie_frame_index(0)
    assert animation.active_section_id == first.id
    animation.set_movie_frame_index(1)
    assert animation.active_section_id == second.id


def test_geometry_reaches_the_layer_plane(viewer):
    scene = Scene()
    scene.append(
        OrthoSlice(
            name="Ortho",
            source="img",
            position=(9.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
            thickness=6.0,
        )
    )

    scene.apply(viewer)

    plane = viewer.layers["Ortho"].plane
    # world z=9 is data z=3 at a z-scale of 3
    assert plane.position[0] == pytest.approx(3.0)
    # 6 world units along z is 2 planes at a z-scale of 3
    assert plane.thickness == pytest.approx(2.0)


def test_thickness_accounts_for_anisotropy(viewer):
    """The same world thickness is a different data thickness per axis."""
    along_z = OrthoSlice(source="img", normal=(1.0, 0.0, 0.0), thickness=6.0)
    along_x = OrthoSlice(source="img", normal=(0.0, 0.0, 1.0), thickness=6.0)
    layer = viewer.layers["img"]

    # z is scaled 3x, x is not
    assert along_z.data_thickness(layer) == pytest.approx(2.0)
    assert along_x.data_thickness(layer) == pytest.approx(6.0)


def test_disabling_hides_the_layer_without_removing_it(viewer):
    scene = Scene()
    slice_ = OrthoSlice(name="Ortho", source="img")
    scene.append(slice_)
    scene.apply(viewer)
    assert viewer.layers["Ortho"].visible is True

    slice_.enabled = False
    scene.apply(viewer)

    # hidden, not deleted: deleting layers mid-animation is disruptive
    assert "Ortho" in viewer.layers
    assert viewer.layers["Ortho"].visible is False


def test_the_slice_layer_is_not_its_own_source(viewer):
    """Auto-selecting a source must not pick the slice's own backing layer."""
    scene = Scene()
    scene.append(OrthoSlice(name="Ortho"))

    scene.apply(viewer)
    scene.apply(viewer)  # second pass: the backing layer now exists

    assert viewer.layers["Ortho"].data is viewer.layers["img"].data


# ------------------------------------------------------------- clip mode


def test_clip_mode_carves_a_slab_out_of_the_source(viewer):
    scene = Scene()
    scene.append(
        OrthoSlice(
            name="Ortho",
            source="img",
            render_mode="clip",
            position=(15.0, 0.0, 0.0),
            thickness=6.0,
        )
    )

    scene.apply(viewer)

    planes = viewer.layers["img"].experimental_clipping_planes
    assert len(planes) == 2
    assert planes[0].position[0] < planes[1].position[0]
    assert planes[0].normal[0] == pytest.approx(1.0)
    assert planes[1].normal[0] == pytest.approx(-1.0)
    # no backing layer is created in clip mode
    assert "Ortho" not in viewer.layers


def test_clip_slab_composites_with_cutaway_planes(viewer):
    scene = Scene()
    scene.append(ClipPlane(name="cut", targets=("img",)))
    scene.append(OrthoSlice(name="Ortho", source="img", render_mode="clip"))

    scene.apply(viewer)

    # the cutaway plane plus the slab's two bounding planes
    assert len(viewer.layers["img"].experimental_clipping_planes) == 3


# ------------------------------------------------------------ sync/capture


def test_geometry_is_read_back_from_the_layer(viewer):
    """An externally repositioned plane is what gets captured."""
    scene = Scene()
    slice_ = OrthoSlice(name="Ortho", source="img")
    scene.append(slice_)
    scene.apply(viewer)

    # something else moves the plane -- a napari-threedee manipulator, the
    # napari controls, a user script
    viewer.layers["Ortho"].plane.position = (2.0, 0.0, 0.0)

    scene.sync_from_viewer(viewer)

    # data z=2 is world z=6 at a z-scale of 3
    assert slice_.position[0] == pytest.approx(6.0)


def test_capture_syncs_before_snapshotting(viewer):
    animation = Animation(viewer)
    slice_ = animation.add_scene_object(OrthoSlice(name="Ortho", source="img"))
    viewer.layers["Ortho"].plane.position = (2.0, 0.0, 0.0)

    animation.capture_keyframe()

    captured = animation.key_frames[0].viewer_state.scene[slice_.id]
    assert captured["position"][0] == pytest.approx(6.0)


def test_slice_sweeps_between_keyframes(viewer):
    animation = Animation(viewer)
    slice_ = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img", position=(0.0, 0.0, 0.0))
    )
    animation.capture_keyframe(steps=1)
    slice_.position = (27.0, 0.0, 0.0)
    animation.capture_keyframe(steps=10)

    positions = [
        animation._frames[i].scene[slice_.id]["position"][0]
        for i in range(len(animation._frames))
    ]

    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(27.0)
    assert positions == sorted(positions)
    assert 0.0 < positions[len(positions) // 2] < 27.0

    # and the swept state drives the real backing layer
    animation._frames[len(positions) // 2].apply(viewer)
    assert 0.0 < viewer.layers["Ortho"].plane.position[0] < 9.0


def test_removing_a_slice_removes_its_layer(viewer):
    animation = Animation(viewer)
    slice_ = animation.add_scene_object(OrthoSlice(name="Ortho", source="img"))
    animation.scene.apply(viewer)
    assert "Ortho" in viewer.layers

    animation.remove_scene_object(slice_)

    assert "Ortho" not in viewer.layers
    assert len(animation.scene) == 0


def test_backing_layer_visibility_toggles_per_keyframe(viewer):
    animation = Animation(viewer)
    slice_ = animation.add_scene_object(OrthoSlice(name="Ortho", source="img"))
    animation.capture_keyframe()
    animation.capture_keyframe()

    animation.set_object_enabled(1, slice_.id, False)

    animation.key_frames[0].viewer_state.apply(viewer)
    assert viewer.layers["Ortho"].visible is True
    animation.key_frames[1].viewer_state.apply(viewer)
    assert viewer.layers["Ortho"].visible is False


# -------------------------------------------------------- optional n3d dep


def test_manipulator_degrades_when_napari_threedee_is_absent(viewer):
    """The optional dependency must never be load-bearing."""
    scene = Scene()
    scene.append(OrthoSlice(name="Ortho", source="img"))
    scene.apply(viewer)

    result = _manipulators.attach_plane_manipulator(
        viewer, viewer.layers["Ortho"]
    )

    # installed or not, this returns cleanly rather than raising
    assert result is None or result is not None
    assert isinstance(_manipulators.is_available(), bool)


def test_manipulator_declines_non_plane_layers(viewer):
    assert (
        _manipulators.attach_plane_manipulator(viewer, viewer.layers["img"])
        is None
    )


def test_viewer_state_captures_the_slice_layer(viewer):
    """The backing layer is a real layer, so it is captured like any other."""
    scene = Scene()
    slice_ = OrthoSlice(name="Ortho", source="img")
    scene.append(slice_)
    scene.apply(viewer)

    state = ViewerState.from_viewer(viewer, scene=scene.to_dict())

    assert "Ortho" in state.layers
    assert state.layers["Ortho"]["depiction"] == "plane"


def test_clip_snapshot_keeps_lazy_editable_plane_then_materializes(viewer):
    from napari_animation.scene.renderer import SceneRenderer

    section = OrthoSlice(
        name="Clipped", source="img", render_mode="clip", position=(9, 10, 10)
    )
    renderer = SceneRenderer.for_viewer(viewer)
    renderer.prepare_snapshot({section.id: section.to_dict()}, ndisplay=3)
    layer = LayerRegistry(viewer).find_managed(section.id, "section_3d")
    assert layer.data.shape == (1, 1, 1)
    assert layer._napari_animation_placeholder
    assert not layer.visible
    # The hidden plane retains the selected source coordinate system.
    np.testing.assert_allclose(layer.affine.linear_matrix, np.diag([3, 1, 1]))
    section.render_mode = "plane"
    section.apply(viewer)
    assert (
        LayerRegistry(viewer).find_managed(section.id, "section_3d") is layer
    )
    assert layer.data is viewer.layers["img"].data
    assert not layer._napari_animation_placeholder
    assert layer.visible
    np.testing.assert_allclose(layer.plane.position, [3, 10, 10])


def test_clip_placeholder_uses_replaced_source_on_plane_transition(viewer):
    section = OrthoSlice(name="Clipped", source="img", render_mode="clip")
    layer = section.ensure_layer(viewer)
    viewer.layers["img"].data = np.ones((8, 12, 14), dtype=np.uint16)
    assert section.ensure_layer(viewer) is layer
    assert layer.data.shape == (1, 1, 1)
    section.render_mode = "plane"
    section.apply(viewer)
    assert layer.data is viewer.layers["img"].data
    assert layer.data.shape == (8, 12, 14)
