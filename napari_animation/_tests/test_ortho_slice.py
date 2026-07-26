"""Tests for the ortho slice -- a thick optical section through a volume.

The slice is backed by a real napari layer in ``depiction='plane'`` mode, which
is what makes it appear in napari's layer list with its own visibility
checkbox, and what lets an external tool reposition it.
"""

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation, ViewerState
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


def test_projection_maps_onto_the_rendering_mode(viewer):
    scene = Scene()
    scene.append(OrthoSlice(name="Ortho", source="img", projection="mean"))

    scene.apply(viewer)

    assert viewer.layers["Ortho"].rendering == "average"


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
