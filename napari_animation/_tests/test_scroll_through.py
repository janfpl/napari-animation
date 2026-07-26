"""Tests for view snapping and the scroll-through helpers.

Two kinds of scroll-through: walking the dims slider (the plain slice-by-slice
pass), and sliding an ortho slice along its own normal (the 3D version, which
works for oblique planes too).
"""

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation
from napari_animation.ortho_slicer import order_for_view
from napari_animation.scene import VIEW_NORMALS, OrthoSlice


class HeadlessViewer(ViewerModel):
    def screenshot(self, *args, **kwargs):
        return np.zeros((60, 60, 4), dtype=np.uint8)


@pytest.fixture
def animation():
    viewer = HeadlessViewer()
    # ViewerModel starts in 2D; these features are about 3D animations
    viewer.dims.ndisplay = 3
    viewer.add_image(
        np.random.random((10, 20, 30)), name="img", scale=(3, 1, 1)
    )
    return Animation(viewer)


def _points(animation, axis):
    return [
        animation._frames[i].dims["point"][axis]
        for i in range(len(animation._frames))
    ]


# ------------------------------------------------------------- view orders


def test_order_puts_the_sliced_axis_first():
    # napari displays the last ndisplay axes, so the odd one out goes to front
    assert order_for_view("XY", 3) == (0, 1, 2)  # slice Z, show Y/X
    assert order_for_view("XZ", 3) == (1, 0, 2)  # slice Y, show Z/X
    assert order_for_view("YZ", 3) == (2, 0, 1)  # slice X, show Z/Y


def test_order_keeps_leading_axes_in_front():
    # a time axis stays leading; only the spatial axes are permuted
    assert order_for_view("XZ", 4) == (0, 2, 1, 3)


def test_order_is_always_a_permutation():
    for ndim in (3, 4, 5):
        for view in ("XY", "XZ", "YZ"):
            assert sorted(order_for_view(view, ndim)) == list(range(ndim))


# ----------------------------------------------------------------- snapping


def test_snap_2d_flattens_and_orients(animation):
    animation.snap_2d("XZ")

    assert animation.viewer.dims.ndisplay == 2
    assert animation.viewer.dims.order == (1, 0, 2)
    assert len(animation.key_frames) == 1


def test_snap_3d_returns_to_volume(animation):
    animation.snap_2d("XY")
    animation.snap_3d()

    assert animation.viewer.dims.ndisplay == 3
    assert len(animation.key_frames) == 2


def test_snapping_without_capturing(animation):
    animation.snap_2d("XY", capture=False)

    assert animation.viewer.dims.ndisplay == 2
    assert len(animation.key_frames) == 0


def test_snap_2d_hides_plane_mode_slices(animation):
    """A slab layer in 2D would just redraw its source on top of itself."""
    animation.add_scene_object(OrthoSlice(name="Ortho", source="img"))
    assert animation.viewer.layers["Ortho"].visible is True

    animation.viewer.dims.ndisplay = 2
    animation.scene.apply(animation.viewer)

    assert animation.viewer.layers["Ortho"].visible is False


# ------------------------------------------------------- dims scroll-through


def test_scroll_through_sweeps_the_full_extent(animation):
    animation.snap_2d("XY", capture=False)

    animation.add_scroll_through(steps=10)

    points = _points(animation, 0)
    # z spans 10 planes at a scale of 3
    assert points[0] == pytest.approx(0.0)
    assert points[-1] == pytest.approx(27.0)
    assert points == sorted(points)
    assert 0.0 < points[len(points) // 2] < 27.0


def test_scroll_through_honours_explicit_limits(animation):
    animation.snap_2d("XY", capture=False)

    animation.add_scroll_through(axis=0, start=6.0, stop=15.0, steps=8)

    points = _points(animation, 0)
    assert points[0] == pytest.approx(6.0)
    assert points[-1] == pytest.approx(15.0)


def test_scroll_through_picks_the_slider_axis(animation):
    """With Y sliced, the sweep should walk Y, not Z."""
    animation.snap_2d("XZ", capture=False)

    animation.add_scroll_through(steps=6)

    points = _points(animation, 1)
    assert points[0] != points[-1]
    # the other axes are untouched
    assert len(set(_points(animation, 0))) == 1


def test_scroll_through_changes_nothing_but_the_slider(animation):
    animation.snap_2d("XY", capture=False)
    animation.viewer.camera.zoom = 3.0

    first, last = animation.add_scroll_through(steps=5)

    # compare through the JSON encoder, which flattens the numpy arrays that
    # make a plain == on layer state ambiguous
    from napari_animation.io import _to_builtin

    before = first.viewer_state
    after = last.viewer_state
    assert _to_builtin(before.camera) == _to_builtin(after.camera)
    assert _to_builtin(before.layers) == _to_builtin(after.layers)
    assert _to_builtin(before.dims["point"]) != _to_builtin(
        after.dims["point"]
    )


def test_scroll_through_returns_its_keyframes(animation):
    first, last = animation.add_scroll_through(steps=4)

    assert list(animation.key_frames) == [first, last]


def test_enter_steps_controls_the_approach(animation):
    animation.capture_keyframe()
    animation.add_scroll_through(steps=10, enter_steps=7)

    assert animation.key_frames[1].steps == 7
    assert animation.key_frames[2].steps == 10


# ------------------------------------------------------------- slice sweeps


def test_slice_sweep_crosses_the_whole_volume(animation):
    slice_ = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img", normal=VIEW_NORMALS["XY"])
    )

    animation.add_slice_sweep(slice_, steps=10)

    positions = [
        animation._frames[i].scene[slice_.id]["position"][0]
        for i in range(len(animation._frames))
    ]
    # z spans 0..27 in world units: 10 planes at a scale of 3, measured
    # centre-to-centre
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(27.0)
    assert positions == sorted(positions)
    assert 0.0 < positions[len(positions) // 2] < 27.0


def test_slice_sweep_moves_along_an_oblique_normal(animation):
    """The point of sweeping along the normal rather than an axis."""
    slice_ = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img", normal=(1.0, 1.0, 0.0))
    )

    first, last = animation.add_slice_sweep(slice_, steps=6)

    start = np.array(first.viewer_state.scene[slice_.id]["position"])
    stop = np.array(last.viewer_state.scene[slice_.id]["position"])
    travel = stop - start
    # motion is purely along the (normalised) normal
    direction = travel / np.linalg.norm(travel)
    assert direction == pytest.approx(
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2), abs=1e-6
    )


def test_slice_sweep_honours_explicit_limits(animation):
    slice_ = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img", normal=(1.0, 0.0, 0.0))
    )

    first, last = animation.add_slice_sweep(slice_, start=6.0, stop=12.0)

    assert first.viewer_state.scene[slice_.id]["position"][0] == (
        pytest.approx(6.0)
    )
    assert last.viewer_state.scene[slice_.id]["position"][0] == (
        pytest.approx(12.0)
    )


def test_slice_sweep_drives_the_backing_layer(animation):
    slice_ = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img", normal=VIEW_NORMALS["XY"])
    )
    animation.add_slice_sweep(slice_, steps=10)

    middle = len(animation._frames) // 2
    animation._frames[middle].apply(animation.viewer)

    plane_z = animation.viewer.layers["Ortho"].plane.position[0]
    assert 0.0 < plane_z < 10.0


def test_slice_sweep_rejects_a_zero_normal(animation):
    slice_ = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img", normal=(0.0, 0.0, 0.0))
    )

    with pytest.raises(ValueError, match="zero normal"):
        animation.add_slice_sweep(slice_)


def test_slice_sweep_needs_a_source_layer(animation):
    slice_ = OrthoSlice(name="Orphan", source="missing")
    animation.scene.append(slice_)

    with pytest.raises(ValueError, match="no source layer"):
        animation.add_slice_sweep(slice_)


def test_camera_can_orbit_while_the_slice_sweeps(animation):
    """The Imaris effect: section travelling while the view moves."""
    slice_ = animation.add_scene_object(
        OrthoSlice(name="Ortho", source="img", normal=VIEW_NORMALS["XY"])
    )
    low, high, _, _ = animation._sweep_geometry(slice_)

    animation.viewer.camera.angles = (0.0, 0.0, 90.0)
    slice_.position = (low, 0.0, 0.0)
    animation.capture_keyframe(steps=1)

    animation.viewer.camera.angles = (-20.0, 45.0, 70.0)
    slice_.position = (high, 0.0, 0.0)
    animation.capture_keyframe(steps=10)

    middle = len(animation._frames) // 2
    state = animation._frames[middle]
    assert not np.allclose(state.camera["angles"], (0.0, 0.0, 90.0))
    assert low < state.scene[slice_.id]["position"][0] < high
