"""Tests for the animatable-state foundations.

These cover the pieces an Imaris-style animation depends on: capturing the
display state that describes *how* a volume is drawn (most importantly the
``plane`` optical section), interpolating structured values field by field
instead of snapping, and applying a state in an order that survives a 2D/3D
switch.

Everything here runs against ``ViewerModel``, which has real dims, camera and
layers but no canvas, so no OpenGL context is required.
"""

from dataclasses import replace

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import KeyFrame, ViewerState
from napari_animation.frame_sequence import (
    FrameSequence,
    resolve_interpolation,
)
from napari_animation.interpolation import (
    Interpolation,
    default,
    interpolate_dict,
    interpolate_seq,
    slerp_vector,
    step_end,
    step_start,
)
from napari_animation.key_frame import KeyFrameList
from napari_animation.viewer_state import ANIMATABLE_LAYER_PROPERTIES

THUMB = np.zeros((30, 30, 4), dtype=np.uint8)


@pytest.fixture
def viewer():
    viewer = ViewerModel()
    viewer.add_image(
        np.random.random((10, 20, 20)), name="img", scale=(3, 1, 1)
    )
    return viewer


def _sequence(*states, steps=10):
    """Build a FrameSequence from a series of viewer states."""
    key_frames = KeyFrameList()
    for i, state in enumerate(states):
        key_frames.append(
            KeyFrame(
                viewer_state=state,
                thumbnail=THUMB,
                steps=1 if i == 0 else steps,
            )
        )
    return FrameSequence(key_frames)


# --------------------------------------------------------------- interpolation


def test_dicts_interpolate_field_by_field():
    a = {"position": (0.0, 0.0, 0.0), "thickness": 1.0}
    b = {"position": (10.0, 0.0, 0.0), "thickness": 11.0}

    halfway = interpolate_dict(a, b, 0.5)

    assert halfway["position"] == pytest.approx((5.0, 0.0, 0.0))
    assert halfway["thickness"] == pytest.approx(6.0)


def test_dict_keys_missing_from_one_side_are_carried_through():
    a = {"shared": 0.0, "only_a": "x"}
    b = {"shared": 10.0, "only_b": "y"}

    result = interpolate_dict(a, b, 0.5)

    assert result["shared"] == pytest.approx(5.0)
    assert result["only_a"] == "x"
    assert result["only_b"] == "y"


def test_clipping_planes_sweep_rather_than_snap():
    """A list of plane dicts must interpolate, not jump to the target."""
    a = [{"position": (0.0, 0.0, 0.0), "normal": (1.0, 0.0, 0.0)}]
    b = [{"position": (10.0, 0.0, 0.0), "normal": (1.0, 0.0, 0.0)}]

    halfway = default(a, b, 0.5)

    assert halfway[0]["position"] == pytest.approx((5.0, 0.0, 0.0))


def test_unequal_length_sequences_hold_instead_of_truncating():
    """Going from two planes to one must not drop a plane mid-transition."""
    a = [{"position": (0.0,)}, {"position": (1.0,)}]
    b = [{"position": (10.0,)}]

    halfway = default(a, b, 0.5)

    assert len(halfway) == 2
    assert halfway[0]["position"] == pytest.approx((5.0,))
    # the plane with no counterpart is held, not discarded
    assert halfway[1]["position"] == pytest.approx((1.0,))


def test_interpolate_seq_preserves_type_and_length():
    assert interpolate_seq((0.0, 0.0), (10.0, 20.0), 0.5) == (5.0, 10.0)
    assert interpolate_seq([0.0], [10.0], 0.25) == [2.5]


def test_slerp_vector_rotates_along_the_arc():
    """A 90 degree turn passes through the diagonal, staying unit length."""
    halfway = slerp_vector((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5)

    assert np.linalg.norm(halfway) == pytest.approx(1.0)
    assert halfway == pytest.approx(
        (np.sqrt(0.5), np.sqrt(0.5), 0.0), abs=1e-9
    )


def test_slerp_vector_beats_linear_at_midpoint():
    """Linear interpolation shrinks a rotating normal; slerp does not."""
    a, b = (1.0, 0.0, 0.0), (-1.0, 1.0, 0.0)
    assert np.linalg.norm(slerp_vector(a, b, 0.5)) > np.linalg.norm(
        interpolate_seq(a, b, 0.5)
    )


def test_slerp_vector_endpoints_and_degenerate_inputs():
    a, b = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    assert slerp_vector(a, b, 0.0) == pytest.approx(a)
    assert slerp_vector(a, b, 1.0) == pytest.approx(b)
    # opposed vectors still produce a unit-length, continuous result
    opposed = slerp_vector((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 0.5)
    assert np.linalg.norm(opposed) == pytest.approx(1.0)
    # a zero vector has no direction; fall back rather than divide by zero
    assert slerp_vector(
        (0.0, 0.0, 0.0), (0.0, 0.0, 2.0), 0.5
    ) == pytest.approx((0.0, 0.0, 1.0))


def test_slerp_vector_preserves_sequence_type():
    assert isinstance(
        slerp_vector((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5), tuple
    )
    assert isinstance(
        slerp_vector([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.5), list
    )


def test_step_interpolators_switch_at_opposite_ends():
    assert step_start("a", "b", 0.0) == "a"
    assert step_start("a", "b", 0.01) == "b"

    assert step_end("a", "b", 0.99) == "a"
    assert step_end("a", "b", 1.0) == "b"


# ------------------------------------------------------- interpolation lookup


def test_glob_patterns_match_state_paths():
    interpolation_map = {"layers.*.rendering": Interpolation.STEP_END}

    assert (
        resolve_interpolation(interpolation_map, "layers.img.rendering")
        is Interpolation.STEP_END
    )
    assert (
        resolve_interpolation(interpolation_map, "layers.img.gamma")
        is Interpolation.DEFAULT
    )


def test_exact_keys_beat_glob_patterns():
    interpolation_map = {
        "layers.*.gamma": Interpolation.STEP_END,
        "layers.img.gamma": Interpolation.LOG,
    }
    assert (
        resolve_interpolation(interpolation_map, "layers.img.gamma")
        is Interpolation.LOG
    )


def test_most_specific_glob_wins():
    interpolation_map = {
        "layers.*": Interpolation.STEP_START,
        "layers.img.*": Interpolation.STEP_END,
    }
    assert (
        resolve_interpolation(interpolation_map, "layers.img.gamma")
        is Interpolation.STEP_END
    )


# ------------------------------------------------------------------- capture


def test_plane_and_depiction_are_captured(viewer):
    layer = viewer.layers["img"]
    layer.depiction = "plane"
    layer.plane = {
        "position": (4, 10, 10),
        "normal": (1, 0, 0),
        "thickness": 5,
    }

    captured = ViewerState.from_viewer(viewer).layers["img"]

    assert captured["depiction"] == "plane"
    assert captured["plane"]["thickness"] == pytest.approx(5.0)
    assert captured["plane"]["position"] == pytest.approx((4.0, 10.0, 10.0))


def test_animatable_properties_are_captured(viewer):
    captured = ViewerState.from_viewer(viewer).layers["img"]

    for prop in ANIMATABLE_LAYER_PROPERTIES:
        assert prop in captured, f"{prop} was not captured"
    assert isinstance(captured["colormap"], str)


def test_layer_data_is_never_captured(viewer):
    """Capturing data would deep-copy the whole volume once per frame."""
    captured = ViewerState.from_viewer(viewer).layers["img"]

    assert "data" not in captured
    assert "metadata" not in captured


def test_capture_tolerates_layers_without_display_properties(viewer):
    viewer.add_points(np.zeros((3, 3)), name="pts")

    captured = ViewerState.from_viewer(viewer).layers["pts"]

    assert "plane" not in captured
    assert captured["visible"] is True


# --------------------------------------------------------------------- apply


def test_plane_position_sweeps_across_frames(viewer):
    """An ortho slice scrolls through the volume between two keyframes."""
    layer = viewer.layers["img"]
    layer.depiction = "plane"

    layer.plane = {
        "position": (0, 10, 10),
        "normal": (1, 0, 0),
        "thickness": 3,
    }
    start = ViewerState.from_viewer(viewer)
    layer.plane = {
        "position": (9, 10, 10),
        "normal": (1, 0, 0),
        "thickness": 3,
    }
    end = ViewerState.from_viewer(viewer)

    frames = _sequence(start, end)
    positions = [
        frames[i].layers["img"]["plane"]["position"][0]
        for i in range(len(frames))
    ]

    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(9.0)
    assert positions == sorted(positions)
    # genuinely swept rather than snapping to either end
    assert 0.0 < positions[len(positions) // 2] < 9.0

    # and a mid-transition state applies cleanly to a live viewer
    frames[len(frames) // 2].apply(viewer)
    assert 0.0 < viewer.layers["img"].plane.position[0] < 9.0


def test_plane_normal_rotates_without_collapsing(viewer):
    layer = viewer.layers["img"]
    layer.depiction = "plane"

    layer.plane = {"position": (5, 10, 10), "normal": (1, 0, 0)}
    start = ViewerState.from_viewer(viewer)
    layer.plane = {"position": (5, 10, 10), "normal": (0, 1, 0)}
    end = ViewerState.from_viewer(viewer)

    frames = _sequence(start, end)
    middle = frames[len(frames) // 2].layers["img"]["plane"]["normal"]

    # slerp keeps the normal unit length through the turn
    assert np.linalg.norm(middle) == pytest.approx(1.0, abs=1e-6)
    assert middle[0] > 0 and middle[1] > 0


def test_ndisplay_holds_until_the_destination_keyframe(viewer):
    viewer.dims.ndisplay = 3
    start = ViewerState.from_viewer(viewer)
    viewer.dims.ndisplay = 2
    end = ViewerState.from_viewer(viewer)

    frames = _sequence(start, end)
    modes = [frames[i].dims["ndisplay"] for i in range(len(frames))]

    assert modes[0] == 3
    assert modes[-1] == 2
    # a discrete switch: never a value in between, and it happens exactly once
    assert set(modes) == {2, 3}
    assert modes.count(3) == len(modes) - 1


def test_camera_survives_a_2d_to_3d_switch(viewer):
    """Regression: dims must be applied before the camera.

    Switching ndisplay makes napari restore its own cached camera for the
    mode being entered. Applying the camera first therefore loses it on every
    2D/3D transition -- the cached value silently wins. The state used here is
    an *interpolated* one, whose camera deliberately differs from anything
    napari has cached, which is exactly the case during an animation.
    """
    viewer.dims.ndisplay = 3
    viewer.camera.synced = False
    viewer.camera.zoom = 4.0
    viewer.camera.angles = (10.0, 20.0, 30.0)
    captured = ViewerState.from_viewer(viewer)

    # a frame partway through a zoom ramp, still in 3D
    frame = replace(captured, camera={**captured.camera, "zoom": 7.0})

    # leave 3D, which caches the 3D camera at zoom 4.0
    viewer.dims.ndisplay = 2

    frame.apply(viewer)

    assert viewer.dims.ndisplay == 3
    # 4.0 here would mean napari's cache overwrote the keyframe's camera
    assert viewer.camera.zoom == pytest.approx(7.0)
    assert viewer.camera.angles == pytest.approx((10.0, 20.0, 30.0))


def test_contrast_limits_and_gamma_interpolate(viewer):
    layer = viewer.layers["img"]

    layer.contrast_limits = [0.0, 1.0]
    layer.gamma = 1.0
    start = ViewerState.from_viewer(viewer)
    layer.contrast_limits = [0.0, 0.5]
    layer.gamma = 2.0
    end = ViewerState.from_viewer(viewer)

    frames = _sequence(start, end)
    middle = frames[len(frames) // 2].layers["img"]

    assert 0.5 < middle["contrast_limits"][1] < 1.0
    assert 1.0 < middle["gamma"] < 2.0
