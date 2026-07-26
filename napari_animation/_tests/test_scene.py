"""Tests for animatable scene objects and the clipping-plane compositor.

The central behaviour under test is that clipping planes *accumulate*. Every
source of clipping geometry contributes planes and a single compositor unions
them, so N cutaway planes -- and an optical section alongside them -- coexist
instead of overwriting one another.

All tests run against ``ViewerModel``, so no OpenGL context is needed.
"""

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation, ViewerState
from napari_animation.ortho_slicer import OrthoSlicer
from napari_animation.scene import (
    ClipPlane,
    Scene,
    SceneObject,
    apply_scene_state,
    scene_contributions,
)


class HeadlessViewer(ViewerModel):
    """A ViewerModel that can stand in for a real viewer when capturing.

    Capturing a keyframe takes a screenshot for the thumbnail, which needs a
    canvas. Returning a blank image keeps the whole scene/keyframe API
    testable without an OpenGL context.
    """

    def screenshot(self, *args, **kwargs):
        return np.zeros((60, 60, 4), dtype=np.uint8)


@pytest.fixture
def viewer():
    viewer = HeadlessViewer()
    viewer.add_image(
        np.random.random((10, 20, 20)), name="img", scale=(3, 1, 1)
    )
    return viewer


def _planes(viewer, name="img"):
    return viewer.layers[name].experimental_clipping_planes


# ------------------------------------------------------------------ the model


def test_clip_plane_round_trips_through_a_dict():
    plane = ClipPlane(
        name="cutaway", position=(1.0, 2.0, 3.0), normal=(0.0, 1.0, 0.0)
    )

    restored = SceneObject.from_dict(plane.to_dict())

    assert isinstance(restored, ClipPlane)
    assert restored == plane
    # identity survives, which is what lets interpolation match across frames
    assert restored.id == plane.id


def test_unknown_object_kinds_are_rejected():
    with pytest.raises(ValueError, match="Unknown scene object kind"):
        SceneObject.from_dict({"kind": "from_the_future", "name": "x"})


def test_flipping_a_plane_reverses_its_normal():
    plane = ClipPlane(normal=(0.0, 1.0, 0.0))

    flipped = plane.flipped()

    assert flipped.normal == (0.0, -1.0, 0.0)
    assert flipped.id == plane.id


def test_positions_convert_into_layer_data_coordinates(viewer):
    # the layer is scaled 3x along z, so world z=9 is data z=3
    plane = ClipPlane(position=(9.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))

    ((_, planes),) = plane.clipping_planes(viewer)

    assert planes[0]["position"][0] == pytest.approx(3.0)


def test_normals_use_the_inverse_transpose(viewer):
    """Under anisotropic scale, normals do not transform like positions."""
    plane = ClipPlane(normal=(1.0, 1.0, 0.0))

    ((_, planes),) = plane.clipping_planes(viewer)
    normal = np.array(planes[0]["normal"])

    assert np.linalg.norm(normal) == pytest.approx(1.0)
    # a position would divide z by the scale; a normal multiplies by it, so
    # the z component grows relative to y rather than shrinking
    assert normal[0] > normal[1]


def test_plane_targets_restrict_which_layers_are_cut(viewer):
    viewer.add_image(np.zeros((10, 20, 20)), name="other")
    plane = ClipPlane(targets=("img",))

    contributions = scene_contributions(viewer, {plane.id: plane.to_dict()})

    assert set(contributions) == {"img"}


# ------------------------------------------------------------- the compositor


def test_many_clipping_planes_accumulate(viewer):
    """The headline requirement: N planes cut the same volume."""
    scene = Scene()
    for i in range(4):
        scene.append(ClipPlane(name=f"cut {i}", position=(float(i), 0.0, 0.0)))

    scene.apply(viewer)

    assert len(_planes(viewer)) == 4


def test_disabled_objects_contribute_nothing(viewer):
    scene = Scene()
    scene.append(ClipPlane(name="on"))
    scene.append(ClipPlane(name="off", enabled=False))

    scene.apply(viewer)

    assert len(_planes(viewer)) == 1


def test_ortho_slab_composites_with_cutaway_planes(viewer):
    """Regression: an optical section must not wipe the cutaway planes.

    Both sources assign to the same layer property, so before compositing
    whichever ran last silently erased the other.
    """
    viewer.dims.current_step = (5, 9, 9)
    scene = Scene()
    scene.append(ClipPlane(name="cut"))

    ortho = OrthoSlicer(
        enabled=True, thickness=4, mode="clip", axis=0
    ).to_dict()
    apply_scene_state(viewer, scene.to_dict(), ortho=ortho)

    # one cutaway plane plus the two planes bounding the slab
    assert len(_planes(viewer)) == 3


def test_compositor_clears_planes_when_the_scene_empties(viewer):
    scene = Scene()
    plane = ClipPlane(name="cut")
    scene.append(plane)
    scene.apply(viewer)
    assert len(_planes(viewer)) == 1

    scene.remove(plane)
    scene.apply(viewer)

    assert len(_planes(viewer)) == 0


def test_adopting_existing_planes_keeps_them(viewer):
    """Planes set by hand survive the compositor becoming the only writer."""
    viewer.layers["img"].experimental_clipping_planes = [
        {"position": (3.0, 0.0, 0.0), "normal": (1.0, 0.0, 0.0)}
    ]
    scene = Scene()

    adopted = scene.adopt_existing_clipping_planes(viewer)
    scene.apply(viewer)

    assert len(adopted) == 1
    assert len(_planes(viewer)) == 1
    # round trip through world coordinates lands back where it started
    assert _planes(viewer)[0].position[0] == pytest.approx(3.0)


# ------------------------------------------------------------- keyframing


def test_scene_geometry_sweeps_between_keyframes(viewer):
    from napari_animation.frame_sequence import FrameSequence
    from napari_animation.key_frame import KeyFrame, KeyFrameList

    plane = ClipPlane(name="cut", position=(0.0, 0.0, 0.0))
    start = plane.to_dict()
    end = {**plane.to_dict(), "position": (9.0, 0.0, 0.0)}

    thumb = np.zeros((30, 30, 4), dtype=np.uint8)
    key_frames = KeyFrameList()
    key_frames.append(
        KeyFrame(
            viewer_state=ViewerState.from_viewer(
                viewer, scene={plane.id: start}
            ),
            thumbnail=thumb,
            steps=1,
        )
    )
    key_frames.append(
        KeyFrame(
            viewer_state=ViewerState.from_viewer(
                viewer, scene={plane.id: end}
            ),
            thumbnail=thumb,
            steps=10,
        )
    )
    frames = FrameSequence(key_frames)

    positions = [
        frames[i].scene[plane.id]["position"][0] for i in range(len(frames))
    ]

    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(9.0)
    assert positions == sorted(positions)
    # swept, not snapped
    assert 0.0 < positions[len(positions) // 2] < 9.0

    # and the swept state drives a real viewer
    frames[len(frames) // 2].apply(viewer)
    assert 0.0 < _planes(viewer)[0].position[0] < 3.0


def test_captured_layer_state_defers_to_the_scene(viewer):
    """With a scene present, per-layer planes are not captured separately."""
    viewer.layers["img"].experimental_clipping_planes = [
        {"position": (1.0, 0.0, 0.0), "normal": (1.0, 0.0, 0.0)}
    ]
    plane = ClipPlane(name="cut")

    state = ViewerState.from_viewer(viewer, scene={plane.id: plane.to_dict()})

    assert "experimental_clipping_planes" not in state.layers["img"]


def test_no_scene_leaves_clipping_planes_alone(viewer):
    """Without a scene the previous per-layer behaviour is unchanged."""
    viewer.layers["img"].experimental_clipping_planes = [
        {"position": (1.0, 0.0, 0.0), "normal": (1.0, 0.0, 0.0)}
    ]

    state = ViewerState.from_viewer(viewer)

    assert len(state.layers["img"]["experimental_clipping_planes"]) == 1


# ------------------------------------------------------ the animation API


def test_added_objects_are_backfilled_into_existing_keyframes(viewer):
    """A newly added plane should not vanish when scrubbing to older frames."""
    animation = Animation(viewer)
    animation.capture_keyframe()
    animation.capture_keyframe()

    plane = animation.add_scene_object(ClipPlane(name="cut"))

    for key_frame in animation.key_frames:
        assert plane.id in key_frame.viewer_state.scene


def test_objects_toggle_independently_per_keyframe(viewer):
    animation = Animation(viewer)
    animation.capture_keyframe()
    animation.capture_keyframe()
    plane = animation.add_scene_object(ClipPlane(name="cut"))

    animation.set_object_enabled(0, plane.id, False)

    assert (
        animation.key_frames[0].viewer_state.scene[plane.id]["enabled"]
        is False
    )
    assert (
        animation.key_frames[1].viewer_state.scene[plane.id]["enabled"] is True
    )

    # applying each keyframe cuts, or does not cut, the volume accordingly
    animation.key_frames[0].viewer_state.apply(viewer)
    assert len(_planes(viewer)) == 0
    animation.key_frames[1].viewer_state.apply(viewer)
    assert len(_planes(viewer)) == 1


def test_toggling_invalidates_the_interpolation_cache(viewer):
    animation = Animation(viewer)
    animation.capture_keyframe()
    animation.capture_keyframe()
    plane = animation.add_scene_object(ClipPlane(name="cut"))

    # populate the cache
    _ = animation._frames[0]
    animation.set_object_enabled(0, plane.id, False)

    assert animation._frames[0].scene[plane.id]["enabled"] is False


def test_toggling_an_unknown_object_is_an_error(viewer):
    animation = Animation(viewer)
    animation.capture_keyframe()

    with pytest.raises(KeyError):
        animation.set_object_enabled(0, "not-an-id", False)


def test_layers_toggle_per_keyframe(viewer):
    animation = Animation(viewer)
    animation.capture_keyframe()
    animation.capture_keyframe()

    animation.set_layer_visible(1, "img", False)

    animation.key_frames[0].viewer_state.apply(viewer)
    assert viewer.layers["img"].visible is True
    animation.key_frames[1].viewer_state.apply(viewer)
    assert viewer.layers["img"].visible is False


def test_enabled_switches_on_arrival_not_mid_transition(viewer):
    animation = Animation(viewer)
    animation.capture_keyframe(steps=5)
    animation.capture_keyframe(steps=5)
    plane = animation.add_scene_object(ClipPlane(name="cut"))
    animation.set_object_enabled(1, plane.id, False)

    states = [
        animation._frames[i].scene[plane.id]["enabled"]
        for i in range(len(animation._frames))
    ]

    # on for the whole transition, off only at the destination keyframe
    assert states[0] is True
    assert states[-1] is False
    assert states.count(False) == 1
