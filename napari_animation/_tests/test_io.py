import json

import numpy as np
import pytest
import tifffile
from napari.components import ViewerModel

from napari_animation import Animation, KeyFrame, ViewerState
from napari_animation.easing import Easing
from napari_animation.io import (
    _decode_thumbnail,
    _encode_thumbnail,
    _to_builtin,
    animation_to_dict,
)


def _make_keyframe(
    viewer, name="kf", steps=10, ease=Easing.CUBIC, ortho=None, scene=None
):
    """Build a KeyFrame without a screenshot (no GL required)."""
    return KeyFrame(
        viewer_state=ViewerState.from_viewer(viewer, ortho=ortho, scene=scene),
        thumbnail=np.random.randint(0, 255, (30, 30, 4), dtype=np.uint8),
        steps=steps,
        ease=ease,
        name=name,
    )


@pytest.fixture
def model_animation():
    viewer = ViewerModel()
    viewer.add_image(np.random.random((6, 20, 20)), name="img")
    return Animation(viewer)


def test_to_builtin_is_json_serialisable(model_animation):
    state = ViewerState.from_viewer(model_animation.viewer)
    payload = {
        "camera": _to_builtin(state.camera),
        "dims": _to_builtin(state.dims),
        "layers": _to_builtin(state.layers),
    }
    # round trips through json without error and preserves structure
    restored = json.loads(json.dumps(payload))
    assert set(restored["camera"]) == set(state.camera)
    assert set(restored["dims"]) == set(state.dims)
    assert "img" in restored["layers"]


def test_scene_survives_a_save_load_round_trip(tmp_path, model_animation):
    """Clipping planes must come back as usable objects, not strings."""
    from napari_animation.scene import ClipPlane

    plane = ClipPlane(
        name="cutaway", position=(2.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0)
    )
    model_animation.scene.append(plane)
    scene = model_animation.scene.to_dict()
    model_animation.key_frames.append(
        _make_keyframe(model_animation.viewer, scene=scene)
    )
    model_animation.key_frames.append(
        _make_keyframe(model_animation.viewer, scene=scene)
    )

    path = model_animation.save_keyframes(tmp_path / "animation.json")
    model_animation.load_keyframes(path, reload_layers=False)

    restored = model_animation.key_frames[0].viewer_state.scene[plane.id]
    assert restored["kind"] == "clip_plane"
    assert restored["name"] == "cutaway"
    assert tuple(restored["position"]) == (2.0, 0.0, 0.0)
    # and it still composites onto a viewer after the round trip
    model_animation.key_frames[0].viewer_state.apply(model_animation.viewer)
    planes = model_animation.viewer.layers["img"].experimental_clipping_planes
    assert len(planes) == 1


def test_keyframes_without_a_scene_still_load(tmp_path, model_animation):
    """Files written before scene objects existed must keep working."""
    model_animation.key_frames.append(_make_keyframe(model_animation.viewer))
    model_animation.key_frames.append(_make_keyframe(model_animation.viewer))
    path = model_animation.save_keyframes(tmp_path / "old.json")

    data = json.loads(path.read_text())
    for key_frame in data["key_frames"]:
        key_frame["viewer_state"].pop("scene")
    path.write_text(json.dumps(data))

    model_animation.load_keyframes(path, reload_layers=False)

    assert model_animation.key_frames[0].viewer_state.scene is None


def test_thumbnail_round_trip():
    thumb = np.random.randint(0, 255, (30, 30, 4), dtype=np.uint8)
    decoded = _decode_thumbnail(_encode_thumbnail(thumb))
    np.testing.assert_array_equal(decoded, thumb)


def test_empty_thumbnail():
    assert _decode_thumbnail("").shape == (30, 30, 4)


def test_animation_to_dict_structure(model_animation):
    model_animation.key_frames.append(_make_keyframe(model_animation.viewer))
    data = animation_to_dict(model_animation)
    assert data["version"] == "1"
    assert [r["name"] for r in data["layers"]] == ["img"]
    assert len(data["key_frames"]) == 1
    assert data["key_frames"][0]["ease"] == "CUBIC"


def test_save_load_round_trip(tmp_path, model_animation):
    viewer = model_animation.viewer
    model_animation.key_frames.append(
        _make_keyframe(viewer, name="A", steps=7, ease=Easing.QUADRATIC)
    )
    viewer.camera.zoom *= 2
    model_animation.key_frames.append(
        _make_keyframe(
            viewer,
            name="B",
            steps=12,
            ease=Easing.LINEAR,
            ortho={
                "enabled": True,
                "thickness": 3,
                "mode": "projection",
                "projection_mode": "max",
                "axis": 0,
            },
        )
    )

    path = model_animation.save_keyframes(tmp_path / "anim")
    assert path.suffix == ".json"

    # load into a fresh animation that already has a matching layer
    viewer2 = ViewerModel()
    viewer2.add_image(np.random.random((6, 20, 20)), name="img")
    animation2 = Animation(viewer2)
    animation2.load_keyframes(path, reload_layers=False)

    assert len(animation2.key_frames) == 2
    kf_a, kf_b = animation2.key_frames
    assert kf_a.name == "A"
    assert kf_a.steps == 7
    assert kf_a.ease is Easing.QUADRATIC
    assert kf_b.ease is Easing.LINEAR
    assert kf_b.viewer_state.ortho["enabled"] is True
    assert kf_b.viewer_state.ortho["thickness"] == 3


def test_load_replaces_existing_keyframes(tmp_path, model_animation):
    model_animation.key_frames.append(_make_keyframe(model_animation.viewer))
    path = model_animation.save_keyframes(tmp_path / "anim.json")

    model_animation.key_frames.append(_make_keyframe(model_animation.viewer))
    model_animation.key_frames.append(_make_keyframe(model_animation.viewer))
    assert len(model_animation.key_frames) == 3

    model_animation.load_keyframes(path, reload_layers=False)
    assert len(model_animation.key_frames) == 1


def test_reload_layers_from_disk(tmp_path, monkeypatch):
    # write a real image to disk so the recorded source path exists on load
    img_path = tmp_path / "img.tif"
    tifffile.imwrite(img_path, np.random.randint(0, 255, (6, 20, 20), "uint8"))

    viewer = ViewerModel()
    viewer.add_image(np.random.random((6, 20, 20)), name="img")
    animation = Animation(viewer)
    animation.key_frames.append(_make_keyframe(viewer, name="A"))
    path = animation.save_keyframes(tmp_path / "anim.json")

    # inject the recorded source path (this layer was created in-memory, so its
    # source path is None; a real layer loaded from a file would have it set).
    data = json.loads(path.read_text())
    data["layers"][0]["path"] = str(img_path)
    path.write_text(json.dumps(data))

    # fresh viewer with NO layers; stub viewer.open since this headless test
    # environment has no file-reader plugin registered.
    viewer2 = ViewerModel()

    def fake_open(self, p, *args, **kwargs):
        from pathlib import Path

        self.add_image(np.zeros((6, 20, 20)), name=Path(p).stem)
        return [self.layers[-1]]

    monkeypatch.setattr(type(viewer2), "open", fake_open, raising=False)

    animation2 = Animation(viewer2)
    animation2.load_keyframes(path, reload_layers=True)
    assert "img" in viewer2.layers


def test_version_mismatch_raises(tmp_path, model_animation):
    model_animation.key_frames.append(_make_keyframe(model_animation.viewer))
    path = model_animation.save_keyframes(tmp_path / "anim.json")
    data = json.loads(path.read_text())
    data["version"] = "999"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported animation file version"):
        model_animation.load_keyframes(path, reload_layers=False)
