import json

import numpy as np
import pytest
from napari.components import ViewerModel
from napari.components.camera import Camera

from napari_animation import Animation
from napari_animation.routines import (
    VIEW_NORMALS,
    apply_routine,
    available_routines,
    head_on_camera,
    load_routine,
    ortho_clip_pan_routine,
    routine_from_keyframes,
    routine_summary,
    save_routine,
    swung_camera,
    validate_routine,
)


class HeadlessViewer(ViewerModel):
    def screenshot(self, *args, **kwargs):
        return np.zeros((60, 60, 4), dtype=np.uint8)


def _animation(shape=(20, 30, 40), scale=(2.0, 1.0, 1.0)):
    viewer = HeadlessViewer()
    for name in ("C1", "C2"):
        viewer.add_image(np.random.random(shape), name=name, scale=scale)
    return Animation(viewer)


def _view(angles):
    return np.asarray(Camera(angles=angles).view_direction)


def test_builtin_routine_pans_each_axis_and_pairs():
    routine = ortho_clip_pan_routine()
    validate_routine(routine)
    frames = routine["keyframes"]
    assert [s["render_mode"] for s in routine["sections"]] == ["clip"] * 3
    for key in VIEW_NORMALS:
        alone = [k for k in frames if k["name"].startswith(f"{key} ")]
        # two round trips: 0 -> 1 -> 0 head-on, then 0 -> 1 -> 0 swinging
        assert [k["sections"][key] for k in alone] == [0, 1, 0, 0, 1, 0]
        others = [o for o in VIEW_NORMALS if o != key]
        assert all(k["sections"][o] == 0.5 for k in alone for o in others)
    pairs = [k for k in frames if "+" in k["name"]]
    assert len(pairs) == 3 * 5
    assert routine_summary(routine).startswith(
        "33 keyframes, 6049 frames (252.0 s at 24 fps)"
    )
    # 10 s passes at 24 fps
    assert frames[1]["steps"] == 240


def test_swing_is_fifteen_degrees_about_the_head_on_view():
    head_on = head_on_camera(VIEW_NORMALS["XZ"])
    for degrees in (-15, 15):
        swung = swung_camera(head_on, degrees)
        cos = np.dot(head_on["view"], swung["view"])
        assert np.degrees(np.arccos(cos)) == pytest.approx(15, abs=1e-6)


def test_apply_builds_keyframes_sized_to_the_data():
    animation = _animation()
    routine = ortho_clip_pan_routine()
    apply_routine(animation, routine)
    viewer = animation.viewer
    assert len(animation.key_frames) == len(routine["keyframes"])
    assert len(animation._frames) == 6049
    assert viewer.dims.ndisplay == 3
    xy, xz, yz = animation.scene
    assert {s.render_mode for s in animation.scene} == {"clip"}
    # every section is a composite of both channels
    assert all(len(s.channels) == 2 for s in animation.scene)
    bounds = viewer.layers["C1"].extent.world[:, -3:]
    size = bounds[1] - bounds[0]
    assert xy.thickness == pytest.approx(0.1 * size[0])
    assert yz.thickness == pytest.approx(0.1 * size[2])
    first, far = (k.viewer_state for k in animation.key_frames[:2])
    assert first.scene[xy.id]["position"][0] == pytest.approx(bounds[0, 0])
    assert far.scene[xy.id]["position"][0] == pytest.approx(bounds[1, 0])
    assert far.scene[xz.id]["position"][1] == pytest.approx(bounds.mean(0)[1])
    # head-on to XY means looking along world Z
    np.testing.assert_allclose(
        np.abs(_view(first.camera["angles"])), (1, 0, 0), atol=1e-6
    )
    xz_start = animation.key_frames[6].viewer_state
    np.testing.assert_allclose(
        np.abs(_view(xz_start.camera["angles"])), (0, 1, 0), atol=1e-6
    )


def test_saved_routine_round_trips_and_applies_to_new_data(tmp_path):
    animation = _animation()
    apply_routine(animation, ortho_clip_pan_routine(pass_seconds=0.5))
    routine = routine_from_keyframes(animation, "My pans")
    path = save_routine(routine, tmp_path / "mine.json")
    loaded = load_routine(path)
    assert loaded["name"] == "My pans"
    assert {"My pans", "Ortho clip pans"} <= set(available_routines(tmp_path))
    original = ortho_clip_pan_routine(pass_seconds=0.5)["keyframes"]
    for saved, built in zip(loaded["keyframes"], original):
        assert saved["steps"] == built["steps"]
        for key, fraction in built["sections"].items():
            assert saved["sections"][f"{key} slab"] == pytest.approx(fraction)
        np.testing.assert_allclose(
            saved["camera"]["view"], built["camera"]["view"], atol=1e-6
        )

    other = _animation(shape=(10, 50, 25), scale=(1.0, 0.5, 2.0))
    apply_routine(other, loaded)
    bounds = other.viewer.layers["C1"].extent.world[:, -3:]
    far = other.key_frames[1].viewer_state.scene
    xy = next(p for p in far.values() if p["name"] == "XY slab")
    assert xy["position"][0] == pytest.approx(bounds[1, 0])


def test_invalid_routines_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        validate_routine({"format": "something else"})
    bad = ortho_clip_pan_routine()
    bad["keyframes"][0]["sections"]["nope"] = 0.5
    with pytest.raises(ValueError):
        validate_routine(bad)
    (tmp_path / "broken.json").write_text(json.dumps({"x": 1}))
    assert "broken" not in available_routines(tmp_path)


def test_apply_needs_a_volume():
    viewer = HeadlessViewer()
    viewer.add_image(np.zeros((10, 10)))
    with pytest.raises(ValueError, match="3D image"):
        apply_routine(Animation(viewer), ortho_clip_pan_routine())


def test_routines_panel_applies_and_saves(qtbot, monkeypatch, tmp_path):
    from napari_animation import routines
    from napari_animation._qt import AnimationWidget
    from napari_animation._qt import routines_widget

    monkeypatch.setattr(routines, "routines_dir", lambda: tmp_path)
    animation = _animation()
    widget = AnimationWidget(animation.viewer)
    qtbot.addWidget(widget)
    panel = widget.routinesWidget
    assert panel.routineComboBox.currentText() == "Ortho clip pans"
    assert "252.0 s at 24 fps" in panel.descriptionLabel.text()
    panel.passSpinBox.setValue(5.0)
    assert "132.0 s at 24 fps" in panel.descriptionLabel.text()
    panel.thicknessSpinBox.setValue(20.0)

    panel.applyButton.click()
    assert len(widget.animation.key_frames) == 33
    xy = widget.animation.scene[0]
    size = np.ptp(animation.viewer.layers["C1"].extent.world[:, -3])
    assert xy.thickness == pytest.approx(0.2 * size)

    monkeypatch.setattr(
        routines_widget.QInputDialog,
        "getText",
        staticmethod(lambda *a, **k: ("Copy", True)),
    )
    panel.saveButton.click()
    assert (tmp_path / "Copy.json").exists()
    assert panel.routineComboBox.currentText() == "Copy"

    # applying over existing keyframes asks first; "Cancel" keeps them
    monkeypatch.setattr(panel, "_confirm", lambda *a: False)
    widget.animation.key_frames.pop()
    panel.applyButton.click()
    assert len(widget.animation.key_frames) == 32


def test_thickness_override_applies_to_every_section():
    animation = _animation()
    apply_routine(animation, ortho_clip_pan_routine(), thickness=0.25)
    bounds = animation.viewer.layers["C1"].extent.world[:, -3:]
    size = bounds[1] - bounds[0]
    for section, axis in zip(animation.scene, range(3)):
        assert section.thickness == pytest.approx(0.25 * size[axis])
    with pytest.raises(ValueError):
        apply_routine(animation, ortho_clip_pan_routine(), thickness=1.5)
