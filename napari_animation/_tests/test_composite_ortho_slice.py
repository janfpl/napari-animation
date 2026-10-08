"""A composite section slices independent registered image channels."""

import numpy as np
from napari.components import ViewerModel

from napari_animation import Animation, KeyFrame, ViewerState
from napari_animation.layer_registry import LayerRegistry
from napari_animation.scene import OrthoSlice, Scene


def _viewer():
    viewer = ViewerModel()
    viewer.dims.ndisplay = 3
    viewer.add_image(np.ones((6, 8, 8)), name="red", colormap="red")
    viewer.add_image(np.ones((6, 8, 8)) * 2, name="green", colormap="green")
    return viewer


def _section(viewer):
    registry = LayerRegistry(viewer)
    red_id = registry.source_id(viewer.layers["red"])
    green_id = registry.source_id(viewer.layers["green"])
    section = OrthoSlice(
        name="Overlay", source="red", source_id=red_id,
        position=(3, 4, 4), thickness=2,
        channels={
            red_id: {"source_id": red_id, "enabled": True},
            green_id: {"source_id": green_id, "enabled": True},
        },
    )
    return section, red_id, green_id


def test_3d_composite_renders_two_colored_planes_and_toggles():
    viewer = _viewer()
    section, red_id, green_id = _section(viewer)
    scene = Scene()
    scene.append(section)
    scene.apply(viewer)
    registry = LayerRegistry(viewer)
    red = registry.find_managed(section.id, "section_3d")
    green = registry.find_managed(
        f"{section.id}::channel::{green_id}", "section_3d"
    )
    assert red.data is viewer.layers["red"].data
    assert green.data is viewer.layers["green"].data
    assert red.blending == green.blending == "additive"
    assert red.colormap.name == "red"
    assert green.colormap.name == "green"
    section.channels[red_id]["enabled"] = False
    scene.apply(viewer)
    assert not red.visible
    assert green.visible


def test_moving_one_plane_moves_every_channel():
    viewer = _viewer()
    section, _, green_id = _section(viewer)
    scene = Scene()
    scene.append(section)
    scene.apply(viewer)
    registry = LayerRegistry(viewer)
    red = registry.find_managed(section.id, "section_3d")
    green = registry.find_managed(
        f"{section.id}::channel::{green_id}", "section_3d"
    )
    green.plane = {
        "position": (2, 4, 4), "normal": (1, 0, 0), "thickness": 2,
    }
    assert np.isclose(section.position[0], 2)
    assert np.isclose(red.plane.position[0], 2)


def test_linked_color_updates_when_source_changes():
    viewer = _viewer()
    section, _, green_id = _section(viewer)
    scene = Scene()
    scene.append(section)
    scene.apply(viewer)
    green = LayerRegistry(viewer).find_managed(
        f"{section.id}::channel::{green_id}", "section_3d"
    )
    viewer.layers["green"].colormap = "magenta"
    assert green.colormap.name == "magenta"


def test_editing_channel_plane_creates_custom_appearance():
    viewer = _viewer()
    section, _, green_id = _section(viewer)
    scene = Scene()
    scene.append(section)
    scene.apply(viewer)
    green = LayerRegistry(viewer).find_managed(
        f"{section.id}::channel::{green_id}", "section_3d"
    )
    green.colormap = "magenta"
    assert section.channels[green_id]["appearance_mode"] == "custom"
    assert section.channels[green_id]["appearance"]["colormap"] == "magenta"


def test_2d_composite_projects_both_and_restores_source_visibility(qtbot):
    viewer = _viewer()
    animation = Animation(viewer)
    section, red_id, green_id = _section(viewer)
    animation.add_scene_object(section)
    viewer.dims.ndisplay = 2
    animation.set_active_section(section.id)
    registry = LayerRegistry(viewer)
    red = registry.find_managed(section.id, "section_2d")
    green = registry.find_managed(
        f"{section.id}::channel::{green_id}", "section_2d"
    )
    qtbot.waitUntil(lambda: registry.find_managed(section.id, "section_2d") is not None, timeout=3000)
    red = registry.find_managed(section.id, "section_2d")
    green = registry.find_managed(
        f"{section.id}::channel::{green_id}", "section_2d"
    )
    assert red is not None and green is not None
    assert red.visible and green.visible
    assert red.blending == green.blending == "additive"
    assert not viewer.layers["red"].visible
    assert not viewer.layers["green"].visible
    viewer.dims.ndisplay = 3
    animation.apply_working_state()
    assert viewer.layers["red"].visible
    assert viewer.layers["green"].visible


def test_composite_round_trips_without_serializing_derived_planes(tmp_path):
    viewer = _viewer()
    animation = Animation(viewer)
    section, _, _ = _section(viewer)
    animation.add_scene_object(section)
    state = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    assert len(state.layers) == 2
    animation.key_frames.append(KeyFrame(
        viewer_state=state,
        thumbnail=np.zeros((30, 30, 4), dtype=np.uint8),
    ))
    path = animation.save_keyframes(tmp_path / "composite.json")
    animation.load_keyframes(path, reload_layers=False)
    restored = animation.key_frames[0].viewer_state.scene[section.id]
    assert len(restored["channels"]) == 2
    assert all(binding["enabled"] for binding in restored["channels"].values())


def test_version_two_slice_migrates_its_saved_color(tmp_path):
    import json

    viewer = _viewer()
    animation = Animation(viewer)
    section = OrthoSlice(name="Old slice", source="red", position=(3, 4, 4))
    animation.add_scene_object(section)
    viewer.layers["Old slice"].colormap = "magenta"
    state = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    thumbnail = np.zeros((30, 30, 4), dtype=np.uint8)
    animation.key_frames.append(KeyFrame(state, thumbnail))
    path = animation.save_keyframes(tmp_path / "old.json")
    payload = json.loads(path.read_text())
    payload["version"] = "2"
    path.write_text(json.dumps(payload))
    animation.load_keyframes(path, reload_layers=False)
    restored = animation.key_frames[0].viewer_state
    section_state = restored.scene[section.id]
    source_id = section_state["source_id"]
    assert section_state["channels"][source_id]["appearance"]["colormap"] == "magenta"
    assert LayerRegistry.managed_id(section.id, "section_3d") not in restored.layers


def test_custom_lut_survives_save_and_load(tmp_path):
    from napari.utils.colormaps import Colormap

    viewer = _viewer()
    animation = Animation(viewer)
    section, _, green_id = _section(viewer)
    animation.add_scene_object(section)
    layer = LayerRegistry(viewer).find_managed(
        f"{section.id}::channel::{green_id}", "section_3d"
    )
    colors = [[0, 0, 0, 1], [0.2, 0.7, 0.4, 1]]
    layer.colormap = Colormap(colors=colors, name="my stains")
    state = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    animation.key_frames.append(KeyFrame(
        state, np.zeros((30, 30, 4), dtype=np.uint8)
    ))
    path = animation.save_keyframes(tmp_path / "lut.json")
    animation.load_keyframes(path, reload_layers=False)
    restored = animation.key_frames[0].viewer_state.scene[section.id]
    specification = restored["channels"][green_id]["appearance"]["colormap"]
    assert specification["name"] == "my stains"
    np.testing.assert_allclose(specification["colors"], colors)
    visible = LayerRegistry(viewer).find_managed(
        f"{section.id}::channel::{green_id}", "section_3d"
    )
    np.testing.assert_allclose(visible.colormap.colors, colors)


def test_channel_membership_changes_at_destination_keyframe():
    viewer = _viewer()
    animation = Animation(viewer)
    section, red_id, green_id = _section(viewer)
    section.channels[green_id]["enabled"] = False
    animation.add_scene_object(section)
    first = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    section.channels[green_id]["enabled"] = True
    second = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    thumbnail = np.zeros((30, 30, 4), dtype=np.uint8)
    animation.key_frames.append(KeyFrame(first, thumbnail, steps=5))
    animation.key_frames.append(KeyFrame(second, thumbnail, steps=5))
    states = [
        animation._frames[index].scene[section.id]["channels"][green_id]["enabled"]
        for index in range(len(animation._frames))
    ]
    assert states == [False] * (len(states) - 1) + [True]


def test_2d_projection_publishes_a_complete_batch(qtbot, monkeypatch):
    import time

    from napari_animation.scene import processing

    original = processing.project_slab

    def slow_projection(*args, **kwargs):
        time.sleep(0.12)
        return original(*args, **kwargs)

    monkeypatch.setattr(processing, "project_slab", slow_projection)
    viewer = _viewer()
    animation = Animation(viewer)
    viewer._napari_animation_projection_service = processing.ProjectionService(
        asynchronous=True
    )
    section, _, green_id = _section(viewer)
    animation.add_scene_object(section)
    viewer.dims.ndisplay = 2
    animation.set_active_section(section.id)
    registry = LayerRegistry(viewer)
    qtbot.waitUntil(
        lambda: len(section._composite_batch["results"]) == 1,
        timeout=3000,
    )
    assert registry.find_managed(section.id, "section_2d") is None
    assert registry.find_managed(
        f"{section.id}::channel::{green_id}", "section_2d"
    ) is None
    qtbot.waitUntil(
        lambda: registry.find_managed(section.id, "section_2d") is not None,
        timeout=3000,
    )
    assert registry.find_managed(
        f"{section.id}::channel::{green_id}", "section_2d"
    ) is not None


def test_failed_channel_does_not_publish_partial_overlay(qtbot, monkeypatch):
    from napari_animation.scene import processing

    original = processing.project_slab

    def fail_green(volume, *args, **kwargs):
        if np.asarray(volume).max() > 1:
            raise RuntimeError("green read failed")
        return original(volume, *args, **kwargs)

    monkeypatch.setattr(processing, "project_slab", fail_green)
    viewer = _viewer()
    animation = Animation(viewer)
    service = processing.ProjectionService(asynchronous=True)
    viewer._napari_animation_projection_service = service
    section, _, _ = _section(viewer)
    animation.add_scene_object(section)
    viewer.dims.ndisplay = 2
    animation.set_active_section(section.id)
    qtbot.waitUntil(lambda: service.last_error is not None, timeout=3000)
    assert isinstance(service.last_error, RuntimeError)
    assert LayerRegistry(viewer).find_managed(section.id, "section_2d") is None


def test_disabled_composite_preserves_channel_selection():
    viewer = _viewer()
    section, red_id, green_id = _section(viewer)
    scene = Scene()
    scene.append(section)
    scene.apply(viewer)
    section.enabled = False
    scene.apply(viewer)
    scene.sync_from_viewer(viewer)
    assert all(c['enabled'] for c in section.channels.values())
    section.enabled = True
    scene.apply(viewer)
    registry = LayerRegistry(viewer)
    assert registry.find_managed(section.id, 'section_3d').visible
    assert registry.find_managed(f'{section.id}::channel::{green_id}', 'section_3d').visible


def test_appearance_mode_transitions_keep_complete_payload():
    from copy import deepcopy
    from dataclasses import replace

    viewer = _viewer()
    section, _, green_id = _section(viewer)
    animation = Animation(viewer)
    animation.add_scene_object(section)
    section.channels[green_id].update(appearance_mode='custom', appearance={
        'colormap': 'magenta', 'opacity': .4, 'gamma': 2.,
        'contrast_limits': [0., 3.],
    })
    custom = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    linked_scene = deepcopy(custom.scene)
    linked_scene[section.id]['channels'][green_id].update(
        appearance_mode='source', appearance=None)
    linked = replace(custom, scene=linked_scene)
    for left, right in ((custom, linked), (linked, custom)):
        for fraction, raw in ((0., 0.), (.5, .5), (1., .5), (1., 1.)):
            state = animation._frames._interpolate_state(
                left, right, fraction, raw_fraction=raw)
            expected = (left if raw < 1 else right).scene[section.id]['channels'][green_id]
            actual = state.scene[section.id]['channels'][green_id]
            assert actual['appearance_mode'] == expected['appearance_mode']
            assert actual['appearance'] == expected['appearance']
            state.apply(viewer)


def test_disabled_channel_cannot_reapply_previous_snapshot():
    from copy import deepcopy
    from dataclasses import replace

    viewer = _viewer()
    section, _, green_id = _section(viewer)
    animation = Animation(viewer)
    animation.add_scene_object(section)
    on = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    on.apply(viewer)
    off_scene = deepcopy(on.scene)
    off_scene[section.id]['channels'][green_id]['enabled'] = False
    off_scene[section.id]['position'] = (1, 4, 4)
    replace(on, scene=off_scene).apply(viewer)
    registry = LayerRegistry(viewer)
    red = registry.find_managed(section.id, 'section_3d')
    green = registry.find_managed(f'{section.id}::channel::{green_id}', 'section_3d')
    viewer.layers['green'].colormap = 'cyan'
    green.opacity = .3
    np.testing.assert_allclose(red.plane.position, (1, 4, 4))
    assert not green.visible
    on.apply(viewer)
    viewer.layers['green'].colormap = 'yellow'
    assert green.visible
    assert green.colormap.name == 'yellow'


def test_2d_ignores_disabled_transformed_primary():
    import pytest

    viewer = _viewer()
    section, red_id, _ = _section(viewer)
    scene = Scene()
    scene.append(section)
    scene.apply(viewer)
    viewer.layers['red'].shear = (.2, 0, 0)
    section.channels[red_id]['enabled'] = False
    viewer.dims.ndisplay = 2
    scene.apply(viewer)
    section.channels[red_id]['enabled'] = True
    with pytest.raises(ValueError, match='transformed|axis-aligned'):
        scene.apply(viewer)


def test_custom_lut_transitions_are_atomic():
    from copy import deepcopy
    from dataclasses import replace
    from napari.utils.colormaps import Colormap

    viewer = _viewer()
    section, _, green_id = _section(viewer)
    animation = Animation(viewer)
    animation.add_scene_object(section)
    lut = section._colormap_spec(Colormap(
        colors=[[0, 0, 0, 1], [1, 0, 1, 1]], name='test LUT'))
    section.channels[green_id].update(appearance_mode='custom', appearance={
        'colormap': lut, 'opacity': .2, 'gamma': 1.,
        'contrast_limits': [0., 3.],
    })
    custom = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    for destination in ('green', section._colormap_spec(Colormap(
            colors=[[0, 0, 0, 1], [0, 1, 1, 1]], name='second LUT'))):
        target_scene = deepcopy(custom.scene)
        target_scene[section.id]['channels'][green_id]['appearance'].update(
            colormap=destination, opacity=.8)
        target = replace(custom, scene=target_scene)
        for left, right in ((custom, target), (target, custom)):
            for fraction, raw in ((0., 0.), (.5, .5), (1., .5), (1., 1.)):
                state = animation._frames._interpolate_state(
                    left, right, fraction, raw_fraction=raw)
                result = state.scene[section.id]['channels'][green_id]['appearance']
                expected = (left if raw < 1 else right).scene[section.id]['channels'][green_id]['appearance']
                assert result['colormap'] == expected['colormap']
                if fraction == .5:
                    assert np.isclose(result['opacity'], .5)
                state.apply(viewer)


def test_secondary_channel_2d_rotation_switches_at_endpoint():
    from copy import deepcopy
    from dataclasses import replace

    viewer = _viewer()
    section, _, green_id = _section(viewer)
    animation = Animation(viewer)
    animation.add_scene_object(section)
    viewer.dims.ndisplay = 2
    animation.set_active_section(section.id)
    left = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    layers = deepcopy(left.layers)
    rotation = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    layers[green_id]['rotate'] = rotation
    right = replace(left, layers=layers)
    for fraction, raw in ((0., 0.), (.5, .5), (1., .5), (1., 1.)):
        state = animation._frames._interpolate_state(
            left, right, fraction, raw_fraction=raw)
        expected = left.layers[green_id]['rotate'] if raw < 1 else rotation
        np.testing.assert_allclose(state.layers[green_id]['rotate'], expected)
        state.apply(viewer)


def test_omitted_composite_retires_callbacks_and_can_be_restored():
    from dataclasses import replace

    viewer = _viewer()
    section, _, green_id = _section(viewer)
    animation = Animation(viewer)
    animation.add_scene_object(section)
    on = ViewerState.from_viewer(viewer, scene=animation.scene.to_dict())
    on.apply(viewer)
    replace(on, scene={}, view={'active_section_id': None}).apply(viewer)
    registry = LayerRegistry(viewer)
    red = registry.find_managed(section.id, 'section_3d')
    green = registry.find_managed(f'{section.id}::channel::{green_id}', 'section_3d')
    viewer.layers['green'].colormap = 'cyan'
    green.opacity = .3
    green.plane.position = (1, 4, 4)
    assert not red.visible
    assert not green.visible
    on.apply(viewer)
    viewer.layers['green'].colormap = 'yellow'
    assert red.visible and green.visible
    assert green.colormap.name == 'yellow'
    green.plane.position = (2, 4, 4)
    np.testing.assert_allclose(red.plane.position, (2, 4, 4))


def test_native_edit_after_selecting_a_keyframe_leaves_it_unchanged():
    """Selecting a keyframe must not let the live section share its state."""
    viewer = _viewer()
    section, _, green_id = _section(viewer)
    animation = Animation(viewer)
    animation.add_scene_object(section)
    for _ in range(2):
        animation.key_frames.append(KeyFrame(
            ViewerState.from_viewer(viewer, scene=animation.scene.to_dict()),
            np.zeros((30, 30, 4), np.uint8), steps=1,
        ))
    animation.set_to_keyframe(1)
    animation.set_to_keyframe(0)
    stored = animation.key_frames[0].viewer_state.scene[section.id]["channels"]
    green = LayerRegistry(viewer).find_managed(
        f"{section.id}::channel::{green_id}", "section_3d"
    )

    green.colormap = "magenta"  # a native napari edit of one channel's plane

    live = animation.scene[0].channels[green_id]
    assert live["appearance_mode"] == "custom"
    assert stored[green_id].get("appearance_mode", "source") == "source"
