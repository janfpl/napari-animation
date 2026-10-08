"""Regression coverage for the renderer/identity architecture."""

from dataclasses import replace

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation, ViewerState
from napari_animation.frame_sequence import FrameSequence
from napari_animation.key_frame import KeyFrameList
from napari_animation.layer_registry import LayerRegistry, SOURCE_ID_METADATA
from napari_animation.scene import OrthoSlice


def _viewer(data=None, **kwargs):
    viewer = ViewerModel()
    viewer.add_image(
        np.arange(6 * 7 * 9, dtype=float).reshape(6, 7, 9)
        if data is None
        else data,
        name="volume",
        **kwargs,
    )
    return viewer


def _section_view(viewer, **kwargs):
    animation = Animation(viewer)
    position = kwargs.pop("position", (2, 3, 4))
    section = animation.add_scene_object(
        OrthoSlice(
            name="section",
            source="volume",
            position=position,
            thickness=2,
            **kwargs,
        )
    )
    animation.set_active_section(section.id)
    animation.snap_2d(capture=False)
    output = LayerRegistry(viewer).find_managed(section.id, "section_2d")
    return animation, section, output


def test_managed_name_collision_never_claims_user_layer():
    viewer = _viewer()
    user = viewer.add_image(np.full((2, 2, 2), 999), name="section")
    animation = Animation(viewer)
    section = animation.add_scene_object(
        OrthoSlice(name="section", source="volume")
    )

    assert np.max(user.data) == 999
    assert "napari_animation.object_id" not in user.metadata
    assert LayerRegistry(viewer).find_managed(section.id, "section_3d") is not user


def test_stable_source_wins_after_rename_and_name_reuse():
    viewer = _viewer()
    animation = Animation(viewer)
    section = animation.add_scene_object(OrthoSlice(name="section", source="volume"))
    original = viewer.layers["volume"]
    original.name = "renamed"
    replacement = viewer.add_image(np.zeros((6, 7, 9)), name="volume")

    animation.apply_working_state()

    assert section.source_layer(viewer) is original
    assert section.source_layer(viewer) is not replacement


def test_duplicate_live_ids_are_repaired_before_capture():
    viewer = _viewer()
    duplicate = viewer.add_image(np.zeros((6, 7, 9)), name="copy")
    source_id = LayerRegistry.ensure_id(viewer.layers["volume"])
    duplicate.metadata[SOURCE_ID_METADATA] = source_id

    snapshot = ViewerState.from_viewer(viewer)

    assert len(snapshot.layers) == 2
    assert len({layer.metadata[SOURCE_ID_METADATA] for layer in viewer.layers}) == 2


@pytest.mark.parametrize(
    "normal, hidden",
    [((1, 0, 0), 0), ((0, 1, 0), 1), ((0, 0, 1), 2)],
)
def test_active_section_selects_matching_world_axis(normal, hidden):
    viewer = _viewer()
    animation, section, output = _section_view(viewer, normal=normal)

    assert viewer.dims.not_displayed[-1] == hidden
    assert output.visible
    np.testing.assert_allclose(
        np.asarray(output._data_view), section.projection_oracle(viewer)
    )


def test_dims_scroll_updates_canonical_section_before_capture():
    viewer = _viewer()
    animation, section, _ = _section_view(viewer)
    viewer.dims.set_point(0, 5)
    _, scene, _ = animation._capture_state()

    captured = scene[section.id]
    assert captured["position"][0] == pytest.approx(5)


def test_oblique_selection_is_transactional():
    viewer = _viewer()
    viewer.dims.ndisplay = 3
    animation = Animation(viewer)
    good = animation.add_scene_object(OrthoSlice(name="good", source="volume"))
    section = animation.add_scene_object(
        OrthoSlice(name="oblique", source="volume", normal=(1, 1, 0))
    )
    animation.set_active_section(good.id)
    animation.snap_2d(capture=False)
    before = (
        animation.active_section_id,
        viewer.dims.ndisplay,
        tuple(viewer.dims.order),
    )

    with pytest.raises(ValueError, match="require a 3D view"):
        animation.set_active_section(section.id)

    after = (
        animation.active_section_id,
        viewer.dims.ndisplay,
        tuple(viewer.dims.order),
    )
    assert after == before


def test_multiscale_projection_uses_requested_level():
    levels = [
        np.arange(8 * 10 * 12).reshape(8, 10, 12),
        np.arange(4 * 5 * 6).reshape(4, 5, 6) + 1000,
    ]
    viewer = ViewerModel()
    viewer.add_image(levels, name="volume", multiscale=True)
    _, section, output = _section_view(viewer, source_level=1)

    np.testing.assert_array_equal(
        np.squeeze(output.data), section.projection_oracle(viewer)
    )
    assert np.min(output.data) >= 1000


@pytest.mark.parametrize(
    "transform",
    [
        {"scale": (-1, 1, 1), "translate": (5, 0, 0)},
        {"rotate": (90, 0, 0)},
    ],
)
def test_signed_and_permuted_axes_match_visible_pixel_oracle(transform):
    viewer = _viewer(**transform)
    _, section, output = _section_view(
        viewer, position=viewer.layers["volume"].data_to_world((2, 3, 4))
    )

    np.testing.assert_array_equal(
        np.asarray(output._data_view), section.projection_oracle(viewer)
    )


def test_outside_section_hides_projection_and_oracle_is_deterministic():
    viewer = _viewer()
    animation, section, output = _section_view(viewer)
    section.position = (1000, 3, 4)
    animation.apply_working_state()

    assert not output.visible
    first = section.projection_oracle(viewer)
    second = section.projection_oracle(viewer)
    np.testing.assert_array_equal(first, second)
    assert not np.any(first)


def test_four_dimensional_source_uses_three_dimensional_managed_layers():
    viewer = _viewer(np.arange(2 * 6 * 7 * 9).reshape(2, 6, 7, 9))
    viewer.dims.set_point(0, 1)
    animation, section, output = _section_view(viewer)

    backing = LayerRegistry(viewer).find_managed(section.id, "section_3d")
    assert backing.ndim == output.ndim == 3
    np.testing.assert_array_equal(
        np.asarray(output._data_view), section.projection_oracle(viewer)
    )


def test_empty_snapshot_hides_all_managed_layers():
    viewer = _viewer()
    animation = Animation(viewer)
    section = animation.add_scene_object(OrthoSlice(name="section", source="volume"))
    backing = LayerRegistry(viewer).find_managed(section.id, "section_3d")
    state = ViewerState.from_viewer(
        viewer, scene={}, view={"active_section_id": None}
    )

    state.apply(viewer)

    assert not backing.visible


def test_2d_visibility_override_preserves_each_snapshot_authored_value():
    viewer = _viewer()
    animation, section, _ = _section_view(viewer)
    source = viewer.layers["volume"]
    source_id = LayerRegistry(viewer).source_id(source)
    state = ViewerState.from_viewer(
        viewer,
        scene=animation.scene.to_dict(),
        view={"active_section_id": section.id},
    )
    layers = {key: dict(value) for key, value in state.layers.items()}
    layers[source_id]["visible"] = False

    replace(state, layers=layers).apply(viewer)
    animation.snap_3d(capture=False)

    assert source.visible is False


def test_discrete_fields_use_raw_progress_and_hold_until_destination():
    base = ViewerState(
        camera={"zoom": 1.0},
        dims={"ndisplay": 2, "order": (0, 1, 2)},
        layers={"a": {"source_id": "a", "name": "a", "visible": True}},
        scene={
            "s": {
                "kind": "ortho_slice",
                "id": "s",
                "name": "s",
                "enabled": True,
                "targets": (),
                "source": "a",
                "source_id": "a",
                "position": (0, 0, 0),
                "normal": (1, 0, 0),
                "thickness": 1.0,
                "projection": "max",
                "render_mode": "plane",
                "source_level": 0,
                "sampling_convention": "voxel-centers-half-open-v1",
            }
        },
        view={"active_section_id": "s"},
    )
    target = replace(
        base,
        dims={"ndisplay": 2, "order": (1, 0, 2)},
        scene={
            "s": {
                **base.scene["s"],
                "source": "b",
                "source_id": "b",
                "source_level": 1,
                "projection": "mean",
            }
        },
    )
    sequence = FrameSequence(KeyFrameList())

    middle = sequence._interpolate_state(base, target, 0.99, raw_fraction=0.25)

    assert tuple(middle.dims["order"]) == (0, 1, 2)
    assert middle.scene["s"]["source"] == "a"
    assert middle.scene["s"]["source_level"] == 0
