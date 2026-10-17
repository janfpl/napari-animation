"""Clip groups and the in-memory projection path (no OpenGL needed)."""

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation.native_clipping import (
    _layer_clip_groups,
    _union_clip_glsl,
)
from napari_animation.scene import ClipPlane, OrthoSlice, Scene
from napari_animation.scene.processing import project_slab
from napari_animation._projection_core import project_slab_core


def _viewer():
    viewer = ViewerModel()
    viewer.add_image(np.random.random((16, 16, 16)), name="img")
    viewer.dims.ndisplay = 3
    return viewer


def test_compositor_groups_each_clip_slab_and_strips_the_key():
    viewer = _viewer()
    scene = Scene()
    xy = OrthoSlice(name="XY", source="img", render_mode="clip")
    xz = OrthoSlice(
        name="XZ", source="img", render_mode="clip", normal=(0, 1, 0)
    )
    cut = ClipPlane(name="cut", targets=("img",))
    scene.extend([xy, cut, xz])
    scene.apply(viewer)
    layer = viewer.layers["img"]
    assert len(layer.experimental_clipping_planes) == 5
    assert layer._napari_animation_clip_groups == (
        xy.id,
        xy.id,
        None,
        xz.id,
        xz.id,
    )
    assert _layer_clip_groups(layer, 5) is not None
    # a single slab keeps vispy's own intersection shader
    xz.enabled = False
    scene.apply(viewer)
    assert _layer_clip_groups(layer, 3) is None


def test_union_shader_intersects_cutaways_and_unites_slabs():
    code = _union_clip_glsl((None, "a", "a", "b", "b"))
    assert code.startswith("float clip_planes(vec3 loc, vec3 vol_shape)")
    assert code.count("max(in_any_slab, slab)") == 2
    assert "$clipping_plane_pos0" in code and "$clipping_plane_norm4" in code


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("projection", ["max", "min", "mean"])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_in_memory_projection_matches_tiled_reference(axis, projection, dtype):
    pytest.importorskip("numba")
    rng = np.random.default_rng(0)
    volume = (rng.random((20, 24, 28)) * 1000).astype(dtype)
    expected = project_slab_core(volume, axis, 3, 11, projection)
    result = project_slab(volume, axis, 3, 11, projection)
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_big_endian_volume_uses_the_tiled_path():
    volume = np.arange(4 * 5 * 6, dtype=">u2").reshape(4, 5, 6)
    np.testing.assert_array_equal(
        project_slab(volume, 0, 0, 4, "max"), volume.max(axis=0)
    )
