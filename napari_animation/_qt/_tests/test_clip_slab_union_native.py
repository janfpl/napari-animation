"""Clip-mode ortho slices show the union of their slabs (real OpenGL)."""

import os

import numpy as np
import pytest
from napari.settings import get_settings
from qtpy.QtWidgets import QApplication

from napari_animation import Animation
from napari_animation.scene import ClipPlane, OrthoSlice

pytestmark = pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") in {"offscreen", "minimal"},
    reason="Requires a native Qt OpenGL context.",
)


@pytest.fixture
def viewer(make_napari_viewer):
    settings = get_settings()
    previous_async = settings.experimental.async_
    settings.experimental.async_ = False
    viewer = make_napari_viewer(show=True)
    viewer.add_image(
        np.ones((64, 64, 64), np.float32),
        name="img",
        rendering="mip",
        contrast_limits=(0, 1),
    )
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (0, 0, 0)  # looking down Z
    QApplication.processEvents()
    try:
        yield viewer
    finally:
        settings.experimental.async_ = previous_async


def _drawn(viewer):
    for _ in range(3):
        QApplication.processEvents()
    frame = viewer.screenshot(canvas_only=True, flash=False)
    return frame[..., :3].max(axis=2) > 128


def _slab(name, axis, centre):
    position = [32.0, 32.0, 32.0]
    position[axis] = centre
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    return OrthoSlice(
        name=name,
        source="img",
        position=tuple(position),
        normal=tuple(normal),
        thickness=8.0,
        render_mode="clip",
    )


def test_two_clip_slabs_show_both_slabs(viewer):
    animation = Animation(viewer)
    animation.add_scene_object(_slab("Y", 1, 16.0))
    y_only = _drawn(viewer)
    animation.scene.clear()
    animation.add_scene_object(_slab("X", 2, 48.0))
    x_only = _drawn(viewer)
    animation.add_scene_object(_slab("Y", 1, 16.0))
    both = _drawn(viewer)

    assert y_only.any() and x_only.any()
    # The union, not the small square where the two slabs overlap.
    union = y_only | x_only
    assert abs(int(both.sum()) - int(union.sum())) <= 0.02 * union.sum()
    assert both.sum() > 1.5 * max(y_only.sum(), x_only.sum())


def test_cutaway_still_cuts_every_slab(viewer):
    animation = Animation(viewer)
    animation.add_scene_object(_slab("Y", 1, 16.0))
    animation.add_scene_object(_slab("X", 2, 48.0))
    both = _drawn(viewer)
    # keep only x >= 40: removes all of the Y slab outside that half
    animation.add_scene_object(
        ClipPlane(
            name="cut", targets=("img",), position=(0, 0, 40), normal=(0, 0, 1)
        )
    )
    cut = _drawn(viewer)
    assert 0 < cut.sum() < both.sum()
