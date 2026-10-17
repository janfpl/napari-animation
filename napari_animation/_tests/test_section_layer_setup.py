"""Creating a section layer adds no full-volume thumbnail recomputes."""

from unittest.mock import patch

import numpy as np
from napari.components import ViewerModel
from napari.layers import Image

from napari_animation.scene import OrthoSlice, Scene


class HeadlessViewer(ViewerModel):
    def screenshot(self, *args, **kwargs):
        return np.zeros((60, 60, 4), dtype=np.uint8)


def _viewer():
    viewer = HeadlessViewer()
    viewer.dims.ndisplay = 3
    viewer.add_image(
        np.random.random((10, 20, 20)),
        name="img",
        colormap="magenta",
        contrast_limits=(0.1, 0.9),
        gamma=0.7,
        opacity=0.6,
    )
    return viewer


def _count_thumbnails():
    calls = []
    original = Image._update_thumbnail

    def counted(layer):
        calls.append(layer.name)
        return original(layer)

    return calls, patch.object(Image, "_update_thumbnail", counted)


def test_new_section_layer_copies_appearance_without_extra_thumbnails():
    viewer = _viewer()
    scene = Scene()
    scene.append(OrthoSlice(name="Ortho", source="img"))
    calls, counting = _count_thumbnails()
    with counting:
        scene.apply(viewer)
    layer = viewer.layers["Ortho"]
    assert layer.colormap.name == "magenta"
    np.testing.assert_allclose(layer.contrast_limits, (0.1, 0.9))
    assert layer.gamma == 0.7
    assert layer.opacity == 0.6
    assert str(layer.projection_mode) == "none"
    # No more than napari spends adding any image layer to a viewer.
    calls_plain, counting = _count_thumbnails()
    with counting:
        viewer.add_image(viewer.layers["img"].data, name="plain")
    assert calls.count("Ortho") <= calls_plain.count("plain")
