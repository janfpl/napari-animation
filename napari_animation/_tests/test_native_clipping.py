from types import SimpleNamespace

import numpy as np
from napari.components import ViewerModel

from napari_animation.native_clipping import (
    release_native_clipping, sync_native_clipping,
)


def test_native_clip_adapter_preserves_model_and_transforms_normals():
    model = ViewerModel(ndisplay=3)
    layer = model.add_image(np.zeros((8, 10, 12), np.uint8),
                            scale=(2, 3, 4), translate=(10, 20, 30))
    layer.experimental_clipping_planes = [dict(position=(1, 2, 3),
                                               normal=(1, 1, 0))]
    original = layer.experimental_clipping_planes.as_array().copy()
    node = SimpleNamespace(clipping_planes_coord_system="scene",
                           clipping_planes=np.empty((0, 2, 3)))
    visual = SimpleNamespace(node=node)
    canvas = SimpleNamespace(layer_to_visual={layer: visual})
    viewer = SimpleNamespace(
        layers=model.layers, dims=model.dims,
        window=SimpleNamespace(_qt_viewer=SimpleNamespace(canvas=canvas)),
    )

    # SimpleNamespace cannot be weak-referenced; use a plain adapter class.
    class Viewer:
        pass
    owner = Viewer()
    owner.__dict__.update(viewer.__dict__)
    sync_native_clipping(owner)
    np.testing.assert_array_equal(original, layer.experimental_clipping_planes.as_array())
    np.testing.assert_allclose(node.clipping_planes[0, 0], (42, 26, 12))
    expected_normal = np.array([1 / 2, 1 / 3, 0.])
    expected_normal /= np.linalg.norm(expected_normal)
    np.testing.assert_allclose(node.clipping_planes[0, 1], expected_normal[::-1])
    count = len(layer.events.scale.callbacks)
    sync_native_clipping(owner)
    assert len(layer.events.scale.callbacks) == count
    layer.translate = (11, 22, 33)
    np.testing.assert_allclose(node.clipping_planes[0, 0], (45, 28, 13))
    # A recreated native node is picked up on the data event.
    replacement = SimpleNamespace(clipping_planes_coord_system="scene",
                                  clipping_planes=np.empty((0, 2, 3)))
    visual.node = replacement
    layer.events.set_data()
    np.testing.assert_array_equal(replacement.clipping_planes, node.clipping_planes)
    release_native_clipping(owner)
    assert len(layer.events.scale.callbacks) == count - 1
    assert not hasattr(owner, "_napari_animation_clip_bindings")


def test_native_clip_adapter_leaves_data_space_visuals_alone():
    model = ViewerModel(ndisplay=3)
    layer = model.add_image(np.zeros((8, 8, 8), np.uint8))
    node = SimpleNamespace(clipping_planes_coord_system="visual",
                           clipping_planes="untouched")

    class Viewer:
        pass
    viewer = Viewer()
    viewer.layers, viewer.dims = model.layers, model.dims
    viewer.window = SimpleNamespace(_qt_viewer=SimpleNamespace(
        canvas=SimpleNamespace(layer_to_visual={layer: SimpleNamespace(node=node)})))
    sync_native_clipping(viewer)
    assert node.clipping_planes == "untouched"
    release_native_clipping(viewer)


def test_native_clip_units_transform_positions_normals_and_follow_edits():
    model = ViewerModel(ndisplay=3)
    layer = model.add_image(np.zeros((8, 8, 8), np.uint8),
                            scale=(1000, 2000, 3000),
                            translate=(1000, 2000, 3000), units=('nm',) * 3)
    model.layers.units = ('um', 'nm', 'um')
    layer.experimental_clipping_planes = [dict(position=(1, 2, 3), normal=(1, 1, 1))]
    original = layer.experimental_clipping_planes.as_array().copy()
    node = SimpleNamespace(clipping_planes_coord_system='scene',
                           clipping_planes=np.empty((0, 2, 3)))
    # Deliberately stale: the canvas applies its unit changes on the next draw.
    visual = SimpleNamespace(node=node, world_units=layer.units)

    class Viewer:
        pass

    viewer = Viewer()
    viewer.layers, viewer.dims = model.layers, model.dims
    viewer.window = SimpleNamespace(_qt_viewer=SimpleNamespace(
        canvas=SimpleNamespace(layer_to_visual={layer: visual})))
    callbacks = len(model.layers.events.units.callbacks)
    try:
        sync_native_clipping(viewer)
        np.testing.assert_allclose(node.clipping_planes[0, 0], (12, 6000, 2))
        normal = np.array([1., 1 / 2000, 1 / 3])
        np.testing.assert_allclose(node.clipping_planes[0, 1], (normal / np.linalg.norm(normal))[::-1])
        model.layers.units = ('um',) * 3
        np.testing.assert_allclose(node.clipping_planes[0, 0], (12, 6, 2))
        layer.units = ('um',) * 3
        np.testing.assert_allclose(node.clipping_planes[0, 0], (12000, 6000, 2000))
        np.testing.assert_array_equal(layer.experimental_clipping_planes.as_array(), original)
    finally:
        release_native_clipping(viewer)
    assert len(model.layers.events.units.callbacks) == callbacks
