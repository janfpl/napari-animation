"""Bridge napari data-space clipping planes to scene-space volume visuals.

Some napari/Vispy combinations document/store planes in layer data coordinates
but pass them unchanged to a Volume configured for scene coordinates. Keep the
model unchanged and adapt only that native visual boundary.
"""

from __future__ import annotations

import weakref

import numpy as np


def _union_clip_glsl(groups) -> str:
    """Vispy ``clip_planes`` GLSL keeping samples inside any grouped slab.

    ``groups[i]`` is the clip group of plane ``i``. Ungrouped planes are
    intersected as in vispy's own builder. Planes sharing a group are
    intersected into one slab, and the slabs are then united. The result is
    signed like vispy's: non-negative means the sample is drawn.
    """
    slabs = {}
    for index, group in enumerate(groups):
        if group is not None:
            slabs.setdefault(group, []).append(index)

    def distance(index):
        return (
            f"dot(loc_transf - $clipping_plane_pos{index}, "
            f"$clipping_plane_norm{index})"
        )

    body = [
        "vec3 loc_transf = $clip_transform(vec4(loc * vol_shape, 1)).xyz;",
        "float distance_from_clip = 3.4e38;",
    ]
    body += [
        f"distance_from_clip = min(distance_from_clip, {distance(index)});"
        for index, group in enumerate(groups)
        if group is None
    ]
    if slabs:
        body.append("float in_any_slab = -3.4e38;")
        for indices in slabs.values():
            body.append("{ float slab = 3.4e38;")
            body += [
                f"slab = min(slab, {distance(index)});" for index in indices
            ]
            body.append("in_any_slab = max(in_any_slab, slab); }")
        body.append(
            "distance_from_clip = min(distance_from_clip, in_any_slab);"
        )
    body.append("return distance_from_clip;")
    lines = "\n    ".join(body)
    return f"float clip_planes(vec3 loc, vec3 vol_shape) {{\n    {lines}\n}}"


def _layer_clip_groups(layer, n_planes):
    """The layer's clip groups when they need the union shader."""
    groups = getattr(layer, "_napari_animation_clip_groups", None)
    if not groups or len(groups) != n_planes:
        return None
    slabs = {group for group in groups if group is not None}
    # A single slab intersects exactly like vispy's own builder.
    return tuple(groups) if len(slabs) > 1 else None


def _set(node, name, value):
    object.__setattr__(node, name, value)


def _sync_clip_groups(node, layer):
    """Let clip-mode slabs on ``node`` unite instead of intersecting."""
    builder = getattr(type(node), "_build_clipping_planes_glsl", None)
    if builder is None or not hasattr(node, "clipping_planes"):
        return
    if not getattr(node, "_napari_animation_union_clip", False):
        layer_ref = weakref.ref(layer)

        def build(n_planes):
            source = layer_ref()
            groups = (
                _layer_clip_groups(source, n_planes)
                if source is not None
                else None
            )
            _set(node, "_napari_animation_built_groups", groups)
            if groups is None:
                return builder(n_planes)
            return _union_clip_glsl(groups)

        # Vispy visuals are frozen against new attributes.
        _set(node, "_build_clipping_planes_glsl", build)
        _set(node, "_napari_animation_union_clip", True)
        _set(node, "_napari_animation_built_groups", None)
    planes = node.clipping_planes
    wanted = _layer_clip_groups(layer, len(planes))
    if wanted != getattr(node, "_napari_animation_built_groups", None):
        # Rebuild the shader for the same planes with the new grouping.
        node.clipping_planes = planes


def _sync_layer(viewer, layer):
    window = getattr(viewer, "window", None)
    canvas = getattr(getattr(window, "_qt_viewer", None), "canvas", None)
    visual = getattr(canvas, "layer_to_visual", {}).get(layer)
    node = getattr(visual, "node", None)
    if node is not None:
        _sync_clip_groups(node, layer)
    if getattr(node, "clipping_planes_coord_system", None) != "scene":
        return
    planes = layer.experimental_clipping_planes.as_array()
    desired = np.empty_like(planes, dtype=float)
    if len(planes):
        # Canvas unit updates are deferred until draw. Read the target world
        # units rather than the visual's potentially stale cached factors.
        units_scale = np.ones(layer.ndim)
        if hasattr(visual, "world_units"):
            units = getattr(
                getattr(viewer.layers, "extent", None), "units", None
            )
            if units is not None:
                import pint

                registry = pint.get_application_registry()
                units_scale = np.asarray(
                    [
                        registry.get_base_units(source)[0]
                        / registry.get_base_units(target)[0]
                        for target, source in zip(
                            units[-layer.ndim :], layer.units
                        )
                    ]
                )
        # The model's planes are on the trailing spatial axes. Include the
        # current leading slice when a source has extra time/channel axes.
        ndim = layer.ndim
        world = (
            np.asarray(viewer.dims.point[-ndim:], dtype=float) / units_scale
        )
        anchor = np.asarray(layer.world_to_data(world), dtype=float)
        matrix = layer._transforms[1:].simplified.linear_matrix[-3:, -3:]
        matrix = units_scale[-3:, None] * matrix
        for index, (position, normal) in enumerate(planes):
            point = anchor.copy()
            point[-3:] = position
            desired[index, 0] = (
                np.asarray(layer.data_to_world(point))[-3:] * units_scale[-3:]
            )
            converted = np.linalg.solve(matrix.T, normal)
            length = np.linalg.norm(converted)
            desired[index, 1] = converted / length if length else converted
    # Vispy uses XYZ while napari stores ZYX.
    desired = desired[..., ::-1]
    if not np.array_equal(node.clipping_planes, desired):
        node.clipping_planes = desired


def sync_native_clipping(viewer):
    """Install layer-local listeners, after napari's native visual listeners."""
    window = getattr(viewer, "window", None)
    canvas = getattr(getattr(window, "_qt_viewer", None), "canvas", None)
    if canvas is None:
        return
    bindings = getattr(viewer, "_napari_animation_clip_bindings", None)
    if bindings is None:
        bindings = weakref.WeakKeyDictionary()
        viewer._napari_animation_clip_bindings = bindings
    active = set(getattr(viewer, "layers", ()))
    for layer in list(bindings):
        if layer not in active or getattr(layer, "ndim", 0) < 3:
            callback, emitters = bindings.pop(layer)
            for emitter in emitters:
                emitter.disconnect(callback)
    for layer in getattr(viewer, "layers", ()):
        if getattr(layer, "ndim", 0) < 3 or not hasattr(
            layer, "experimental_clipping_planes"
        ):
            continue
        if layer not in bindings:
            viewer_ref, layer_ref = weakref.ref(viewer), weakref.ref(layer)

            def changed(event=None, vr=viewer_ref, lr=layer_ref):
                owner, source = vr(), lr()
                if owner is not None and source is not None:
                    _sync_layer(owner, source)

            emitters = [layer.experimental_clipping_planes.events]
            emitters.extend(
                getattr(layer.events, name)
                for name in (
                    "scale",
                    "translate",
                    "rotate",
                    "shear",
                    "affine",
                    "set_data",
                )
            )
            for events in (
                layer.events,
                getattr(viewer.layers, "events", None),
            ):
                units_event = getattr(events, "units", None)
                if units_event is not None:
                    emitters.append(units_event)
            for emitter in emitters:
                emitter.connect(changed, position="last")
            bindings[layer] = (changed, emitters)
        _sync_layer(viewer, layer)


def release_native_clipping(viewer):
    bindings = getattr(viewer, "_napari_animation_clip_bindings", None)
    if bindings is not None:
        for callback, emitters in list(bindings.values()):
            for emitter in emitters:
                emitter.disconnect(callback)
        delattr(viewer, "_napari_animation_clip_bindings")
