"""Single rendering entry point for authored and evaluated scene state."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np
from napari.utils.transforms import Affine, CompositeAffine

from ..layer_registry import OBJECT_ID_METADATA, LayerRegistry
from ..perf import timed_render
from .base import SceneObject
from .scene import Scene, apply_scene_state

_UNSET = object()


class SceneRenderer:
    """Apply scene state through one guarded compositor path.

    The existing ``Scene`` class remains the editable object registry.  This
    facade owns the boundary where that registry (or a detached frame
    snapshot) is translated into napari layers and clipping planes.  Keeping
    the boundary explicit prevents Qt controls, playback, and persistence
    from growing independent rendering implementations.
    """

    def __init__(self, viewer, scene: Optional[Scene] = None, registry=None):
        self.viewer = viewer
        self.scene = scene
        self.registry = registry or LayerRegistry(viewer)
        self.active_section_id: Optional[str] = None
        self._source_visibility = {}
        self.applying = False
        viewer._napari_animation_renderer = self

    @classmethod
    def for_viewer(cls, viewer, scene: Optional[Scene] = None, registry=None):
        renderer = getattr(viewer, "_napari_animation_renderer", None)
        if renderer is None:
            renderer = cls(viewer, scene=scene, registry=registry)
        elif scene is not None:
            renderer.scene = scene
        return renderer

    def authored_visible(self, layer, current):
        source_id = self.registry.ensure_id(layer)
        return self._source_visibility.get(source_id, current)

    @contextmanager
    def applying_state(self):
        """Suppress native edit callbacks across nested rendering operations."""
        previous = self.applying
        self.applying = True
        try:
            yield
        finally:
            self.applying = previous

    def _restore_source_visibility(self, keep_ids=()):
        saved = self._source_visibility
        keep_ids = set(keep_ids)
        self._source_visibility = {
            source_id: visible
            for source_id, visible in saved.items()
            if source_id in keep_ids
        }
        for source_id, visible in saved.items():
            if source_id in keep_ids:
                continue
            layer = self.registry.find(source_id)
            if layer is not None:
                if layer.visible != visible:
                    layer.visible = visible

    def restore_view_overrides(self, keep_ids=()):
        """Remove temporary 2D suppression before authored state is applied."""
        self._restore_source_visibility(keep_ids=keep_ids)

    def _suppressed_source_ids(self, active_obj, ndisplay=None):
        if (ndisplay or self.viewer.dims.ndisplay) >= 3 or active_obj is None:
            return set()
        return {
            self.registry.source_id(source)
            for source in active_obj.source_layers(
                self.viewer, include_disabled=True
            )
        }

    def _suppressed_source_id(self, active_obj, ndisplay=None):
        """Compatibility helper for callers needing the first source."""
        return next(
            iter(self._suppressed_source_ids(active_obj, ndisplay)), None
        )

    def _objects(self, scene_state=_UNSET):
        if scene_state is _UNSET:
            return list(self.scene or ())
        objects = []
        for object_id, params in (scene_state or {}).items():
            obj = SceneObject.from_dict(params)
            if str(obj.id) != str(object_id):
                raise ValueError(
                    f"Scene key {object_id!r} does not match object id {obj.id!r}"
                )
            objects.append(obj)
        return objects

    def preflight(
        self,
        scene_state=_UNSET,
        ndisplay=None,
        active_section_id=_UNSET,
        layer_states=None,
    ):
        objects = self._objects(scene_state)
        ids = [obj.id for obj in objects]
        if len(ids) != len(set(ids)):
            raise ValueError("Scene contains duplicate object IDs")
        active = (
            self.active_section_id
            if active_section_id is _UNSET
            else active_section_id
        )
        sections = [obj for obj in objects if obj.kind == "ortho_slice"]
        if active is not None and not any(
            obj.id == active for obj in sections
        ):
            raise ValueError(f"Active section {active!r} is missing")
        active_obj = next((obj for obj in sections if obj.id == active), None)
        if active is None:
            active_obj = next((obj for obj in sections if obj.enabled), None)
        if active_obj is not None and not active_obj.enabled:
            active_obj = None
        for obj in objects:
            if obj.kind == "ortho_slice":
                target_ndisplay = ndisplay or self.viewer.dims.ndisplay
                targets = (
                    obj._channel_sections(self.viewer)
                    if obj.channels is not None
                    else (obj,)
                )
                if obj.channels is not None:
                    obj.validate(self.viewer)
                for target in targets:
                    linear_matrix = None
                    source = target.source_layer(self.viewer)
                    if source is not None and layer_states:
                        state = layer_states.get(
                            self.registry.source_id(source)
                        )
                        if state is not None:
                            transform = CompositeAffine(
                                ndim=source.ndim,
                                scale=state.get("scale", source.scale),
                                translate=state.get(
                                    "translate", source.translate
                                ),
                                rotate=state.get("rotate", source.rotate),
                                shear=state.get("shear", source.shear),
                            )
                            extra = Affine(
                                affine_matrix=np.asarray(
                                    state.get(
                                        "affine", source.affine.affine_matrix
                                    ),
                                    dtype=float,
                                ),
                                ndim=source.ndim,
                            )
                            linear_matrix = (
                                extra.linear_matrix @ transform.linear_matrix
                            )
                    target.validate(
                        self.viewer,
                        for_2d=target_ndisplay < 3 and obj is active_obj,
                        linear_matrix=linear_matrix,
                    )
        return objects, active_obj

    @timed_render("scene.prepare")
    def prepare_snapshot(
        self,
        scene_state,
        active_section_id=_UNSET,
        ndisplay=None,
        layer_states=None,
    ):
        """Create deterministic managed layers before appearance is restored."""
        objects, active = self.preflight(
            scene_state,
            active_section_id=active_section_id,
            ndisplay=ndisplay,
            layer_states=layer_states,
        )
        for obj in objects:
            if obj.kind == "ortho_slice":
                obj.ensure_layer(
                    self.viewer,
                    for_2d=(ndisplay or self.viewer.dims.ndisplay) < 3,
                )
        return objects, active

    def _reconcile(self, objects, active_obj):
        desired = {obj.id for obj in objects}
        desired.update(
            child.id
            for obj in objects
            if obj.kind == "ortho_slice" and obj.channels is not None
            for child in obj._channel_sections(self.viewer)
        )
        service = getattr(
            self.viewer, "_napari_animation_projection_service", None
        )
        if service is not None:
            service.retain_sections(
                {
                    child.id
                    for child in active_obj._channel_sections(self.viewer)
                }
                if active_obj is not None and self.viewer.dims.ndisplay < 3
                else set()
            )
        for layer in self.viewer.layers:
            bindings = getattr(
                layer, "_napari_animation_composite_appearance", {}
            )
            for section_id in tuple(bindings):
                if section_id not in desired:
                    _, callback, emitters = bindings.pop(section_id)
                    for emitter in emitters:
                        emitter.disconnect(callback)
            metadata = getattr(layer, "metadata", {})
            object_id = metadata.get(OBJECT_ID_METADATA)
            if object_id and object_id not in desired:
                binding = getattr(
                    layer, "_napari_animation_plane_listener", None
                )
                if binding is not None:
                    _, callback, emitters = binding
                    for emitter in emitters:
                        emitter.disconnect(callback)
                    delattr(layer, "_napari_animation_plane_listener")
                layer.visible = False
        if self.viewer.dims.ndisplay < 3 and active_obj is not None:
            for source in active_obj.source_layers(
                self.viewer, include_disabled=True
            ):
                source_id = self.registry.source_id(source)
                self._source_visibility.setdefault(
                    source_id, bool(source.visible)
                )
                if source.visible:
                    source.visible = False

    def apply_working(self, ortho: Optional[dict] = None) -> None:
        """Render the live editable scene."""
        self.registry.records()
        if self.scene is None:
            return
        objects, active = self.preflight(ndisplay=self.viewer.dims.ndisplay)
        with self.applying_state():
            keep = self._suppressed_source_ids(active)
            self._restore_source_visibility(keep_ids=keep)
            self.scene.apply(
                self.viewer,
                ortho=ortho,
                active_section_id=self.active_section_id,
            )
            self._reconcile(objects, active)

    @timed_render("scene.apply")
    def apply_snapshot(
        self,
        scene_state: Optional[dict],
        ortho: Optional[dict] = None,
        active_section_id=_UNSET,
        prepared=None,
    ) -> None:
        """Render a detached snapshot; reuse same-frame preflight if supplied."""
        self.registry.records()
        active_id = (
            self.active_section_id
            if active_section_id is _UNSET
            else active_section_id
        )
        objects, active = (
            prepared
            if prepared is not None
            else self.preflight(
                scene_state,
                ndisplay=self.viewer.dims.ndisplay,
                active_section_id=active_id,
            )
        )
        with self.applying_state():
            # Cached results publish immediately during scene application.
            self.active_section_id = active_id
            keep = self._suppressed_source_ids(active)
            self._restore_source_visibility(keep_ids=keep)
            apply_scene_state(
                self.viewer,
                scene_state,
                ortho=ortho,
                active_section_id=active_id,
                prepared_objects=objects,
            )
            self._reconcile(objects, active)
