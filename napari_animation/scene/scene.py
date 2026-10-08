"""The scene: a list of animatable objects, and the clipping-plane compositor.

The compositor exists to enforce one rule: **exactly one piece of code ever
assigns a layer's** ``experimental_clipping_planes``. Every source of clipping
geometry -- each :class:`~napari_animation.scene.clip_plane.ClipPlane`, and the
ortho slicer's clip-mode slab -- *contributes* planes, and the compositor
unions those contributions and performs a single assignment per layer.

Without this, sources silently wipe one another: two cutaway planes, or a
cutaway plus an optical section, cannot coexist if each assigns the whole list.
Since napari accepts an arbitrary number of clipping planes per layer,
compositing is what makes "cut the volume with N planes" work at all.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

from napari.utils.events import SelectableEventedList

from ..ortho_slicer import OrthoSlicer
from ..layer_registry import LayerRegistry
from .base import SceneObject

if TYPE_CHECKING:
    import napari


def _planes_differ(current, desired: List[dict]) -> bool:
    """Whether a layer's clipping planes need reassigning."""
    if len(current) != len(desired):
        return True
    for existing, wanted in zip(current, desired):
        if tuple(existing.position) != tuple(wanted["position"]):
            return True
        if tuple(existing.normal) != tuple(wanted["normal"]):
            return True
        if bool(existing.enabled) != bool(wanted.get("enabled", True)):
            return True
    return False


def _objects_from_state(scene_state: Optional[dict]) -> Iterator[SceneObject]:
    """Rebuild the scene objects described by a captured state."""
    for params in (scene_state or {}).values():
        try:
            yield SceneObject.from_dict(params)
        except (ValueError, TypeError):
            # an object kind this version doesn't know about; ignore it
            # rather than failing the whole frame
            continue


def _validate_active_2d_section(viewer, objects, active_section_id=None):
    """Reject an unsupported active section before mutating the viewer."""
    if getattr(viewer.dims, "ndisplay", 3) >= 3:
        return
    candidates = [
        obj
        for obj in objects
        if obj.kind == "ortho_slice" and obj.enabled
    ]
    if not candidates:
        return
    active = next(
        (obj for obj in candidates if obj.id == active_section_id), None
    )
    if active_section_id is None:
        active = candidates[0]
    if active is None:
        return
    sections = active._channel_sections(viewer)
    if any(
        section.source_layer(viewer) is not None
        and not section.supports_2d(viewer)
        for section in sections
    ):
        raise ValueError(
            "Oblique or transformed ortho sections require a 3D view; "
            "choose an axis-aligned section for 2D"
        )


def scene_contributions(
    viewer: "napari.viewer.Viewer", scene_state: Optional[dict]
) -> Dict[str, List[dict]]:
    """Collect the clipping planes contributed by a scene state.

    Returns a map of layer name -> list of plane dicts in that layer's data
    coordinates. Disabled objects contribute nothing, which is how the
    per-keyframe on/off switch takes effect.
    """
    contributions: Dict[str, List[dict]] = defaultdict(list)
    for scene_object in _objects_from_state(scene_state):
        if not scene_object.enabled:
            continue
        for layer_name, planes in scene_object.clipping_planes(viewer):
            contributions[layer_name].extend(planes)
    return contributions


def composite_clipping_planes(
    viewer: "napari.viewer.Viewer", contributions: Dict[str, List[dict]]
) -> None:
    """Assign every layer the union of the planes contributed to it."""
    for layer in viewer.layers:
        if not hasattr(layer, "experimental_clipping_planes"):
            continue
        desired = list(contributions.get(layer.name, []))
        if _planes_differ(layer.experimental_clipping_planes, desired):
            layer.experimental_clipping_planes = desired

    from ..native_clipping import sync_native_clipping
    sync_native_clipping(viewer)


def apply_scene_state(
    viewer: "napari.viewer.Viewer",
    scene_state: Optional[dict],
    ortho: Optional[dict] = None,
    active_section_id: Optional[str] = None,
    prepared_objects=None,
) -> None:
    """Apply a captured scene state, compositing it with the ortho slicer.

    The ortho slicer's clip-mode slab is folded into the same composite rather
    than assigned separately, so an optical section and a set of cutaway planes
    can be active at the same time.
    """
    contributions: Dict[str, List[dict]] = defaultdict(list)
    objects = (
        list(prepared_objects) if prepared_objects is not None
        else list(_objects_from_state(scene_state))
    )
    has_ortho_slice = any(obj.kind == "ortho_slice" for obj in objects)
    if active_section_id is None:
        active_section_id = next(
            (
                obj.id
                for obj in objects
                if obj.kind == "ortho_slice" and obj.enabled
            ),
            None,
        )
    _validate_active_2d_section(viewer, objects, active_section_id)
    if has_ortho_slice and getattr(viewer.dims, "ndisplay", 3) < 3:
        # Scene-owned optical sections use managed 2D projection layers.
        # Clear margins left by a previous legacy slicer
        # before the live object reapplies its own section.
        OrthoSlicer._reset_margins(viewer)
    for scene_object in objects:
        # applied even when disabled, so an object that is switched off can
        # hide the layer it owns
        if scene_object.kind == "ortho_slice":
            scene_object.apply(
                viewer, active_section_id=active_section_id
            )
        else:
            scene_object.apply(viewer)
        if not scene_object.enabled:
            continue
        for name, planes in scene_object.clipping_planes(viewer):
            contributions[name].extend(planes)

    if ortho and not (
        has_ortho_slice and getattr(viewer.dims, "ndisplay", 3) < 3
    ):
        for name, planes in OrthoSlicer.clip_contributions(
            viewer, ortho
        ).items():
            contributions[name].extend(planes)
    composite_clipping_planes(viewer, contributions)


class Scene(SelectableEventedList[SceneObject]):
    """An evented, ordered list of the animatable objects in the 3D scene."""

    def __init__(self) -> None:
        super().__init__(basetype=SceneObject)

    def to_dict(self) -> Dict[str, dict]:
        """Snapshot the scene as ``{object id: parameters}``.

        Keyed by id rather than stored as a list so that interpolation matches
        objects across keyframes by identity.
        """
        return {obj.id: obj.to_dict() for obj in self}

    def restore_snapshot(
        self, viewer: "napari.viewer.Viewer", scene_state: Optional[dict]
    ) -> None:
        """Reconcile the editable scene with a captured scene snapshot.

        Playback used to rebuild throwaway objects from every frame while the
        Qt editors continued to point at the old live objects.  This method is
        the single reconciliation path used when selecting/loading a frame.
        Matching IDs update in place so backing layers and external bindings
        survive; absent objects relinquish their owned layers.
        """
        desired = scene_state or {}
        existing = {obj.id: obj for obj in self}
        restored: List[SceneObject] = []

        for object_id, params in desired.items():
            restored_object = SceneObject.from_dict(params)
            current = existing.pop(object_id, None)
            if current is not None:
                if current.kind != restored_object.kind:
                    raise ValueError(
                        f"Scene object {object_id!r} changes kind from "
                        f"{current.kind!r} to {restored_object.kind!r}"
                    )
                old_name = current.name
                for field in fields(restored_object):
                    if field.name == "id":
                        continue
                    setattr(current, field.name, getattr(restored_object, field.name))
                if old_name != current.name:
                    # Backing layers are owned by the object.  Rename only the
                    # layer carrying this object's ownership marker; a source
                    # layer with a colliding name must remain untouched.
                    old_layer = LayerRegistry(viewer).find_managed(
                        current.id, "section_3d"
                    )
                    if old_layer is not None:
                        old_layer.name = LayerRegistry(viewer).unique_name(
                            current.name, exclude=old_layer
                        )
                restored.append(current)
            else:
                restored.append(restored_object)

        # Any remaining live object is absent from the selected snapshot. Keep
        # it in the editable registry as a disabled object so selecting an
        # older keyframe cannot destroy the Qt editor or a native napari
        # binding. An explicit ``remove_scene_object`` still removes it from
        # the registry permanently.
        for stale in existing.values():
            stale.enabled = False
            restored.append(stale)

        self.clear()
        self.extend(restored)

    def apply(
        self,
        viewer: "napari.viewer.Viewer",
        ortho: Optional[dict] = None,
        active_section_id: Optional[str] = None,
    ) -> None:
        """Push the live scene onto ``viewer``.

        Drives the live objects directly rather than going through
        :func:`apply_scene_state`, which rebuilds throwaway objects from a
        dict. The live objects need to record what they wrote so they can
        later tell an external edit from one of their own.
        """
        contributions: Dict[str, List[dict]] = defaultdict(list)
        has_ortho_slice = any(obj.kind == "ortho_slice" for obj in self)
        if active_section_id is None:
            active_section_id = next(
                (
                    obj.id
                    for obj in self
                    if obj.kind == "ortho_slice" and obj.enabled
                ),
                None,
            )
        _validate_active_2d_section(viewer, list(self), active_section_id)
        if has_ortho_slice and getattr(viewer.dims, "ndisplay", 3) < 3:
            OrthoSlicer._reset_margins(viewer)
        for scene_object in self:
            if scene_object.kind == "ortho_slice":
                scene_object.apply(
                    viewer, active_section_id=active_section_id
                )
            else:
                scene_object.apply(viewer)
            if not scene_object.enabled:
                continue
            for name, planes in scene_object.clipping_planes(viewer):
                contributions[name].extend(planes)

        if ortho and not (
            has_ortho_slice and getattr(viewer.dims, "ndisplay", 3) < 3
        ):
            for name, planes in OrthoSlicer.clip_contributions(
                viewer, ortho
            ).items():
                contributions[name].extend(planes)
        composite_clipping_planes(viewer, contributions)

    def sync_from_viewer(self, viewer: "napari.viewer.Viewer") -> None:
        """Refresh every object from the live viewer before capturing.

        Objects backed by a real napari layer treat that layer as the source
        of truth, so a plane repositioned by an external tool (a
        napari-threedee manipulator, napari's own controls) is captured as the
        user actually sees it.
        """
        for scene_object in self:
            scene_object.sync_from_viewer(viewer)

    def adopt_existing_clipping_planes(
        self, viewer: "napari.viewer.Viewer"
    ) -> List[SceneObject]:
        """Take ownership of clipping planes already set on the viewer.

        Because the compositor is the only writer, planes set by hand or by
        another plugin would be wiped the first time the scene is applied.
        Adopting them turns each into a real scene object -- named, listed and
        keyframable -- so nothing is silently lost.

        Returns the newly adopted objects.
        """
        from .base import data_to_world_normal
        from .clip_plane import ClipPlane

        adopted: List[SceneObject] = []
        for layer in viewer.layers:
            planes = getattr(layer, "experimental_clipping_planes", None)
            if not planes:
                continue
            for index, plane in enumerate(planes):
                position = layer.data_to_world(plane.position)
                normal = data_to_world_normal(layer, plane.normal)
                adopted.append(
                    ClipPlane(
                        name=f"{layer.name} clip {index + 1}",
                        targets=(layer.name,),
                        position=tuple(
                            float(value) for value in position[-3:]
                        ),
                        normal=tuple(float(value) for value in normal[-3:]),
                        enabled=bool(plane.enabled),
                    )
                )
        self.extend(adopted)
        return adopted
