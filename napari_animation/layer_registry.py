"""Stable identities and bindings for live napari layers.

Napari layer names are user-editable labels.  They are useful for display but
are not safe keys for an animation project: a rename, a reader returning two
layers, or a managed backing layer with a colliding name can otherwise bind a
snapshot to the wrong data.  ``LayerRegistry`` keeps a small identity marker
in layer metadata. Bound IDs are authoritative; only unbound legacy state
uses display-name lookup.

The registry deliberately stores no image data and does not own the viewer's
layer list.  It is a binding adapter used by the renderer and persistence
code, so a layer can still be added, removed, or renamed by napari itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4


SOURCE_ID_METADATA = "napari_animation.source_id"
OBJECT_ID_METADATA = "napari_animation.object_id"
MANAGED_ROLE_METADATA = "napari_animation.managed_role"
TRANSIENT_ROLE = "section_2d"


@dataclass(frozen=True)
class SourceRecord:
    """Serializable identity information for one live layer."""

    source_id: str
    name: str
    layer_type: str
    path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "type": self.layer_type,
            "path": self.path,
        }


class LayerRegistry:
    """Resolve stable source IDs to the current napari layer objects."""

    metadata_key = SOURCE_ID_METADATA

    def __init__(self, viewer=None):
        self.viewer = viewer

    @classmethod
    def ensure_id(cls, layer) -> str:
        """Return and persist a source ID for ``layer``."""
        metadata = getattr(layer, "metadata", None)
        if metadata is None:
            # Napari layers expose a mutable metadata mapping.  Raising here
            # gives a caller a clear failure instead of silently using names.
            raise TypeError(f"Layer {layer!r} has no metadata mapping")
        source_id = metadata.get(cls.metadata_key)
        if not source_id:
            source_id = uuid4().hex
            metadata[cls.metadata_key] = source_id
        return str(source_id)

    def source_id(self, layer) -> str:
        self.ensure_unique_ids()
        return self.ensure_id(layer)

    def ensure_unique_ids(self) -> None:
        """Repair duplicate IDs without ever merging two live layers."""
        if self.viewer is None:
            return
        seen = set()
        for layer in self.viewer.layers:
            source_id = self.ensure_id(layer)
            if source_id in seen:
                source_id = uuid4().hex
                layer.metadata[self.metadata_key] = source_id
            seen.add(source_id)

    @staticmethod
    def managed_id(object_id: str, role: str) -> str:
        """Return the deterministic ID for a renderer-owned layer."""
        return f"napari-animation:{object_id}:{role}"

    def find_managed(self, object_id: str, role: Optional[str] = None):
        if self.viewer is None:
            return None
        for layer in self.viewer.layers:
            metadata = getattr(layer, "metadata", {})
            if metadata.get(OBJECT_ID_METADATA) != object_id:
                continue
            if role is None or metadata.get(MANAGED_ROLE_METADATA) == role:
                return layer
        return None

    @classmethod
    def managed_metadata(cls, object_id: str, role: str) -> dict:
        """Metadata identifying a renderer-owned layer.

        Pass it when creating the layer, so listeners of napari's
        ``layers.events.inserted`` already see the layer as managed.
        """
        return {
            OBJECT_ID_METADATA: object_id,
            MANAGED_ROLE_METADATA: role,
            cls.metadata_key: cls.managed_id(object_id, role),
        }

    def mark_managed(self, layer, object_id: str, role: str) -> str:
        layer.metadata.update(self.managed_metadata(object_id, role))
        return self.managed_id(object_id, role)

    def unique_name(self, preferred: str, exclude=None) -> str:
        names = {
            layer.name for layer in self.viewer.layers if layer is not exclude
        } if self.viewer is not None else set()
        if preferred not in names:
            return preferred
        index = 2
        while f"{preferred} {index}" in names:
            index += 1
        return f"{preferred} {index}"

    def find(
        self, source_id: Optional[str] = None, name: Optional[str] = None
    ):
        """Find by authoritative ID, or by name for an unbound legacy state."""
        if self.viewer is None:
            return None
        self.ensure_unique_ids()
        if source_id:
            source_id = str(source_id)
            for layer in self.viewer.layers:
                if getattr(layer, "metadata", {}).get(self.metadata_key) == source_id:
                    return layer
            return None
        if name is not None and name in self.viewer.layers:
            return self.viewer.layers[name]
        return None

    def records(self) -> dict[str, SourceRecord]:
        """Return current source records keyed by stable ID."""
        if self.viewer is None:
            return {}
        self.ensure_unique_ids()
        records = {}
        for layer in self.viewer.layers:
            role = getattr(layer, "metadata", {}).get(MANAGED_ROLE_METADATA)
            if role == TRANSIENT_ROLE or layer.metadata.get(
                "napari_animation.composite"
            ):
                continue
            source_id = self.ensure_id(layer)
            path = getattr(getattr(layer, "source", None), "path", None)
            records[source_id] = SourceRecord(
                source_id=source_id,
                name=layer.name,
                layer_type=getattr(layer, "_type_string", type(layer).__name__),
                path=str(Path(path)) if path else None,
            )
        return records

    def attach_id(self, layer, source_id: Optional[str]) -> str:
        """Attach a loaded project ID to a layer, creating one if absent."""
        if source_id:
            if self.viewer is not None:
                for other in self.viewer.layers:
                    if other is layer:
                        continue
                    other_id = getattr(other, "metadata", {}).get(
                        self.metadata_key
                    )
                    if other_id == str(source_id):
                        raise ValueError(f"source ID {source_id!r} is already bound")
            layer.metadata[self.metadata_key] = str(source_id)
        return self.ensure_id(layer)
