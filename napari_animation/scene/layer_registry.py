"""Compatibility export for the scene rendering layer registry.

The registry is shared by viewer snapshots, persistence, and scene rendering,
so its implementation lives at :mod:`napari_animation.layer_registry`.  Keep
this import path available for callers that naturally discover it next to
``SceneRenderer``.
"""

from ..layer_registry import LayerRegistry, SourceRecord, SOURCE_ID_METADATA

__all__ = ["LayerRegistry", "SourceRecord", "SOURCE_ID_METADATA"]
