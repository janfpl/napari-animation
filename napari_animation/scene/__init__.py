"""Animatable objects that live in the 3D scene alongside napari's layers."""

from .base import SceneObject
from .clip_plane import ORIENTATIONS, ClipPlane
from .layer_registry import SOURCE_ID_METADATA, LayerRegistry, SourceRecord
from .ortho_slice import PROJECTION_TO_RENDERING, VIEW_NORMALS, OrthoSlice
from .renderer import SceneRenderer
from .scene import (
    Scene,
    apply_scene_state,
    composite_clipping_planes,
    scene_contributions,
)

__all__ = [
    "ORIENTATIONS",
    "PROJECTION_TO_RENDERING",
    "VIEW_NORMALS",
    "ClipPlane",
    "OrthoSlice",
    "Scene",
    "SceneObject",
    "SceneRenderer",
    "LayerRegistry",
    "SourceRecord",
    "SOURCE_ID_METADATA",
    "apply_scene_state",
    "composite_clipping_planes",
    "scene_contributions",
]
