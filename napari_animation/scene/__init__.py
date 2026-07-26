"""Animatable objects that live in the 3D scene alongside napari's layers."""

from .base import SceneObject
from .clip_plane import ORIENTATIONS, ClipPlane
from .ortho_slice import PROJECTION_TO_RENDERING, VIEW_NORMALS, OrthoSlice
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
    "apply_scene_state",
    "composite_clipping_planes",
    "scene_contributions",
]
