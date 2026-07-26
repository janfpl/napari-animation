"""An animatable clipping plane -- the Imaris "clipping plane" object.

Each :class:`ClipPlane` cuts away everything on the back side of a plane
defined by a world-space position and normal. Any number of them can be added;
:class:`~napari_animation.scene.scene.Scene` composites their contributions so
they accumulate on a layer instead of overwriting one another.

Sweeping the cutaway across an animation is then just interpolation of
``position``; tilting it is interpolation of ``normal`` (spherically, so the
plane rotates rather than passing through the origin).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Tuple

from .base import (
    SceneObject,
    register_kind,
    world_to_data_normal,
    world_to_data_position,
)

if TYPE_CHECKING:
    import napari

#: Named axis-aligned orientations, as world-space normals on the trailing
#: (Z, Y, X) spatial axes.
ORIENTATIONS = {
    "+Z": (1.0, 0.0, 0.0),
    "-Z": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+X": (0.0, 0.0, 1.0),
    "-X": (0.0, 0.0, -1.0),
}


@register_kind
@dataclass
class ClipPlane(SceneObject):
    """A single clipping plane that cuts away part of a volume.

    Parameters
    ----------
    position : tuple of float
        A point on the plane, in world coordinates, ordered ``(z, y, x)``.
    normal : tuple of float
        The plane normal in world coordinates. Data on the side the normal
        points *away* from is clipped away.
    """

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    kind = "clip_plane"

    def flipped(self) -> "ClipPlane":
        """Return a copy of this plane cutting away the opposite side."""
        return ClipPlane(
            name=self.name,
            enabled=self.enabled,
            targets=self.targets,
            id=self.id,
            position=self.position,
            normal=tuple(-value for value in self.normal),
        )

    def clipping_planes(
        self, viewer: "napari.viewer.Viewer"
    ) -> Iterator[Tuple[str, List[dict]]]:
        for layer in self.target_layers(viewer):
            # clipping a plane out of a 2D layer has no meaning
            if layer.ndim < 3:
                continue
            position = world_to_data_position(layer, self.position)
            normal = world_to_data_normal(layer, self.normal)
            yield layer.name, [
                {
                    "position": tuple(position.tolist()),
                    "normal": tuple(normal.tolist()),
                    "enabled": True,
                }
            ]
