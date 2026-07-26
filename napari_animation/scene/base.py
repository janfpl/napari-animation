"""Base class for animatable scene objects.

A :class:`SceneObject` is something that exists in the 3D scene alongside the
napari layers -- a clipping plane, an ortho slice -- and that can be switched
on or off, and reshaped, independently at every keyframe.

Two properties make the keyframing work:

* Each object carries a stable ``id``. Keyframe state is stored as
  ``{id: params}`` rather than as a list, so interpolation matches objects by
  identity instead of by position. A list would silently truncate to the
  shorter side when an object is added or removed between keyframes.
* Every parameter is a plain float, bool, string or tuple thereof, so the
  existing interpolation machinery walks and blends scene state with no
  special cases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, ClassVar, Dict, Iterator, List, Tuple, Type
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    import napari

#: Maps a serialised ``kind`` back to its :class:`SceneObject` subclass.
_KINDS: Dict[str, Type["SceneObject"]] = {}


def register_kind(cls: Type["SceneObject"]) -> Type["SceneObject"]:
    """Register a :class:`SceneObject` subclass for deserialisation.

    Also restores identity hashing. ``@dataclass`` generates an ``__eq__`` for
    each subclass and, in doing so, resets ``__hash__`` to ``None`` -- which
    would make the object unusable in napari's selectable evented list. Scene
    objects need value equality *and* identity hashing, so the decorator (which
    runs after ``@dataclass``) puts the hash back.
    """
    _KINDS[cls.kind] = cls
    cls.__hash__ = SceneObject.__hash__
    return cls


@dataclass
class SceneObject:
    """An animatable object in the 3D scene.

    Parameters
    ----------
    name : str
        User-facing name, shown in the scene list and editable.
    enabled : bool
        Whether the object contributes to the scene. This is the per-keyframe
        on/off switch.
    targets : tuple of str
        Names of the layers this object acts on. Empty (the default) means
        every eligible layer.
    id : str
        Stable identifier, used to match the object across keyframes. Assigned
        automatically; only set it explicitly when restoring from a file.
    """

    name: str = "Object"
    enabled: bool = True
    targets: Tuple[str, ...] = ()
    id: str = field(default_factory=lambda: uuid4().hex)

    #: Serialised discriminator, set by each subclass.
    kind: ClassVar[str] = "object"

    def __hash__(self) -> int:
        # Hash by identity so scene objects can live in napari's selectable
        # evented list. Equality stays field-based, which is what the
        # serialisation round trip checks.
        return id(self)

    def to_dict(self) -> dict:
        """Return a plain-dict snapshot of this object's parameters."""
        state = asdict(self)
        state["kind"] = self.kind
        state["targets"] = tuple(self.targets)
        return state

    @staticmethod
    def from_dict(state: dict) -> "SceneObject":
        """Rebuild a scene object from a dict written by :meth:`to_dict`."""
        state = dict(state)
        kind = state.pop("kind", None)
        cls = _KINDS.get(kind)
        if cls is None:
            raise ValueError(f"Unknown scene object kind: {kind!r}")
        fields = {f for f in cls.__dataclass_fields__}
        known = {k: v for k, v in state.items() if k in fields}
        if "targets" in known:
            known["targets"] = tuple(known["targets"])
        return cls(**known)

    def target_layers(self, viewer: "napari.viewer.Viewer") -> Iterator:
        """Yield the layers this object acts on.

        Only layers that support clipping planes are considered, and layers
        named in :attr:`targets` that are not present are skipped rather than
        raising -- a saved animation may be replayed against a viewer that is
        missing some layers.
        """
        for layer in viewer.layers:
            if not hasattr(layer, "experimental_clipping_planes"):
                continue
            if self.targets and layer.name not in self.targets:
                continue
            yield layer

    def clipping_planes(
        self, viewer: "napari.viewer.Viewer"
    ) -> Iterator[Tuple[str, List[dict]]]:
        """Yield ``(layer_name, planes)`` contributions for this object.

        Subclasses that cut geometry override this. The planes are in each
        layer's own data coordinates, ready to be handed to napari.
        """
        return iter(())


def world_to_data_position(layer, position) -> np.ndarray:
    """Convert a world-space position to a layer's data coordinates.

    ``position`` is a spatial (3D) point; it is placed on the layer's trailing
    axes so that layers with extra leading dimensions (time, channel) are
    handled by napari's own trailing-dims convention.
    """
    world = np.zeros(layer.ndim, dtype=float)
    world[-len(position) :] = position
    return np.asarray(layer.world_to_data(world), dtype=float)


def world_to_data_normal(layer, normal) -> np.ndarray:
    """Convert a world-space normal vector to a layer's data coordinates.

    Normals do not transform like positions: under a non-uniform scale they
    must be transformed by the transpose of the layer-to-world linear matrix,
    or an anisotropic volume ends up with planes that are not perpendicular to
    the direction the user chose.

    See https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals.html
    """
    world = np.zeros(layer.ndim, dtype=float)
    world[-len(normal) :] = normal

    magnitude = np.linalg.norm(world)
    if magnitude == 0:
        return world
    world = world / magnitude

    matrix = layer._transforms[1:].simplified.linear_matrix
    transformed = np.matmul(matrix.T, world)

    magnitude = np.linalg.norm(transformed)
    if magnitude == 0:
        return transformed
    return transformed / magnitude


def data_to_world_normal(layer, normal) -> np.ndarray:
    """Convert a layer-space normal vector back to world coordinates.

    The inverse of :func:`world_to_data_normal`, used when adopting clipping
    planes that were set directly on a layer.
    """
    world = np.zeros(layer.ndim, dtype=float)
    world[-len(normal) :] = normal

    magnitude = np.linalg.norm(world)
    if magnitude == 0:
        return world
    world = world / magnitude

    matrix = layer._transforms[1:].simplified.inverse.linear_matrix
    transformed = np.matmul(matrix.T, world)

    magnitude = np.linalg.norm(transformed)
    if magnitude == 0:
        return transformed
    return transformed / magnitude
