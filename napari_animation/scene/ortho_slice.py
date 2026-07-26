"""An animatable ortho slice -- the Imaris "ortho slicer" object.

An :class:`OrthoSlice` is a slab through a volume: a centre plane, a normal,
and a thickness, displayed as a projection through that thickness. Sweeping it
across an animation is interpolation of ``position``; tilting it is
interpolation of ``normal``.

Two render modes, trading memory against appearance:

``"plane"`` (default)
    Backed by a real napari Image layer sharing the source layer's data, in
    napari's ``depiction='plane'`` mode. This is the true Imaris ortho slicer:
    an oblique thick slab drawn *alongside* the volume, and -- because it is a
    genuine layer -- it appears in napari's layer list with its own visibility
    checkbox. Costs one GPU texture per slice.

``"clip"``
    A slab carved out of the source layer by a pair of opposed clipping planes
    contributed to the scene compositor. No extra texture, but the rest of the
    volume is cut away rather than shown alongside the slab.

The backing layer is the source of truth for the slab geometry:
:meth:`OrthoSlice.sync_from_viewer` reads ``layer.plane`` back out before a
keyframe is captured. That means any external tool that moves the plane -- a
napari-threedee manipulator, napari's own controls, a user script -- is picked
up automatically, with no glue code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

import numpy as np

from .base import (
    SceneObject,
    register_kind,
    world_to_data_normal,
    world_to_data_position,
)

if TYPE_CHECKING:
    import napari

#: How an Imaris-style projection type maps onto a napari volume rendering
#: mode. Projecting through the slab *is* the rendering mode in napari.
PROJECTION_TO_RENDERING = {
    "max": "mip",
    "mean": "average",
    "min": "minip",
}

#: Named orthogonal views, as world-space normals on the trailing (Z, Y, X)
#: spatial axes. "XY" is the plane you see looking down Z, so it is sliced
#: along Z -- matching :data:`napari_animation.ortho_slicer.ORTHO_VIEWS`.
VIEW_NORMALS = {
    "XY": (1.0, 0.0, 0.0),
    "XZ": (0.0, 1.0, 0.0),
    "YZ": (0.0, 0.0, 1.0),
}


@register_kind
@dataclass
class OrthoSlice(SceneObject):
    """A thick optical section through a volume.

    Parameters
    ----------
    source : str, optional
        Name of the layer whose data is sliced. When ``None`` the first image
        layer with 3 or more dimensions is used.
    position : tuple of float
        Centre of the slab, in world coordinates, ordered ``(z, y, x)``.
    normal : tuple of float
        Slab normal in world coordinates. Use :data:`VIEW_NORMALS` for the
        named orthogonal views, or any vector for an oblique section.
    thickness : float
        Slab thickness in world units, measured along the normal.
    projection : str
        How the slab is projected: ``"max"``, ``"mean"`` or ``"min"``.
    render_mode : str
        ``"plane"`` for a slab layer drawn alongside the volume, ``"clip"``
        for a slab carved out of the source layer.
    """

    source: Optional[str] = None
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    thickness: float = 1.0
    projection: str = "max"
    render_mode: str = "plane"

    kind = "ortho_slice"

    #: The plane geometry this object last wrote to its backing layer, used to
    #: tell an external edit from one made through this object. Deliberately
    #: not a dataclass field: it is bookkeeping, not captured state.
    _applied_plane = None

    # ------------------------------------------------------------- geometry

    @property
    def layer_name(self) -> str:
        """Name of the backing napari layer for this slice."""
        return self.name

    def source_layer(self, viewer: "napari.viewer.Viewer"):
        """Return the layer whose data this slice cuts through."""
        if self.source is not None:
            layer = (
                viewer.layers[self.source]
                if (self.source in viewer.layers)
                else None
            )
            return layer
        for layer in viewer.layers:
            if layer.name == self.layer_name:
                continue
            if getattr(layer, "ndim", 0) >= 3 and hasattr(layer, "plane"):
                return layer
        return None

    def data_thickness(self, layer) -> float:
        """Convert the world-units thickness into the layer's data units.

        napari's ``plane.thickness`` is in data coordinates, so an anisotropic
        volume needs the world offset projected onto the data-space normal --
        simply dividing by one axis' scale would be wrong for oblique planes.
        """
        normal = np.zeros(layer.ndim, dtype=float)
        normal[-len(self.normal) :] = self.normal
        magnitude = np.linalg.norm(normal)
        if magnitude == 0:
            return 1.0
        normal /= magnitude

        origin_data = np.asarray(
            layer.world_to_data(np.zeros(layer.ndim)), dtype=float
        )
        offset_data = (
            np.asarray(
                layer.world_to_data((self.thickness / 2.0) * normal),
                dtype=float,
            )
            - origin_data
        )
        normal_data = world_to_data_normal(layer, self.normal)
        return float(2.0 * abs(np.dot(offset_data, normal_data)))

    # ----------------------------------------------------------- rendering

    def ensure_layer(self, viewer: "napari.viewer.Viewer"):
        """Create or return this slice's backing layer."""
        if self.layer_name in viewer.layers:
            return viewer.layers[self.layer_name]

        source = self.source_layer(viewer)
        if source is None:
            return None

        data = source.data
        if getattr(source, "multiscale", False):
            # plane depiction reads a single 3D texture; use the coarsest
            # level so a large pyramid does not blow up VRAM
            data = data[-1]

        layer = viewer.add_image(
            data,
            name=self.layer_name,
            scale=source.scale,
            translate=source.translate,
            rotate=source.rotate,
            shear=source.shear,
            opacity=source.opacity,
            depiction="plane",
            rendering=PROJECTION_TO_RENDERING.get(self.projection, "mip"),
        )
        try:
            layer.colormap = source.colormap.name
            layer.contrast_limits = list(source.contrast_limits)
        except (AttributeError, ValueError, KeyError):
            pass
        return layer

    def remove_layer(self, viewer: "napari.viewer.Viewer") -> None:
        """Remove this slice's backing layer, if it exists."""
        if self.layer_name in viewer.layers:
            viewer.layers.remove(viewer.layers[self.layer_name])

    def apply(self, viewer: "napari.viewer.Viewer") -> None:
        """Push this slice's geometry onto its backing layer."""
        if self.render_mode != "plane":
            # in clip mode the slab is contributed to the compositor instead;
            # hide any backing layer left over from a mode switch
            if self.layer_name in viewer.layers:
                viewer.layers[self.layer_name].visible = False
            return

        layer = self.ensure_layer(viewer)
        if layer is None:
            return

        layer.visible = bool(self.enabled)
        if not self.enabled:
            return

        layer.depiction = "plane"
        layer.rendering = PROJECTION_TO_RENDERING.get(self.projection, "mip")
        plane = {
            "position": tuple(
                world_to_data_position(layer, self.position).tolist()
            ),
            "normal": tuple(world_to_data_normal(layer, self.normal).tolist()),
            "thickness": max(1e-6, self.data_thickness(layer)),
        }
        layer.plane = plane
        self._applied_plane = plane

    def sync_from_viewer(self, viewer: "napari.viewer.Viewer") -> None:
        """Adopt the slab geometry from the backing layer if it was moved.

        The layer is the source of truth *for external edits*: a
        napari-threedee manipulator or napari's own controls write straight to
        ``layer.plane``, and those edits must be what gets captured. But
        changes made through this object have not reached the layer yet, so
        blindly reading back would silently discard them.

        Comparing against the geometry last written in :meth:`apply`
        distinguishes the two: if the layer no longer matches, something else
        moved it.
        """
        if self.render_mode != "plane":
            return
        if self.layer_name not in viewer.layers:
            return
        layer = viewer.layers[self.layer_name]
        plane = getattr(layer, "plane", None)
        if plane is None:
            return

        applied = self._applied_plane
        if (
            applied is not None
            and np.allclose(plane.position, applied["position"])
            and np.allclose(plane.normal, applied["normal"])
        ):
            # the layer still holds what we put there; this object's own
            # parameters are the newer truth
            return

        from .base import data_to_world_normal

        world_position = layer.data_to_world(plane.position)
        self.position = tuple(float(v) for v in world_position[-3:])
        self.normal = tuple(
            float(v) for v in data_to_world_normal(layer, plane.normal)[-3:]
        )

    # ------------------------------------------------------ clipping planes

    def clipping_planes(
        self, viewer: "napari.viewer.Viewer"
    ) -> Iterator[Tuple[str, List[dict]]]:
        if self.render_mode != "clip":
            return
        source = self.source_layer(viewer)
        if source is None or source.ndim < 3:
            return

        position = world_to_data_position(source, self.position)
        normal = world_to_data_normal(source, self.normal)
        half = self.data_thickness(source) / 2.0

        yield source.name, [
            {
                "position": tuple((position - half * normal).tolist()),
                "normal": tuple(normal.tolist()),
                "enabled": True,
            },
            {
                "position": tuple((position + half * normal).tolist()),
                "normal": tuple((-normal).tolist()),
                "enabled": True,
            },
        ]
