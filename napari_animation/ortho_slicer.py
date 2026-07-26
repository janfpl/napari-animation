"""Imaris-style orthogonal "optical section" slicing for napari.

The :class:`OrthoSlicer` restricts the displayed data to a slab of a chosen
number of planes (the *optical section thickness*) centered on the currently
selected plane along one axis.  Two display modes are supported:

* ``"projection"`` -- the slab is projected into the displayed 2D slice using
  napari's thick-slice machinery (``dims.margin_left`` / ``dims.margin_right``)
  together with the layer ``projection_mode`` (``max``/``mean``/``min``/
  ``sum``).  This mimics an Imaris thick optical section.
* ``"clip"`` -- a 3D rendering is kept but only the planes inside the slab are
  rendered, using each layer's ``experimental_clipping_planes`` (a moving
  sub-volume).

Only a small set of *parameters* (enabled, thickness, mode, projection type and
axis) are stored on each keyframe.  The actual slab geometry is recomputed from
those parameters and the current dims at apply time, which means the optical
section interpolates smoothly during an animation (e.g. a slab sweeping through
z, or a thickness that grows over the movie).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import napari

#: Available display modes for the ortho slicer.
ORTHO_MODES = ("projection", "clip")
#: Available projection types for the ``"projection"`` mode.
PROJECTION_MODES = ("max", "mean", "min", "sum")

#: Named orthogonal views mapped to the slab (depth) axis, expressed as an
#: offset from the end of the axis list.  napari's convention puts X on the
#: last axis, Y on the second-last and Z on the third-last, so e.g. the "XY"
#: view is sliced along Z.  Listed in canonical (X, Y, Z) order.
ORTHO_VIEWS = {"XY": -3, "XZ": -2, "YZ": -1}


def view_to_axis(view: str, ndim: int) -> int:
    """Return the slab (depth) axis for a named orthogonal view.

    Uses napari's axis convention (last axis = X, second-last = Y, third-last
    = Z), so ``"XY"`` slices along Z, ``"XZ"`` along Y and ``"YZ"`` along X.
    For data with more than three dimensions the named views map to the last
    three (spatial) axes; leading axes (e.g. time) are unaffected.
    """
    offset = ORTHO_VIEWS[view.upper()]
    axis = ndim + offset
    return max(0, min(axis, ndim - 1))


def axis_to_view(axis: int, ndim: int) -> Optional[str]:
    """Return the named orthogonal view for a slab ``axis`` (or ``None``)."""
    for name, offset in ORTHO_VIEWS.items():
        if ndim + offset == axis:
            return name
    return None


#: Map common pint unit names to compact labels for display.
_UNIT_ABBREVIATIONS = {
    "micrometer": "µm",
    "micrometre": "µm",
    "micron": "µm",
    "um": "µm",
    "µm": "µm",
    "nanometer": "nm",
    "nanometre": "nm",
    "nm": "nm",
    "millimeter": "mm",
    "millimetre": "mm",
    "mm": "mm",
    "meter": "m",
    "metre": "m",
    "pixel": "px",
    "pixels": "px",
    "": "px",
}


def physical_step(viewer, axis) -> tuple:
    """Return the world-coordinate size of one plane along ``axis``.

    Returns a ``(step, unit_label)`` tuple, where ``step`` is the physical
    spacing between planes (e.g. the Z scale / voxel depth from the layer
    metadata) and ``unit_label`` is a compact unit string (``"µm"``, ``"px"``,
    ...).  ``step`` is ``None`` if it cannot be determined.
    """
    if axis is None:
        return None, "px"
    try:
        axis_range = viewer.dims.range[axis]
        step = float(getattr(axis_range, "step", axis_range[2]))
    except (AttributeError, IndexError, TypeError):
        return None, "px"
    unit = "px"
    try:
        raw = str(viewer.dims.units[axis]).lower()
        unit = _UNIT_ABBREVIATIONS.get(raw, raw or "px")
    except (AttributeError, IndexError, TypeError):
        pass
    return step, unit


@dataclass
class OrthoSlicer:
    """Configuration for an Imaris-style optical-section slicer.

    Parameters
    ----------
    enabled : bool
        Whether the optical section is active.  When ``False`` the slicer
        restores the full view.
    thickness : int
        Optical section thickness, expressed as a number of planes centered on
        the currently selected plane.  ``1`` shows a single plane.
    mode : str
        Either ``"projection"`` (2D thick slice) or ``"clip"`` (3D sub-volume).
    projection_mode : str
        How the slab is projected in ``"projection"`` mode: one of
        ``"max"``, ``"mean"``, ``"min"`` or ``"sum"``.
    axis : int, optional
        The axis the optical-section slab runs along.  When ``None`` (default)
        the slicer uses the first not-displayed axis of the viewer (the active
        slider axis), falling back to the Z axis in a 3D display -- i.e. an XY
        optical section by default.  Use :func:`view_to_axis` to set this from a
        named orthogonal view (``"XY"`` / ``"XZ"`` / ``"YZ"``).
    """

    enabled: bool = False
    thickness: int = 1
    mode: str = "projection"
    projection_mode: str = "max"
    axis: Optional[int] = None

    def to_dict(self) -> dict:
        """Return a plain-dict snapshot of the slicer parameters."""
        return asdict(self)

    @classmethod
    def from_dict(cls, state: Optional[dict]) -> "OrthoSlicer":
        """Create an :class:`OrthoSlicer` from a dict (``None`` -> disabled)."""
        if not state:
            return cls()
        fields = {"enabled", "thickness", "mode", "projection_mode", "axis"}
        return cls(**{k: v for k, v in state.items() if k in fields})

    def apply(self, viewer: "napari.viewer.Viewer") -> None:
        """Push this slicer's state onto a live ``viewer``."""
        self.apply_state(viewer, self.to_dict())

    def resolve_axis(self, viewer: "napari.viewer.Viewer"):
        """Return the concrete slab axis this slicer would use for ``viewer``."""
        return self._resolve_axis(viewer, self.axis)

    # -- the heavy lifting is done by stateless helpers so that a keyframe's
    #    stored parameters can be applied without an OrthoSlicer instance --

    @staticmethod
    def apply_state(
        viewer: "napari.viewer.Viewer",
        params: Optional[dict],
        assign_clipping_planes: bool = True,
    ) -> None:
        """Apply ortho-slicer ``params`` (or reset the view if disabled).

        Parameters
        ----------
        assign_clipping_planes : bool
            Whether clip mode should write the slab's clipping planes onto the
            layers itself. Set to ``False`` when a scene compositor owns the
            layers' clipping planes and will fold in this slicer's
            contribution (see :func:`napari_animation.scene.apply_scene_state`)
            -- otherwise the two would overwrite each other.
        """
        if not params or not params.get("enabled", False):
            OrthoSlicer._reset(
                viewer, clear_clipping_planes=assign_clipping_planes
            )
            return

        thickness = max(1, int(round(params.get("thickness", 1))))
        mode = params.get("mode", "projection")
        axis = OrthoSlicer._resolve_axis(viewer, params.get("axis"))
        if axis is None:
            # Nothing sensible to slice (e.g. projection mode with no
            # not-displayed axis); leave the view untouched but reset.
            OrthoSlicer._reset(
                viewer, clear_clipping_planes=assign_clipping_planes
            )
            return

        if mode == "clip":
            OrthoSlicer._apply_clip(
                viewer, axis, thickness, assign=assign_clipping_planes
            )
        else:
            projection = params.get("projection_mode", "max")
            OrthoSlicer._apply_projection(
                viewer,
                axis,
                thickness,
                projection,
                clear_clipping_planes=assign_clipping_planes,
            )

    @staticmethod
    def clip_contributions(
        viewer: "napari.viewer.Viewer", params: Optional[dict]
    ) -> dict:
        """Return the clipping planes this slicer contributes, per layer.

        Empty unless the slicer is enabled and in ``"clip"`` mode. Used by the
        scene compositor so the optical section and any cutaway planes
        accumulate rather than overwrite one another.
        """
        if not params or not params.get("enabled", False):
            return {}
        if params.get("mode", "projection") != "clip":
            return {}
        axis = OrthoSlicer._resolve_axis(viewer, params.get("axis"))
        if axis is None:
            return {}
        thickness = max(1, int(round(params.get("thickness", 1))))
        return OrthoSlicer._slab_planes(viewer, axis, thickness)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _resolve_axis(viewer, axis):
        if axis is not None:
            return int(axis)
        not_displayed = tuple(viewer.dims.not_displayed)
        if not_displayed:
            return int(not_displayed[0])
        # 3D display has no slider axis; default to Z (an XY optical section).
        ndim = viewer.dims.ndim
        if ndim >= 3:
            return ndim - 3
        return None

    @staticmethod
    def _axis_step(viewer, axis) -> float:
        """World-coordinate size of one plane along ``axis``."""
        try:
            step = float(viewer.dims.range[axis].step)
        except (AttributeError, TypeError, IndexError):
            # Older napari exposes range as a plain (start, stop, step) tuple.
            step = float(viewer.dims.range[axis][2])
        return step or 1.0

    @staticmethod
    def _apply_projection(
        viewer, axis, thickness, projection, clear_clipping_planes=True
    ):
        step = OrthoSlicer._axis_step(viewer, axis)
        # half-width in world units so that exactly ``thickness`` planes are
        # included symmetrically around the center plane.
        half = ((thickness - 1) / 2.0) * step

        margin_left = list(viewer.dims.margin_left)
        margin_right = list(viewer.dims.margin_right)
        margin_left[axis] = half
        margin_right[axis] = half
        viewer.dims.margin_left = tuple(margin_left)
        viewer.dims.margin_right = tuple(margin_right)

        for layer in viewer.layers:
            # clip mode (if previously active) is mutually exclusive
            if clear_clipping_planes:
                OrthoSlicer._clear_clipping_planes(layer)
            if hasattr(layer, "projection_mode"):
                try:
                    layer.projection_mode = projection
                except (ValueError, KeyError):
                    # e.g. labels layers only allow a subset of modes
                    pass

    @staticmethod
    def _slab_planes(viewer, axis, thickness) -> dict:
        """Return the pair of planes bounding the slab, keyed by layer name."""
        step = OrthoSlicer._axis_step(viewer, axis)
        center = float(viewer.dims.point[axis])
        half = (thickness / 2.0) * step
        low, high = center - half, center + half

        contributions = {}
        for layer in viewer.layers:
            if not hasattr(layer, "experimental_clipping_planes"):
                continue
            # align the (global) viewer axis with this layer's own axes, which
            # may have fewer dimensions (trailing-dims convention).
            layer_axis = axis - (viewer.dims.ndim - layer.ndim)
            if layer_axis < 0 or layer_axis >= layer.ndim:
                continue
            scale = float(layer.scale[layer_axis]) or 1.0
            translate = float(layer.translate[layer_axis])
            data_low = (low - translate) / scale
            data_high = (high - translate) / scale

            pos_low = [0.0] * layer.ndim
            pos_high = [0.0] * layer.ndim
            normal = [0.0] * layer.ndim
            pos_low[layer_axis] = data_low
            pos_high[layer_axis] = data_high
            normal[layer_axis] = 1.0

            contributions[layer.name] = [
                {
                    "position": tuple(pos_low),
                    "normal": tuple(normal),
                    "enabled": True,
                },
                {
                    "position": tuple(pos_high),
                    "normal": tuple(-n for n in normal),
                    "enabled": True,
                },
            ]
        return contributions

    @staticmethod
    def _apply_clip(viewer, axis, thickness, assign=True):
        # projection margins are mutually exclusive with clip mode
        OrthoSlicer._reset_margins(viewer)

        if not assign:
            # a scene compositor owns the layers' clipping planes and will
            # fold in this slab via ``clip_contributions``
            return

        for name, planes in OrthoSlicer._slab_planes(
            viewer, axis, thickness
        ).items():
            viewer.layers[name].experimental_clipping_planes = planes

    @staticmethod
    def _reset(viewer, clear_clipping_planes=True):
        OrthoSlicer._reset_margins(viewer)
        for layer in viewer.layers:
            if clear_clipping_planes:
                OrthoSlicer._clear_clipping_planes(layer)
            if hasattr(layer, "projection_mode"):
                try:
                    layer.projection_mode = "none"
                except (ValueError, KeyError):
                    pass

    @staticmethod
    def _reset_margins(viewer):
        ndim = viewer.dims.ndim
        viewer.dims.margin_left = (0.0,) * ndim
        viewer.dims.margin_right = (0.0,) * ndim

    @staticmethod
    def _clear_clipping_planes(layer):
        if hasattr(layer, "experimental_clipping_planes"):
            try:
                layer.experimental_clipping_planes = []
            except (ValueError, TypeError):
                pass
