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

from ..perf import timed_render
from ..layer_registry import (
    LayerRegistry,
    OBJECT_ID_METADATA,
)
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
    # Stable source identity.  ``source`` remains as a display-name fallback
    # for files written before the registry existed.
    source_id: Optional[str] = None
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    thickness: float = 1.0
    projection: str = "max"
    render_mode: str = "plane"
    # Pyramid level used by the backing layer.  ``-1`` means the coarsest
    # available level for multiscale data; 0 is the finest (default).
    source_level: int = 0
    sampling_convention: str = "voxel-centers-half-open-v1"
    # None denotes a legacy single-source section.  An empty mapping is an
    # intentionally empty composite, and must not select the first image.
    channels: Optional[dict] = None

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
        registry = LayerRegistry(viewer)
        if self.source_id:
            layer = registry.find(self.source_id)
            # A stable binding is authoritative.  Display names can be reused
            # after a rename and must never silently redirect a section.
            if layer is not None:
                self.source = layer.name
            return layer
        if self.source is not None:
            layer = registry.find(name=self.source)
            if layer is not None:
                self.source_id = registry.source_id(layer)
            return layer
        for layer in viewer.layers:
            if layer.metadata.get(OBJECT_ID_METADATA):
                continue
            if getattr(layer, "ndim", 0) >= 3 and hasattr(layer, "plane"):
                self.source_id = registry.source_id(layer)
                self.source = layer.name
                return layer
        return None

    def _channel_sections(self, viewer):
        """Materialize per-source render adapters for a composite section.

        The adapters use the existing scalar napari plane path.  Their stable
        IDs give each channel its own managed texture and projection request;
        the parent remains the only scene/keyframe object.
        """
        if self.channels is None:
            return (self,)
        registry = LayerRegistry(viewer)
        children = getattr(self, "_channel_children", {})
        current = {}
        for source_id, settings in self.channels.items():
            if not isinstance(settings, dict) or not settings.get("enabled", False):
                continue
            layer = registry.find(source_id)
            if layer is None:
                continue
            child_id = self.id if source_id == self.source_id else f"{self.id}::channel::{source_id}"
            child = children.get(source_id)
            if child is None:
                child = OrthoSlice(id=child_id)
            child._composite_owner = self
            child.name = (
                self.name if child_id == self.id else f"{self.name} · {layer.name}"
            )
            child.enabled = self.enabled
            child.source = layer.name
            child.source_id = source_id
            child.position = self.position
            child.normal = self.normal
            child.thickness = self.thickness
            child.projection = self.projection
            child.render_mode = self.render_mode
            child.source_level = int(settings.get("source_level", self.source_level))
            child.sampling_convention = self.sampling_convention
            current[source_id] = child
        self._channel_children = current
        return tuple(current.values())

    def source_layers(self, viewer, include_disabled=False):
        if self.channels is None:
            source = self.source_layer(viewer)
            return (source,) if source is not None else ()
        registry = LayerRegistry(viewer)
        return tuple(
            layer for source_id, settings in self.channels.items()
            if (include_disabled or settings.get("enabled", False))
            for layer in (registry.find(source_id),) if layer is not None
        )

    @staticmethod
    def _colormap_spec(colormap):
        """Return a JSON-safe name or the full definition of a custom LUT."""
        from napari.utils.colormaps import ALL_COLORMAPS

        registered = ALL_COLORMAPS.get(colormap.name)
        if registered is not None and (
            np.array_equal(registered.colors, colormap.colors)
            and np.array_equal(registered.controls, colormap.controls)
        ):
            return colormap.name
        return {
            "name": colormap.name,
            "colors": np.asarray(colormap.colors).tolist(),
            "controls": np.asarray(colormap.controls).tolist(),
            "interpolation": colormap.interpolation.value,
        }

    @staticmethod
    def _apply_channel_appearance(output, source, settings):
        custom = settings.get("appearance") if settings.get("appearance_mode") == "custom" else None
        custom = custom or {}
        colormap = custom.get("colormap", source.colormap)
        if isinstance(colormap, dict):
            from napari.utils.colormaps import Colormap

            if OrthoSlice._colormap_spec(output.colormap) != colormap:
                output.colormap = Colormap(
                    colors=colormap["colors"],
                    controls=colormap["controls"],
                    interpolation=colormap["interpolation"],
                    name=colormap["name"],
                )
        elif output.colormap.name != (
            colormap if isinstance(colormap, str) else colormap.name
        ):
            output.colormap = colormap
        limits = custom.get("contrast_limits", source.contrast_limits)
        if not np.array_equal(output.contrast_limits, limits):
            output.contrast_limits = list(limits)
        gamma = custom.get("gamma", source.gamma)
        if output.gamma != gamma:
            output.gamma = gamma
        opacity = custom.get("opacity", source.opacity)
        if output.opacity != opacity:
            output.opacity = opacity

    def _bind_composite_appearance_events(self, viewer, source):
        """Keep linked plane colors current when napari edits a source."""
        registry = LayerRegistry(viewer)
        source_id = registry.source_id(source)
        bindings = getattr(source, "_napari_animation_composite_appearance", {})
        previous = bindings.pop(self.id, None)
        if previous is not None and previous[0] is self:
            bindings[self.id] = previous
            return
        if previous is not None:
            for emitter in previous[2]:
                emitter.disconnect(previous[1])

        def changed(_event):
            renderer = getattr(viewer, "_napari_animation_renderer", None)
            if renderer is not None and renderer.applying:
                return
            settings = (self.channels or {}).get(source_id)
            if settings is None or settings.get("appearance_mode") == "custom":
                return
            for child in self._channel_sections(viewer):
                if child.source_id != source_id:
                    continue
                for role in ("section_3d", "section_2d"):
                    output = registry.find_managed(child.id, role)
                    if output is not None:
                        self._apply_channel_appearance(output, source, settings)

        emitters = []
        for name in ("colormap", "contrast_limits", "gamma", "opacity"):
            emitter = getattr(source.events, name, None)
            if emitter is not None:
                emitter.connect(changed)
                emitters.append(emitter)
        bindings[self.id] = (self, changed, emitters)
        source._napari_animation_composite_appearance = bindings

    def _capture_channel_appearance(self, viewer, child):
        source = child.source_layer(viewer)
        managed = LayerRegistry(viewer).find_managed(child.id, "section_3d")
        if source is None or managed is None:
            return
        settings = self.channels[child.source_id]
        custom = (
            settings.get("appearance") or {}
            if settings.get("appearance_mode") == "custom" else {}
        )
        expected_colormap = custom.get(
            "colormap", self._colormap_spec(source.colormap)
        )
        expected_limits = custom.get("contrast_limits", source.contrast_limits)
        expected_gamma = custom.get("gamma", source.gamma)
        expected_opacity = custom.get("opacity", source.opacity)
        if (
            self._colormap_spec(managed.colormap) != expected_colormap
            or not np.array_equal(managed.contrast_limits, expected_limits)
            or managed.gamma != expected_gamma
            or managed.opacity != expected_opacity
        ):
            settings["appearance_mode"] = "custom"
            settings["appearance"] = {
                "colormap": self._colormap_spec(managed.colormap),
                "contrast_limits": list(managed.contrast_limits),
                "gamma": float(managed.gamma),
                "opacity": float(managed.opacity),
            }

    def validate(self, viewer: "napari.viewer.Viewer", for_2d=False,
                 linear_matrix=None) -> None:
        if self.channels is not None:
            if not isinstance(self.channels, dict):
                raise ValueError("Section channels must be a mapping")
            for source_id, settings in self.channels.items():
                if not isinstance(source_id, str) or not isinstance(settings, dict):
                    raise ValueError("Invalid section channel binding")
                if settings.get("source_id", source_id) != source_id:
                    raise ValueError("Section channel ID does not match its binding")
            children = self._channel_sections(viewer)
            if not children:
                OrthoSlice(
                    source="<unbound composite>",
                    source_id="<unbound composite>",
                    position=self.position,
                    normal=self.normal,
                    thickness=self.thickness,
                    projection=self.projection,
                    render_mode=self.render_mode,
                    sampling_convention=self.sampling_convention,
                ).validate(viewer)
            for child in children:
                child.validate(viewer, for_2d=for_2d)
            return
        if self.projection not in PROJECTION_TO_RENDERING:
            raise ValueError(f"Unknown projection {self.projection!r}")
        if self.render_mode not in {"plane", "clip"}:
            raise ValueError(f"Unknown render mode {self.render_mode!r}")
        if not np.isfinite(float(self.thickness)) or self.thickness < 0:
            raise ValueError("Section thickness must be finite and non-negative")
        position = np.asarray(self.position, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Section position must contain three finite values")
        if (
            normal.shape != (3,)
            or not np.isfinite(normal).all()
            or np.linalg.norm(normal) == 0
        ):
            raise ValueError(
                "Section normal must contain three finite, non-zero values"
            )
        if self.sampling_convention != "voxel-centers-half-open-v1":
            raise ValueError(
                f"Unknown sampling convention {self.sampling_convention!r}"
            )
        source = self.source_layer(viewer)
        if source is not None:
            self._level_volume(viewer, source)
        if for_2d and source is not None and not self.supports_2d(
            viewer, linear_matrix=linear_matrix
        ):
            raise ValueError(
                "Oblique or transformed ortho sections require a 3D view; "
                "choose an axis-aligned section for 2D"
            )

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

    def supports_2d(self, viewer: "napari.viewer.Viewer", linear_matrix=None) -> bool:
        """Whether napari's native 2D slicer can represent this section.

        Native dims margins can sample an axis-aligned slab only.  A rotation,
        shear, or oblique authored normal needs a resampling backend; silently
        choosing the dominant axis would display a different section.
        """
        source = self.source_layer(viewer)
        if source is None or source.ndim < 3:
            return False
        matrix = np.asarray(
            source._transforms[1:].simplified.linear_matrix
            if linear_matrix is None else linear_matrix, dtype=float
        )
        world_normal = np.zeros(source.ndim)
        world_normal[-3:] = self.normal
        normal = (matrix.T @ world_normal)[-3:]
        if not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
            return False
        normal /= np.linalg.norm(normal)
        if np.max(np.abs(normal)) < 1.0 - 1e-6:
            return False
        spatial = matrix[-3:, -3:]
        if matrix.shape[0] > 3:
            if np.max(np.abs(matrix[-3:, :-3])) > 1e-8:
                return False
            if np.max(np.abs(matrix[:-3, -3:])) > 1e-8:
                return False
        nonzero_rows = np.sum(np.abs(spatial) > 1e-8, axis=1)
        nonzero_cols = np.sum(np.abs(spatial) > 1e-8, axis=0)
        return bool(np.all(nonzero_rows == 1) and np.all(nonzero_cols == 1))

    def projection_oracle(self, viewer: "napari.viewer.Viewer") -> np.ndarray:
        """Return the expected axis-aligned 2D projection as a NumPy array.

        This is a numerical oracle for tests and headless validation.  It uses
        voxel centres in world units and the same finite slab interval as the
        native section path; it never changes viewer state.
        """
        source = self.source_layer(viewer)
        if source is None:
            raise ValueError("Ortho slice has no source layer")
        if not self.supports_2d(viewer):
            raise ValueError("oblique or transformed section is not 2D-native")
        data, level_factors, leading = self._level_volume(viewer, source)
        normal = np.asarray(world_to_data_normal(source, self.normal), float)[-3:]
        axis = int(np.argmax(np.abs(normal)))
        source_axis = axis + source.ndim - 3
        centre = (
            world_to_data_position(source, self.position)[source_axis]
            / level_factors[axis]
        )
        thickness = self.data_thickness(source) / level_factors[axis]
        interval = self._sample_interval(
            float(centre), float(thickness), data.shape[axis]
        )
        if interval is None:
            shape = tuple(
                size for index, size in enumerate(data.shape) if index != axis
            )
            projected = np.zeros(shape, dtype=data.dtype)
            return self._world_aligned_projection(
                projected,
                data.shape,
                axis,
                self._spatial_affine(source, level_factors, leading),
                0.0,
            )[0]
        start, stop = interval
        index = [slice(None)] * 3
        index[axis] = slice(start, stop)
        slab = np.asarray(data[tuple(index)])
        if self.projection == "mean":
            projected = np.mean(slab, axis=axis)
        elif self.projection == "min":
            projected = np.min(slab, axis=axis)
        else:
            projected = np.max(slab, axis=axis)
        return self._world_aligned_projection(
            projected,
            data.shape,
            axis,
            self._spatial_affine(source, level_factors, leading),
            (start + stop - 1) / 2.0,
        )[0]

    def _level_volume(self, viewer, source, level_override=None):
        """Return the selected pyramid/time volume and its spatial factors."""
        data = source.data
        factors = np.ones(source.ndim, dtype=float)
        level = 0
        if getattr(source, "multiscale", False):
            level = self.source_level if level_override is None else level_override
            if level < 0:
                level += len(data)
            if level < 0 or level >= len(data):
                raise ValueError(
                    f"source_level {self.source_level} is out of range for "
                    f"{source.name!r}"
                )
            data = data[level]
            factors = np.asarray(source.downsample_factors[level], dtype=float)

        leading_count = max(0, int(np.ndim(data)) - 3)
        leading = []
        if leading_count:
            world = np.asarray(viewer.dims.point[-source.ndim :], dtype=float)
            point = np.asarray(source.world_to_data(world), dtype=float) / factors
            for axis in range(leading_count):
                index = int(np.clip(round(point[axis]), 0, data.shape[axis] - 1))
                leading.append(index)
            data = data[tuple(leading) + (slice(None),) * 3]
        if np.ndim(data) != 3:
            raise ValueError("Ortho sections require at least three spatial axes")
        return data, np.asarray(factors[-3:], dtype=float), tuple(leading)

    def _spatial_affine(self, source, factors, leading=(), level_override=None):
        """Map selected-level spatial coordinates to trailing world axes."""
        matrix = np.asarray(
            source._transforms[1:].simplified.affine_matrix, dtype=float
        )
        spatial = np.eye(4, dtype=float)
        spatial[:3, :3] = matrix[-4:-1, -4:-1]
        spatial[:3, 3] = matrix[-4:-1, -1]
        if leading:
            lead = np.asarray(leading, dtype=float)
            all_factors = np.ones(source.ndim, dtype=float)
            if getattr(source, "multiscale", False):
                level = (
                    self.source_level if level_override is None else level_override
                )
                if level < 0:
                    level += len(source.data)
                all_factors = np.asarray(source.downsample_factors[level], dtype=float)
            spatial[:3, 3] += matrix[-4:-1, : len(leading)] @ (
                lead * all_factors[: len(leading)]
            )
        spatial[:3, :3] = spatial[:3, :3] @ np.diag(factors)
        return spatial

    @staticmethod
    def _world_aligned_projection(
        projected, volume_shape, source_axis, affine, effective
    ):
        """Reorient a signed-permutation data projection into world axes."""
        linear = np.asarray(affine[:3, :3], dtype=float)
        world_for_data = {
            data_axis: int(np.argmax(np.abs(linear[:, data_axis])))
            for data_axis in range(3)
        }
        hidden_world = world_for_data[source_axis]
        remaining_data = [axis for axis in range(3) if axis != source_axis]
        remaining_world = [axis for axis in range(3) if axis != hidden_world]
        ordered_data = [
            next(axis for axis in remaining_data if world_for_data[axis] == world_axis)
            for world_axis in remaining_world
        ]
        transpose = [remaining_data.index(axis) for axis in ordered_data]
        world_2d = np.transpose(np.asarray(projected), transpose)
        for output_axis, data_axis in enumerate(ordered_data):
            world_axis = world_for_data[data_axis]
            if linear[world_axis, data_axis] < 0:
                world_2d = np.flip(world_2d, axis=output_axis)

        output = np.expand_dims(world_2d, axis=hidden_world)
        output_affine = np.eye(4, dtype=float)
        output_affine[:3, :3] = 0.0
        output_affine[:3, 3] = np.asarray(affine[:3, 3], dtype=float)
        for data_axis, world_axis in world_for_data.items():
            step = float(linear[world_axis, data_axis])
            output_affine[world_axis, world_axis] = abs(step)
            if data_axis == source_axis:
                output_affine[world_axis, 3] += step * effective
            elif step < 0:
                output_affine[world_axis, 3] += step * (volume_shape[data_axis] - 1)
        return world_2d, output, output_affine

    @staticmethod
    def _sample_interval(
        centre: float, thickness: float, size: int
    ) -> Optional[Tuple[int, int]]:
        """Return the exact voxel interval used by napari's native slicer.

        Voxel centers are integer data coordinates.  New sections use the
        half-open physical interval ``[centre - t/2, centre + t/2)``.  Napari
        0.8 rounds the resulting data bounds and adds one voxel when the upper
        bound lands exactly on an integer, so the renderer uses equivalent
        integer bounds directly and maps them back to effective point/margins.
        ``None`` means the physical interval does not overlap the voxel
        support ``[-0.5, size - 0.5)``.
        """
        if size <= 0 or not np.isfinite(centre) or not np.isfinite(thickness):
            return None
        thickness = max(0.0, float(thickness))
        low = float(centre) - thickness / 2.0
        high = float(centre) + thickness / 2.0
        if high <= -0.5 or low >= float(size) - 0.5:
            return None

        # Subtract a tiny tolerance so a transform which produces 2.0 + 1e-15
        # does not unexpectedly include the next voxel.  Values meaningfully
        # above an integer still round up as required by the half-open rule.
        start = max(0, int(np.ceil(low - 1e-9)))
        stop = min(size, int(np.ceil(high - 1e-9)))
        if stop <= start:
            # A sub-voxel interval can overlap the support without containing
            # a voxel centre.  Choose the nearest valid centre, breaking ties
            # toward the lower index, then render exactly that one voxel.
            nearest = int(np.ceil(float(centre) - 0.5))
            nearest = int(np.clip(nearest, 0, size - 1))
            return nearest, nearest + 1
        return start, stop

    @staticmethod
    def world_thickness(layer, data_normal, data_thickness) -> float:
        """Convert a napari data-space thickness back to world units."""
        normal = np.asarray(data_normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return float(data_thickness)
        normal /= norm
        matrix = layer._transforms[1:].simplified.linear_matrix
        # If n_d is unit length, the corresponding unit world normal is
        # normalize(A^-T n_d).  Since n_d = A^T n_w / ||A^T n_w||, the
        # world/data thickness factor is the reciprocal of that norm.
        inverse_transpose = np.linalg.inv(matrix).T
        factor = np.linalg.norm(inverse_transpose @ normal)
        if factor == 0:
            return float(data_thickness)
        return float(data_thickness) / float(factor)

    # ----------------------------------------------------------- rendering

    @timed_render("ortho.ensure_layer")
    def ensure_layer(self, viewer: "napari.viewer.Viewer", for_2d=False):
        """Create or return this slice's backing layer."""
        if self.channels is not None:
            layers = [
                child.ensure_layer(viewer, for_2d=for_2d)
                for child in self._channel_sections(viewer)
            ]
            return next((layer for layer in layers if layer is not None), None)
        registry = LayerRegistry(viewer)
        source = self.source_layer(viewer)
        if source is None:
            for role in ("section_3d", "section_2d"):
                layer = registry.find_managed(self.id, role)
                if layer is not None:
                    layer.visible = False
            return None

        self._bind_source_events(viewer, source)
        # Clip mode renders the source volume. Its editable plane is hidden,
        # so keep only geometry until a mode actually needs plane voxels.
        defer_volume = for_2d or self.render_mode == "clip"

        data, factors, leading = self._level_volume(viewer, source)
        affine = self._spatial_affine(source, factors, leading)
        base_data = (
            source.data[self.source_level]
            if getattr(source, "multiscale", False)
            else source.data
        )
        binding = (base_data, self.source_level, leading)
        layer = registry.find_managed(self.id, "section_3d")
        if layer is not None:
            # A source edit must rebind the existing managed layer.  Keeping
            # the layer object avoids duplicate napari list entries and keeps
            # any attached canvas manipulator alive.
            # Setting Image.data resets its pyramid/thumbnail state and emits
            # a fresh slice request, even when the array is unchanged. Scene
            # application runs on every button press and sometimes more than
            # once per press, so keep the existing binding when possible.
            previous = getattr(layer, "_napari_animation_data_binding", None)
            same_binding = (
                previous is not None
                and previous[0] is base_data
                and previous[1:] == binding[1:]
            )
            if defer_volume and getattr(
                layer, "_napari_animation_placeholder", False
            ):
                pass
            elif not same_binding and layer.data is not data:
                layer.data = data
                layer._napari_animation_placeholder = False
            if not getattr(layer, "_napari_animation_placeholder", False):
                layer._napari_animation_data_binding = binding
            if not np.array_equal(layer.scale, (1.0, 1.0, 1.0)):
                layer.scale = (1.0, 1.0, 1.0)
            if not np.array_equal(layer.translate, (0.0, 0.0, 0.0)):
                layer.translate = (0.0, 0.0, 0.0)
            if not np.array_equal(layer.rotate, np.eye(3)):
                layer.rotate = 0.0
            if not np.array_equal(layer.shear, (0.0, 0.0, 0.0)):
                layer.shear = (0.0, 0.0, 0.0)
            if not np.array_equal(layer.affine.affine_matrix, affine):
                layer.affine = affine
            # Managed images are views of the source, so their world axes
            # have the same units.  A 4D source contributes its trailing 3
            # spatial axes to this 3D layer.
            spatial_units = tuple(source.units[-3:])
            if tuple(layer.units) != spatial_units:
                layer.units = spatial_units
            if layer.name != self.layer_name:
                layer.name = registry.unique_name(self.layer_name, exclude=layer)
            layer.metadata["napari_animation.source"] = source.name
            layer.metadata["napari_animation.source_level"] = self.source_level
            self._bind_layer_events(layer, viewer)
            return layer

        # Preserve the editable plane and source transform without reading or
        # uploading another full volume for 2D and clip-only sections.
        placeholder = bool(defer_volume)
        layer = viewer.add_image(
            np.zeros((1, 1, 1), dtype=np.uint8) if placeholder else data,
            name=registry.unique_name(self.layer_name),
            visible=not placeholder,
            affine=affine,
            opacity=source.opacity,
            contrast_limits=tuple(source.contrast_limits),
            units=tuple(source.units[-3:]),
            depiction="plane",
            rendering=PROJECTION_TO_RENDERING.get(self.projection, "mip"),
        )
        registry.mark_managed(layer, self.id, "section_3d")
        layer._napari_animation_data_binding = None if placeholder else binding
        layer._napari_animation_placeholder = placeholder
        layer.metadata["napari_animation.source"] = source.name
        layer.metadata["napari_animation.source_level"] = self.source_level
        try:
            layer.colormap = source.colormap.name
            layer.contrast_limits = list(source.contrast_limits)
            layer.gamma = source.gamma
        except (AttributeError, ValueError, KeyError):
            pass
        self._bind_layer_events(layer, viewer)
        return layer

    @staticmethod
    def _bind_source_events(viewer, source):
        """Invalidate cached projections and bindings on source replacement.

        ``set_data`` also covers an in-place edit followed by ``refresh``.
        It may invalidate more often than strictly necessary, but never lets
        a previous projected array survive an explicit source refresh.
        """
        if getattr(source, "_napari_animation_source_listener", None) is not None:
            return
        source_id = LayerRegistry(viewer).source_id(source)
        source._napari_animation_source_revision = 0
        source._napari_animation_last_dims_point = tuple(viewer.dims.point)

        def changed(event):
            current_point = tuple(viewer.dims.point)
            previous_point = source._napari_animation_last_dims_point
            source._napari_animation_last_dims_point = current_point
            renderer = getattr(viewer, "_napari_animation_renderer", None)
            if event.type == "set_data" and (
                current_point != previous_point
                or (renderer is not None and renderer.applying)
            ):
                # Napari emits set_data when dims moves. That is a view
                # change, not a source edit, and must not force rebinding.
                return
            source._napari_animation_source_revision += 1
            from .processing import service_for_viewer

            service_for_viewer(viewer).invalidate_source(source_id)
            for layer in viewer.layers:
                binding = getattr(layer, "_napari_animation_data_binding", None)
                if binding is not None and layer.metadata.get(
                    "napari_animation.source"
                ) == source.name:
                    layer._napari_animation_data_binding = None

        connected = []
        for event_name in ("data", "set_data"):
            emitter = getattr(source.events, event_name, None)
            if emitter is not None:
                emitter.connect(changed)
                connected.append(emitter)
        source._napari_animation_source_listener = (changed, connected)

    def _bind_layer_events(self, layer, viewer) -> None:
        """Listen for native napari edits to the managed plane layer.

        The callback is deliberately attached to the layer rather than to a
        Qt widget.  A threedee manipulator, a script, or napari's own layer
        controls therefore update the editable scene object before the next
        capture.  The ``_applying`` guard prevents our own assignments from
        being mistaken for external edits.
        """
        binding = getattr(layer, "_napari_animation_plane_listener", None)
        if binding is not None and binding[0] is self:
            return
        if binding is not None:
            # ViewerState.apply may have installed a short-lived object while
            # rendering an interpolated frame. Replace that callback when the
            # editable live object takes ownership again.
            _, _, old_emitters = binding
            for emitter in old_emitters:
                try:
                    emitter.disconnect(binding[1])
                except (TypeError, ValueError):
                    pass

        def changed(_event):
            renderer = getattr(viewer, "_napari_animation_renderer", None)
            owner = getattr(self, "_composite_owner", None)
            if getattr(self, "_applying", False) or (
                renderer is not None and renderer.applying
            ) or (
                owner is not None and getattr(owner, "_composite_applying", False)
            ):
                return
            # The layer emits setup events while it is first being created;
            # there is no authored geometry to compare until ``apply`` has
            # written one.  Ignoring that initial burst prevents napari's
            # default plane thickness from overwriting the object's value.
            if getattr(self, "_applied_plane", None) is None:
                return
            try:
                self.sync_from_viewer(viewer)
                if owner is not None and owner.channels is not None:
                    owner.position = self.position
                    owner.normal = self.normal
                    owner.thickness = self.thickness
                    if owner.enabled:
                        owner.channels[self.source_id]["enabled"] = self.enabled
                    owner._capture_channel_appearance(viewer, self)
                    owner.apply(viewer)
            except (AttributeError, TypeError, ValueError):
                # A layer can emit a final event while it is being removed;
                # never let that transient state break the viewer.
                return

        connected = []
        for event_name in (
            "plane", "visible", "rendering", "colormap", "contrast_limits",
            "gamma", "opacity",
        ):
            emitter = getattr(layer.events, event_name, None)
            if emitter is not None:
                emitter.connect(changed)
                connected.append(emitter)
        layer._napari_animation_plane_listener = (self, changed, connected)

    def remove_layer(self, viewer: "napari.viewer.Viewer") -> None:
        """Remove this slice's backing layer, if it exists."""
        if self.channels is not None:
            for source in viewer.layers:
                bindings = getattr(
                    source, "_napari_animation_composite_appearance", None
                )
                if bindings is None:
                    continue
                binding = bindings.pop(self.id, None)
                if binding is not None:
                    for emitter in binding[2]:
                        emitter.disconnect(binding[1])
            for child in getattr(self, "_channel_children", {}).values():
                child.remove_layer(viewer)
            # Playback can have created adapters on another transient object.
            prefix = f"{self.id}::channel::"
            for layer in tuple(viewer.layers):
                owner = layer.metadata.get(OBJECT_ID_METADATA, "")
                if owner == self.id or owner.startswith(prefix):
                    viewer.layers.remove(layer)
            return
        service = getattr(viewer, "_napari_animation_projection_service", None)
        if service is not None:
            service.cancel_section(self.id)
        registry = LayerRegistry(viewer)
        for role in ("section_3d", "section_2d"):
            layer = registry.find_managed(self.id, role)
            if layer is None:
                continue
            # A scene object must never delete a user-owned source layer whose
            # name happens to match the object name.
            if layer.metadata.get(OBJECT_ID_METADATA) == self.id:
                binding = getattr(
                    layer, "_napari_animation_plane_listener", None
                )
                if binding is not None:
                    _, callback, emitters = binding
                    for emitter in emitters:
                        try:
                            emitter.disconnect(callback)
                        except (TypeError, ValueError):
                            pass
                    try:
                        delattr(layer, "_napari_animation_plane_listener")
                    except AttributeError:
                        pass
                viewer.layers.remove(layer)

    @timed_render("ortho.apply")
    def apply(
        self,
        viewer: "napari.viewer.Viewer",
        active_section_id: Optional[str] = None,
    ) -> None:
        """Push this slice's geometry onto its backing layer."""
        if self.channels is not None:
            children = self._channel_sections(viewer)
            active_ids = {child.id for child in children}
            prefix = f"{self.id}::channel::"
            self._composite_applying = True
            # Retire callbacks before hiding outputs. They may belong to a
            # different, transient snapshot with the same section ID.
            for layer in viewer.layers:
                bindings = getattr(layer, "_napari_animation_composite_appearance", {})
                binding = bindings.pop(self.id, None)
                if binding is not None:
                    for emitter in binding[2]:
                        emitter.disconnect(binding[1])
                owner = layer.metadata.get(OBJECT_ID_METADATA, "")
                if owner == self.id or owner.startswith(prefix):
                    binding = getattr(layer, "_napari_animation_plane_listener", None)
                    if binding is not None:
                        for emitter in binding[2]:
                            emitter.disconnect(binding[1])
                        delattr(layer, "_napari_animation_plane_listener")
                if (owner == self.id or owner.startswith(prefix)) and owner not in active_ids:
                    layer.visible = False
            in_2d = viewer.dims.ndisplay < 3
            is_active = active_section_id in (None, self.id)
            service = None
            if in_2d and is_active and self.enabled:
                from .processing import service_for_viewer

                service = service_for_viewer(viewer)
                service.begin_batch()
                self._composite_batch = {
                    "expected": active_ids, "results": {}, "errors": {},
                }
            try:
                for child in children:
                    child._composite_parent = self if service is not None else None
                    child.apply(viewer, active_section_id=(
                        child.id if is_active else active_section_id
                    ))
                    managed = LayerRegistry(viewer).find_managed(
                        child.id, "section_3d"
                    )
                    if managed is not None and child.render_mode == "plane":
                        managed.metadata["napari_animation.composite"] = self.id
                        managed.blending = "additive"
                        source = child.source_layer(viewer)
                        if source is not None:
                            self._bind_composite_appearance_events(viewer, source)
                            self._apply_channel_appearance(
                                managed, source, self.channels[child.source_id]
                            )
            finally:
                self._composite_applying = False
                if service is not None:
                    service.end_batch()
            return
        in_3d = getattr(viewer.dims, "ndisplay", 3) >= 3
        if in_3d or not self.enabled or (
            active_section_id is not None and self.id != active_section_id
        ):
            service = getattr(
                viewer, "_napari_animation_projection_service", None
            )
            if service is not None:
                service.cancel_section(self.id)
        if self.render_mode != "plane" and in_3d:
            # in clip mode the slab is contributed to the compositor instead;
            # hide any backing layer left over from a mode switch
            for role in ("section_3d", "section_2d"):
                managed = LayerRegistry(viewer).find_managed(self.id, role)
                if managed is not None:
                    managed.visible = False
            return

        layer = self.ensure_layer(viewer, for_2d=not in_3d)
        if layer is None:
            return

        # In 2D the slab layer would draw the same slice as its source, on top
        # of it. Plane depiction only means something in a 3D view.  Use the
        # managed projection path in 2D, keeping section appearance separate
        # from the source volume.
        projection_layer = LayerRegistry(viewer).find_managed(self.id, "section_2d")
        if projection_layer is not None and (
            in_3d or not self.enabled
            or (active_section_id is not None and self.id != active_section_id)
        ):
            projection_layer.visible = False
        wanted_visible = bool(self.enabled) and in_3d
        if layer.visible != wanted_visible:
            layer.visible = wanted_visible
        if not self.enabled:
            return

        self._applying = True
        try:
            if layer.depiction != "plane":
                layer.depiction = "plane"
            rendering = PROJECTION_TO_RENDERING.get(
                self.projection, "mip"
            )
            if layer.rendering != rendering:
                layer.rendering = rendering
            plane = {
                "position": tuple(
                    world_to_data_position(layer, self.position)[-3:].tolist()
                ),
                "normal": tuple(
                    world_to_data_normal(layer, self.normal)[-3:].tolist()
                ),
                "thickness": max(1e-6, self.data_thickness(layer)),
            }
            # Keep the native napari plane populated even while it is hidden
            # in 2D. Editors and external manipulators can then inspect or
            # modify the same geometry before the next 3D transition.
            current_plane = layer.plane
            if (
                not np.allclose(current_plane.position, plane["position"])
                or not np.allclose(current_plane.normal, plane["normal"])
                or not np.isclose(current_plane.thickness, plane["thickness"])
            ):
                layer.plane = plane
            self._applied_plane = plane
            self._applied_visible = layer.visible
            self._applied_rendering = layer.rendering
        finally:
            self._applying = False

        if not in_3d:
            if active_section_id is not None and self.id != active_section_id:
                # Keep the managed layer and native plane available for a
                # later 3D transition, but only one section owns global 2D
                # dims margins at a time.
                layer.visible = False
                return
            if not self.supports_2d(viewer):
                raise ValueError(
                    "Oblique or transformed ortho sections require a 3D view"
                )
            self._apply_2d(viewer, layer)
            return

    @timed_render("ortho.plan_2d")
    def _apply_2d(self, viewer, backing_layer) -> None:
        """Request a 2D projection, refining a multiscale preview on idle."""
        source = self.source_layer(viewer)
        if source is None or getattr(source, "ndim", 0) < 3:
            return
        from .processing import service_for_viewer

        service = service_for_viewer(viewer)
        revision = service.begin_section(self.id)
        world_axis = int(np.argmax(np.abs(np.asarray(self.normal, float))))
        global_axis = viewer.dims.ndim - 3 + world_axis
        viewer.dims.set_point(global_axis, float(self.position[world_axis]))
        exact = self._plan_2d(viewer, source)
        output = LayerRegistry(viewer).find_managed(self.id, "section_2d")
        if exact is None:
            service.cancel_section(self.id)
            if output is not None:
                output.visible = False
            parent = getattr(self, "_composite_parent", None)
            if parent is not None:
                parent._stage_channel_projection(viewer, self, None)
            return

        # A projection is a singleton slab positioned at the last completed
        # plane. Keep its pixels as a preview while reading the next plane,
        # and move that singleton with the slider so napari cannot slice it
        # away. Never show an old orientation/source as the new section.
        if output is not None and output.visible:
            binding = getattr(output, "_napari_animation_projection_binding", None)
            if (
                getattr(output, "_napari_animation_projection_axis", None) != world_axis
                or binding is None or binding[0] != exact[0][0]
            ):
                output.visible = False
            else:
                affine = output.affine.affine_matrix.copy()
                affine[world_axis, 3] = float(self.position[world_axis])
                if not np.array_equal(output.affine.affine_matrix, affine):
                    output.affine = affine

        multiscale = bool(getattr(source, "multiscale", False))
        finest = (
            self.source_level if self.source_level >= 0
            else len(source.data) + self.source_level
        ) if multiscale else 0
        coarse = len(source.data) - 1 if multiscale else 0
        progressive = (
            getattr(self, "_composite_parent", None) is None
            and
            service.asynchronous and multiscale and finest < coarse
            and not getattr(viewer, "_napari_animation_export_exact", False)
            and service.cache.get(exact[0]) is None
        )
        if not progressive:
            self._submit_2d_plan(
                viewer, backing_layer, source, service, exact, revision
            )
            return

        preview = self._plan_2d(viewer, source, level_override=coarse)
        if preview is not None:
            self._submit_2d_plan(
                viewer, backing_layer, source, service, preview, revision
            )

        from qtpy.QtCore import QTimer

        def refine():
            if not service.section_is_current(self.id, revision):
                return
            if self.source_layer(viewer) is not source:
                return
            if viewer.dims.ndisplay >= 3:
                return
            current = self._plan_2d(viewer, source)
            if current is not None:
                self._submit_2d_plan(
                    viewer, backing_layer, source, service, current, revision
                )

        QTimer.singleShot(300, refine)

    def _plan_2d(self, viewer, source, level_override=None):
        """Freeze geometry and array selection without materializing data."""
        data, factors, leading = self._level_volume(
            viewer, source, level_override=level_override
        )
        data_normal = np.asarray(world_to_data_normal(source, self.normal), float)[-3:]
        axis = int(np.argmax(np.abs(data_normal)))
        source_axis = source.ndim - 3 + axis
        centre = (
            float(world_to_data_position(source, self.position)[source_axis])
            / factors[axis]
        )
        thickness = float(self.data_thickness(source)) / factors[axis]
        interval = self._sample_interval(centre, thickness, int(data.shape[axis]))
        if interval is None:
            return None
        start, stop = interval
        affine = self._spatial_affine(
            source, factors, leading, level_override=level_override
        )
        level = self.source_level if level_override is None else level_override
        base = (
            source.data[level]
            if getattr(source, "multiscale", False) else source.data
        )
        key = (
            LayerRegistry(viewer).source_id(source),
            getattr(source, "_napari_animation_source_revision", 0),
            id(base), level, leading, axis, start, stop,
            self.projection,
        )
        return key, data, axis, start, stop, affine, centre

    def _submit_2d_plan(self, viewer, backing_layer, source, service, plan, revision):
        key, data, axis, start, stop, affine, centre = plan

        def publish(projected):
            if not service.section_is_current(self.id, revision):
                return
            if self.source_layer(viewer) is not source:
                return
            if viewer.dims.ndisplay >= 3:
                return
            renderer = getattr(viewer, "_napari_animation_renderer", None)
            parent = getattr(self, "_composite_parent", None)
            if renderer is not None and renderer.active_section_id not in (
                None, self.id, parent.id if parent is not None else self.id
            ):
                return
            # The output is displayed at the authored plane, not the centre
            # of the clipped interval, or napari may slice it away.
            _, output_data, output_affine = self._world_aligned_projection(
                projected, data.shape, axis, affine, centre
            )
            if parent is None:
                self._publish_2d(
                    viewer, backing_layer, output_data, output_affine, key
                )
            else:
                parent._stage_channel_projection(
                    viewer, self,
                    (backing_layer, output_data, output_affine, key),
                )

        parent = getattr(self, "_composite_parent", None)

        def failed(error):
            if parent is not None and service.section_is_current(self.id, revision):
                parent._stage_channel_projection(viewer, self, error)

        service.request(
            self.id, key, data, axis, start, stop, self.projection, publish,
            on_error=failed if parent is not None else None,
        )

    @timed_render("ortho.publish_2d")
    def _publish_2d(self, viewer, backing_layer, projected, affine, key):
        """Install a completed projection on the GUI thread."""
        registry = LayerRegistry(viewer)
        output = registry.find_managed(self.id, "section_2d")
        if output is None:
            output = viewer.add_image(
                projected,
                name=registry.unique_name(f"{self.name} [2D]"),
                affine=affine,
                units=tuple(backing_layer.units[-3:]),
                opacity=backing_layer.opacity,
                contrast_limits=tuple(backing_layer.contrast_limits),
            )
            registry.mark_managed(output, self.id, "section_2d")
        else:
            spatial_units = tuple(backing_layer.units[-3:])
            if tuple(output.units) != spatial_units:
                output.units = spatial_units
            if getattr(output, "_napari_animation_projection_binding", None) != key:
                output.data = projected
            if not np.array_equal(output.affine.affine_matrix, affine):
                output.affine = affine
            preferred = f"{self.name} [2D]"
            if output.name != preferred:
                output.name = registry.unique_name(preferred, exclude=output)
        output._napari_animation_projection_binding = key
        output._napari_animation_projection_axis = int(
            np.argmax(np.abs(np.asarray(self.normal, float)))
        )
        try:
            if output.colormap.name != backing_layer.colormap.name:
                output.colormap = backing_layer.colormap
            if not np.array_equal(
                output.contrast_limits, backing_layer.contrast_limits
            ):
                output.contrast_limits = list(backing_layer.contrast_limits)
            if output.gamma != backing_layer.gamma:
                output.gamma = backing_layer.gamma
        except (AttributeError, ValueError, KeyError):
            pass
        if output.opacity != backing_layer.opacity:
            output.opacity = backing_layer.opacity
        if not output.visible:
            output.visible = True
        world_axis = int(np.argmax(np.abs(np.asarray(self.normal, float))))
        global_axis = viewer.dims.ndim - 3 + world_axis
        viewer.dims.set_point(global_axis, float(self.position[world_axis]))

    def _stage_channel_projection(self, viewer, child, result):
        """Publish a coherent 2D composite after every channel is ready."""
        batch = getattr(self, "_composite_batch", None)
        if batch is None or child.id not in batch["expected"]:
            return
        if isinstance(result, BaseException):
            batch["errors"][child.id] = result
        batch["results"][child.id] = (child, result)
        if batch["results"].keys() != batch["expected"]:
            return
        if batch["errors"]:
            return
        registry = LayerRegistry(viewer)
        for item, payload in batch["results"].values():
            if payload is None:
                output = registry.find_managed(item.id, "section_2d")
                if output is not None:
                    output.visible = False
                continue
            backing, pixels, affine, key = payload
            item._publish_2d(viewer, backing, pixels, affine, key)
            output = registry.find_managed(item.id, "section_2d")
            if output is not None:
                output.metadata["napari_animation.composite"] = self.id
                output.blending = "additive"
                source = item.source_layer(viewer)
                if source is not None:
                    self._apply_channel_appearance(
                        output, source, self.channels[item.source_id]
                    )

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
        if self.channels is not None:
            for child in self._channel_sections(viewer):
                previous = (child.position, child.normal, child.thickness)
                child.sync_from_viewer(viewer)
                settings = self.channels[child.source_id]
                if self.enabled and child.enabled != bool(settings.get("enabled", False)):
                    settings["enabled"] = child.enabled
                if child.enabled:
                    self._capture_channel_appearance(viewer, child)
                if (child.position, child.normal, child.thickness) != previous:
                    self.position, self.normal, self.thickness = (
                        child.position, child.normal, child.thickness
                    )
                    break
            return
        if self.render_mode != "plane":
            return
        layer = LayerRegistry(viewer).find_managed(self.id, "section_3d")
        if layer is None:
            return
        if getattr(self, "_applied_plane", None) is None:
            return
        plane = getattr(layer, "plane", None)
        if plane is None:
            return

        applied = self._applied_plane
        if (
            applied is not None
            and np.allclose(plane.position, applied["position"])
            and np.allclose(plane.normal, applied["normal"])
            and np.isclose(plane.thickness, applied["thickness"])
        ):
            # the layer still holds what we put there; this object's own
            # parameters are the newer truth
            geometry_changed = False
        else:
            geometry_changed = True

        from .base import data_to_world_normal

        if geometry_changed:
            world_position = layer.data_to_world(plane.position)
            world_normal = data_to_world_normal(layer, plane.normal)
            self.position = tuple(float(v) for v in world_position[-3:])
            self.normal = tuple(float(v) for v in world_normal[-3:])
            self.thickness = self.world_thickness(
                layer, plane.normal, plane.thickness
            )
            self._applied_plane = {
                "position": tuple(float(v) for v in plane.position),
                "normal": tuple(float(v) for v in plane.normal),
                "thickness": float(plane.thickness),
            }

        # A hidden backing layer is an authored off switch only in 3D.  In 2D
        # the renderer hides plane depictions as a view override, so do not
        # turn that temporary suppression into a keyframed disable.
        if getattr(viewer.dims, "ndisplay", 3) >= 3:
            applied_visible = getattr(self, "_applied_visible", None)
            if applied_visible is not None and layer.visible != applied_visible:
                self.enabled = bool(layer.visible)

        applied_rendering = getattr(self, "_applied_rendering", None)
        if applied_rendering is not None and layer.rendering != applied_rendering:
            reverse = {value: key for key, value in PROJECTION_TO_RENDERING.items()}
            if layer.rendering in reverse:
                self.projection = reverse[layer.rendering]

    # ------------------------------------------------------ clipping planes

    def clipping_planes(
        self, viewer: "napari.viewer.Viewer"
    ) -> Iterator[Tuple[str, List[dict]]]:
        if self.channels is not None:
            for child in self._channel_sections(viewer):
                yield from child.clipping_planes(viewer)
            return
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
                "position": tuple((position - half * normal)[-3:].tolist()),
                "normal": tuple(normal[-3:].tolist()),
                "enabled": True,
            },
            {
                "position": tuple((position + half * normal)[-3:].tolist()),
                "normal": tuple((-normal)[-3:].tolist()),
                "enabled": True,
            },
        ]
