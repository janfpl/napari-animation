from dataclasses import dataclass, field
from typing import Optional

import napari
import numpy as np

from .layer_registry import (
    MANAGED_ROLE_METADATA,
    TRANSIENT_ROLE,
    LayerRegistry,
)
from .ortho_slicer import OrthoSlicer
from .perf import render_phase, timed_render


class LayerStateMap(dict):
    """Canonical source-ID mapping with a legacy name lookup bridge.

    New snapshots iterate by stable source ID.  ``state.layers["img"]`` still
    resolves the entry by its captured display name so existing scripts and
    version-1 files keep working while projects migrate.
    """

    def _resolve_key(self, key):
        if dict.__contains__(self, key):
            return key
        for source_id, state in dict.items(self):
            if isinstance(state, dict) and state.get("name") == key:
                return source_id
        raise KeyError(key)

    def __getitem__(self, key):
        return dict.__getitem__(self, self._resolve_key(key))

    def __contains__(self, key):
        try:
            self._resolve_key(key)
            return True
        except KeyError:
            return False

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _resilient_update(model, state: dict):
    """``model.update(state)`` that tolerates non-restorable fields.

    When a viewer state is restored from disk, a few fields (e.g. pint ``units``
    on dims) deserialize to plain strings that napari's validators won't coerce
    back on a bulk update. Rather than failing the whole update, drop the
    offending keys and retry so everything else is still applied.
    """
    state = dict(state)
    while state:
        try:
            model.update(state)
            return
        except (
            Exception
        ) as err:  # noqa: BLE001 - tolerate any validation error
            errors = getattr(err, "errors", None)
            bad_keys = set()
            if callable(errors):
                for entry in errors():
                    loc = entry.get("loc")
                    if loc:
                        bad_keys.add(loc[0])
            bad_keys &= set(state)
            if not bad_keys:
                raise
            for key in bad_keys:
                state.pop(key, None)


#: Per-layer display properties captured in addition to the layer's
#: ``_get_base_state()``.
#:
#: ``_get_base_state()`` covers geometry and visibility but omits everything
#: that describes how a volume is *drawn*. Most importantly it omits ``plane``
#: (``position`` / ``normal`` / ``thickness``) and ``depiction``, which are
#: napari's native optical-section primitive -- without them an ortho slice
#: cannot be keyframed at all.
#:
#: ``data`` is deliberately excluded: :class:`FrameSequence` deep-copies the
#: captured state for every interpolated frame, so capturing the array would
#: copy the entire volume once per frame of the movie.
ANIMATABLE_LAYER_PROPERTIES = (
    "depiction",
    "plane",
    "rendering",
    "contrast_limits",
    "gamma",
    "iso_threshold",
    "attenuation",
    "interpolation2d",
    "interpolation3d",
)


def _layer_state(layer) -> dict:
    """Capture the animatable display state of a single layer."""
    state = layer._get_base_state()
    state["name"] = layer.name
    state["source_id"] = LayerRegistry.ensure_id(layer)
    renderer = getattr(
        getattr(layer, "_viewer", None), "_napari_animation_renderer", None
    )
    state.pop("metadata", None)
    if getattr(layer, "multiscale", False) and hasattr(
        layer, "locked_data_level"
    ):
        state["locked_data_level"] = layer.locked_data_level

    try:
        full_state = layer._get_state()
    except Exception:  # noqa: BLE001 - never fail a capture over one layer
        return state

    for key in ANIMATABLE_LAYER_PROPERTIES:
        if key in full_state:
            state[key] = full_state[key]

    # A 2D section temporarily suppresses its source.  Persist the authored
    # visibility rather than this renderer-owned view override.
    if renderer is not None:
        state["visible"] = renderer.authored_visible(
            layer, state.get("visible", True)
        )

    # Colormaps are captured by name rather than as the full definition: the
    # expanded form is a dict of colour arrays that neither interpolates
    # meaningfully nor survives a JSON round trip.
    name = getattr(getattr(layer, "colormap", None), "name", None)
    if name:
        state["colormap"] = name

    return state


def _differs(original, value) -> bool:
    """Whether ``value`` differs from ``original`` (assume it does on error).

    Layer state now includes structured values (a ``plane`` dict, a colormap
    name) that ``np.array_equal`` cannot always compare; treating those as
    changed is the safe direction, since it costs a redundant assignment
    rather than a silently skipped one.
    """
    try:
        return not np.array_equal(original, value)
    except Exception:  # noqa: BLE001
        return True


@dataclass(frozen=True)
class ViewerState:
    """The state of the viewer camera, dims, and layers.

    Parameters
    ----------
    camera : dict
        The state of the `napari.components.Camera` in the viewer.
    dims : dict
        The state of the `napari.components.Dims` in the viewer.
    layers : dict
        A map of stable source ID -> animatable layer state for each layer in
        the viewer (excluding metadata).  ``LayerStateMap`` retains a lookup
        bridge by the captured display name for older callers.
    ortho : dict, optional
        Parameters of the :class:`~napari_animation.ortho_slicer.OrthoSlicer`
        optical section (``None`` when the slicer is inactive).  Stored so the
        optical section can be animated alongside the rest of the viewer state.
    scene : dict, optional
        Parameters of every :class:`~napari_animation.scene.SceneObject`,
        keyed by object id, so clipping planes and optical sections can be
        reshaped and switched on or off independently at each keyframe.
        ``None`` when no scene objects exist.
    """

    camera: dict
    dims: dict
    layers: dict
    ortho: Optional[dict] = field(default=None)
    scene: Optional[dict] = field(default=None)
    view: Optional[dict] = field(default=None)

    def __post_init__(self):
        if not isinstance(self.layers, LayerStateMap):
            object.__setattr__(self, "layers", LayerStateMap(self.layers))

    @classmethod
    def from_viewer(
        cls,
        viewer: napari.viewer.Viewer,
        ortho: dict = None,
        scene: dict = None,
        view: dict = None,
    ):
        """Create a ViewerState from a viewer instance.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            A napari viewer.
        ortho : dict, optional
            Ortho-slicer parameters to record alongside the viewer state.
        scene : dict, optional
            Scene-object parameters to record alongside the viewer state.
        """
        layers = LayerStateMap()
        registry = LayerRegistry(viewer)
        registry.ensure_unique_ids()
        renderer = getattr(viewer, "_napari_animation_renderer", None)
        for layer in viewer.layers:
            role = getattr(layer, "metadata", {}).get(MANAGED_ROLE_METADATA)
            if role == TRANSIENT_ROLE or layer.metadata.get(
                "napari_animation.composite"
            ):
                continue
            layer_state = _layer_state(layer)
            if renderer is not None:
                layer_state["visible"] = renderer.authored_visible(
                    layer, layer_state.get("visible", True)
                )
            layers[layer_state["source_id"]] = layer_state
        if scene:
            # The scene compositor is the sole writer of clipping planes;
            # capturing them per layer as well would apply a stale set just
            # before the compositor recomputes them.
            for layer_state in layers.values():
                layer_state.pop("experimental_clipping_planes", None)
        return cls(
            camera=viewer.camera.dict(),
            dims=viewer.dims.dict(),
            layers=layers,
            ortho=ortho,
            scene=scene,
            view=view,
        )

    def apply(self, viewer: napari.viewer.Viewer):
        """Update `viewer` to match this ViewerState.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            A napari viewer. (viewer state will be directly modified)
        """

        from .scene import SceneRenderer

        renderer = SceneRenderer.for_viewer(viewer)
        with renderer.applying_state():
            self._apply_guarded(viewer, renderer)

    @timed_render("state.apply")
    def _apply_guarded(self, viewer, renderer):
        active_section_id = None
        if isinstance(self.view, dict):
            active_section_id = self.view.get("active_section_id")
        target_ndisplay = int(self.dims.get("ndisplay", viewer.dims.ndisplay))
        suppressed_source_ids = set()
        prepared_scene = None
        if self.scene is not None:
            prepared_scene = renderer.prepare_snapshot(
                self.scene,
                active_section_id=active_section_id,
                ndisplay=target_ndisplay,
                layer_states=self.layers,
            )
            _, active_obj = prepared_scene
            suppressed_source_ids = renderer._suppressed_source_ids(
                active_obj, ndisplay=target_ndisplay
            )
        renderer.restore_view_overrides(keep_ids=suppressed_source_ids)

        # Dims must be applied before the camera. Changing ``ndisplay``
        # makes napari recompute the camera itself -- restoring a per-mode
        # cache or calling ``fit_to_view()`` -- so a camera applied first
        # would be silently discarded by a 2D/3D switch.
        with render_phase("state.dims"):
            changed = {
                key: value
                for key, value in self.dims.items()
                if _differs(getattr(viewer.dims, key, None), value)
            }
            if changed:
                _resilient_update(viewer.dims, changed)

        with render_phase("state.layers"):
            registry = LayerRegistry(viewer)
            for source_id, layer_state in self.layers.items():
                # Layers may be missing if the same data has not been loaded (for
                # instance when resuming a saved animation); skip them gracefully.
                layer = registry.find(source_id, layer_state.get("name"))
                if layer is None and not layer_state.get("source_id"):
                    layer = registry.find(
                        name=layer_state.get("name", source_id)
                    )
                if layer is None:
                    continue
                for key, value in layer_state.items():
                    if key in {"name", "source_id"}:
                        continue
                    if key == "visible" and source_id in suppressed_source_ids:
                        # Preserve the authored visibility while the 2D section
                        # temporarily hides its source.  Repeated frame applies
                        # must not upload the full source slice twice by toggling
                        # visible True then False.
                        renderer._source_visibility[source_id] = bool(value)
                        continue
                    original_value = getattr(layer, key, None)
                    # Only set if value differs to avoid expensive redraws
                    if _differs(original_value, value):
                        try:
                            setattr(layer, key, value)
                        except Exception:  # noqa: BLE001
                            # A property may be unsettable for this layer's data
                            # (e.g. plane depiction on 2D data). Skip it rather
                            # than abandoning the rest of the frame.
                            pass

        # The optical section is recomputed from its parameters and the (now
        # applied) dims, so a sweeping/growing slab interpolates smoothly.
        # When a scene is present it owns every layer's clipping planes, so
        # the slicer contributes its slab to the composite instead of
        # assigning it -- otherwise the two would overwrite each other.
        has_scene = self.scene is not None
        if self.ortho is not None:
            OrthoSlicer.apply_state(
                viewer, self.ortho, assign_clipping_planes=not has_scene
            )
        if has_scene:
            renderer.apply_snapshot(
                self.scene,
                ortho=self.ortho,
                active_section_id=active_section_id,
                prepared=prepared_scene,
            )
        else:
            # A legacy snapshot without a scene must retire outputs from a
            # previously rendered scene without replacing its clipping state.
            renderer._reconcile([], None)

        # Camera last, so the keyframe's camera wins over anything an
        # ndisplay switch recomputed above.
        with render_phase("state.camera"):
            changed = {
                key: value
                for key, value in self.camera.items()
                if _differs(getattr(viewer.camera, key, None), value)
            }
            if changed:
                _resilient_update(viewer.camera, changed)

    def render(
        self, viewer: napari.viewer.Viewer, canvas_only=True
    ) -> np.ndarray:
        """Render this ViewerState to an image.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            A napari viewer to render screenshots from.
        canvas_only : bool, optional
            Whether to include only the canvas (and exclude the napari
            gui), by default True

        Returns
        -------
        np.ndarray
            An RGBA image of shape (h, w, 4).
        """
        from .scene.processing import exact_render

        with exact_render(viewer):
            self.apply(viewer)
            return viewer.screenshot(canvas_only=canvas_only)

    def __eq__(self, other):
        if isinstance(other, ViewerState):
            return (
                self.camera == other.camera
                and self.dims == other.dims
                and self.layers == other.layers
                and self.ortho == other.ortho
                and self.scene == other.scene
                and self.view == other.view
            )
        else:
            return False
