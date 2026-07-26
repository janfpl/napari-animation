from dataclasses import dataclass, field
from typing import Optional

import napari
import numpy as np

from .ortho_slicer import OrthoSlicer


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
    state.pop("metadata", None)

    try:
        full_state = layer._get_state()
    except Exception:  # noqa: BLE001 - never fail a capture over one layer
        return state

    for key in ANIMATABLE_LAYER_PROPERTIES:
        if key in full_state:
            state[key] = full_state[key]

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
        A map of layer.name -> _base_state for each layer in the viewer
        (excluding metadata).
    ortho : dict, optional
        Parameters of the :class:`~napari_animation.ortho_slicer.OrthoSlicer`
        optical section (``None`` when the slicer is inactive).  Stored so the
        optical section can be animated alongside the rest of the viewer state.
    """

    camera: dict
    dims: dict
    layers: dict
    ortho: Optional[dict] = field(default=None)

    @classmethod
    def from_viewer(cls, viewer: napari.viewer.Viewer, ortho: dict = None):
        """Create a ViewerState from a viewer instance.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            A napari viewer.
        ortho : dict, optional
            Ortho-slicer parameters to record alongside the viewer state.
        """
        layers = {layer.name: _layer_state(layer) for layer in viewer.layers}
        return cls(
            camera=viewer.camera.dict(),
            dims=viewer.dims.dict(),
            layers=layers,
            ortho=ortho,
        )

    def apply(self, viewer: napari.viewer.Viewer):
        """Update `viewer` to match this ViewerState.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            A napari viewer. (viewer state will be directly modified)
        """

        # Dims must be applied before the camera. Changing ``ndisplay``
        # makes napari recompute the camera itself -- restoring a per-mode
        # cache or calling ``fit_to_view()`` -- so a camera applied first
        # would be silently discarded by a 2D/3D switch.
        _resilient_update(viewer.dims, self.dims)

        for layer_name, layer_state in self.layers.items():
            # Layers may be missing if the same data has not been loaded (for
            # instance when resuming a saved animation); skip them gracefully.
            if layer_name not in viewer.layers:
                continue
            layer = viewer.layers[layer_name]
            for key, value in layer_state.items():
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
        if self.ortho is not None:
            OrthoSlicer.apply_state(viewer, self.ortho)

        # Camera last, so the keyframe's camera wins over anything an
        # ndisplay switch recomputed above.
        _resilient_update(viewer.camera, self.camera)

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
        self.apply(viewer)
        return viewer.screenshot(canvas_only=canvas_only)

    def __eq__(self, other):
        if isinstance(other, ViewerState):
            return (
                self.camera == other.camera
                and self.dims == other.dims
                and self.layers == other.layers
                and self.ortho == other.ortho
            )
        else:
            return False
