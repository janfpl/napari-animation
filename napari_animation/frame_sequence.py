from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from napari.utils.events import EmitterGroup

from .interpolation import Interpolation, InterpolationMap
from .utils import pairwise
from .viewer_state import ViewerState

#: Characters that make an interpolation-map key a glob pattern rather than a
#: literal state path.
_GLOB_CHARS = ("*", "?", "[")


def resolve_interpolation(
    interpolation_map: InterpolationMap, path: str
) -> Interpolation:
    """Return the interpolation to use for a dotted state ``path``.

    Exact keys take precedence. Otherwise keys containing glob characters are
    matched against the path, which lets a single rule cover every layer or
    scene object (``"layers.*.rendering"``) without naming them. When several
    patterns match, the most specific one wins -- specificity being the length
    of the pattern's leading literal (non-wildcard) text.
    """
    if path in interpolation_map:
        return interpolation_map[path]

    best = Interpolation.DEFAULT
    best_specificity = -1
    for pattern, interpolation in interpolation_map.items():
        if not any(char in pattern for char in _GLOB_CHARS):
            continue
        if not fnmatchcase(path, pattern):
            continue
        specificity = len(pattern.split("*")[0])
        if specificity > best_specificity:
            best, best_specificity = interpolation, specificity
    return best


if TYPE_CHECKING:
    import napari

    from .key_frame import KeyFrame, KeyFrameList


def _normalized_scene_pair(left, right):
    """Return scene dictionaries with the same stable object IDs.

    An object missing from one endpoint is represented by the last known
    geometry and ``enabled=False`` at that endpoint.  This makes topology a
    discrete visibility change instead of asking numeric interpolation to
    operate on ``None``.  The caller owns the returned copies.
    """
    if not left and not right:
        return None, None
    left = left or {}
    right = right or {}
    left_out, right_out = {}, {}
    for object_id in sorted(set(left) | set(right)):
        if object_id in left:
            left_params = deepcopy(left[object_id])
        else:
            left_params = deepcopy(right[object_id])
            left_params["enabled"] = False
        if object_id in right:
            right_params = deepcopy(right[object_id])
        else:
            right_params = deepcopy(left[object_id])
            right_params["enabled"] = False

        left_kind = left_params.get("kind")
        right_kind = right_params.get("kind")
        if left_kind != right_kind:
            raise ValueError(
                f"Scene object {object_id!r} changes kind "
                f"from {left_kind!r} to {right_kind!r}"
            )
        if left_kind == "ortho_slice" and (
            left_params.get("channels") is not None
            or right_params.get("channels") is not None
        ):

            def bindings(params):
                existing = params.get("channels")
                if existing is not None:
                    return deepcopy(existing)
                source_id = params.get("source_id") or params.get("source")
                if not source_id:
                    return {}
                return {
                    source_id: {
                        "source_id": source_id,
                        "source_name": params.get("source"),
                        "enabled": True,
                        "source_level": params.get("source_level", 0),
                    }
                }

            left_channels = bindings(left_params)
            right_channels = bindings(right_params)
            for source_id in set(left_channels) | set(right_channels):
                if source_id not in left_channels:
                    left_channels[source_id] = deepcopy(
                        right_channels[source_id]
                    )
                    left_channels[source_id]["enabled"] = False
                if source_id not in right_channels:
                    right_channels[source_id] = deepcopy(
                        left_channels[source_id]
                    )
                    right_channels[source_id]["enabled"] = False
            left_params["channels"] = left_channels
            right_params["channels"] = right_channels
        left_out[object_id] = left_params
        right_out[object_id] = right_params
    return left_out, right_out


def _normalized_layer_pair(left, right):
    """Align layer maps by stable source ID before walking their leaves."""

    def canonical(mapping):
        result = {}
        for key, value in (mapping or {}).items():
            value = deepcopy(value)
            identity = str(value.get("source_id") or key)
            value["source_id"] = identity
            if identity in result:
                raise ValueError(f"Duplicate layer source ID {identity!r}")
            result[identity] = value
        return result

    left, right = canonical(left), canonical(right)
    left_out, right_out = {}, {}
    for identity in sorted(set(left) | set(right)):
        if identity in left:
            a = deepcopy(left[identity])
        else:
            a = deepcopy(right[identity])
            a["visible"] = False
        if identity in right:
            b = deepcopy(right[identity])
        else:
            b = deepcopy(left[identity])
            b["visible"] = False
        left_out[identity], right_out[identity] = a, b
    return left_out, right_out


class FrameSequence(Sequence[ViewerState]):
    """Final sequence of of rendered animation frames, based on keyframes.

    This object acts like an immutable sequence of frames, interpolated from
    a sequence of (mutable) KeyFrames.  It can be indexed at any (valid) frame
    in the animation, and will inteprolate (and cache) viewer state on demand.

    If the KeyFrameList changes in any way, the cache is cleared.

    Parameters
    ----------
    key_frames : KeyFrameList
        A KeyFrameList from which to render the final frame sequence.
    """

    def __init__(self, key_frames: KeyFrameList) -> None:
        super().__init__()
        self._key_frames = key_frames
        key_frames.events.inserted.connect(self._rebuild_frame_index)
        key_frames.events.removed.connect(self._rebuild_frame_index)
        key_frames.events.changed.connect(self._rebuild_frame_index)
        key_frames.events.reordered.connect(self._rebuild_frame_index)

        self.__current_index = 0
        self.events = EmitterGroup(
            source=self, n_frames=None, _current_index=None
        )

        self.state_interpolation_map: InterpolationMap = {
            "camera.angles": Interpolation.SLERP,
            "camera.zoom": Interpolation.LOG,
            # 2D/3D is a discrete mode, not a number to blend: hold the
            # starting mode for the whole transition and switch on arrival.
            "dims.ndisplay": Interpolation.STEP_END,
            "dims.order": Interpolation.STEP_END,
            # plane normals rotate along the shortest arc rather than
            # collapsing toward the origin halfway through a large turn.
            "layers.*.plane.normal": Interpolation.SLERP_VECTOR,
            # discrete display state, likewise switched on arrival.
            "layers.*.depiction": Interpolation.STEP_END,
            "layers.*.rendering": Interpolation.STEP_END,
            "layers.*.locked_data_level": Interpolation.STEP_END,
            "layers.*.colormap": Interpolation.STEP_END,
            "layers.*.visible": Interpolation.STEP_END,
            "layers.*.name": Interpolation.STEP_END,
            "layers.*.source_id": Interpolation.STEP_END,
            "layers.*.projection_mode": Interpolation.STEP_END,
            "layers.*.interpolation2d": Interpolation.STEP_END,
            "layers.*.interpolation3d": Interpolation.STEP_END,
            # scene objects: geometry blends, identity and on/off do not.
            "scene.*.normal": Interpolation.SLERP_VECTOR,
            "scene.*.enabled": Interpolation.STEP_END,
            "scene.*.kind": Interpolation.STEP_END,
            "scene.*.name": Interpolation.STEP_END,
            "scene.*.id": Interpolation.STEP_END,
            "scene.*.targets": Interpolation.STEP_END,
            "scene.*.source": Interpolation.STEP_END,
            "scene.*.source_id": Interpolation.STEP_END,
            "scene.*.source_level": Interpolation.STEP_END,
            "scene.*.channels.*.source_id": Interpolation.STEP_END,
            "scene.*.channels.*.source_name": Interpolation.STEP_END,
            "scene.*.channels.*.enabled": Interpolation.STEP_END,
            "scene.*.channels.*.source_level": Interpolation.STEP_END,
            "scene.*.channels.*.appearance_mode": Interpolation.STEP_END,
            "scene.*.channels.*.appearance.colormap": Interpolation.STEP_END,
            "scene.*.projection": Interpolation.STEP_END,
            "scene.*.render_mode": Interpolation.STEP_END,
            "scene.*.sampling_convention": Interpolation.STEP_END,
            "view.active_section_id": Interpolation.STEP_END,
        }

        # cache of interpolated viewer states
        # A long export walks frames once.  Retaining every interpolated
        # snapshot would grow memory in proportion to movie duration.
        self._cache: OrderedDict[int, ViewerState] = OrderedDict()
        self._cache_limit = 128

        # map of frame number -> (kf0, kf1, fraction)
        self._frame_index: Dict[int, Tuple[KeyFrame, KeyFrame, float]] = {}
        self._raw_fractions: Dict[int, float] = {}
        self._rebuild_frame_index()

    def _rebuild_frame_index(self, event=None):
        """Create a map of frame number -> (kf0, kf1, fraction)"""
        self._frame_index.clear()
        self._raw_fractions.clear()
        self._cache.clear()

        n_keyframes = len(self._key_frames)

        if n_keyframes == 0:
            self.events.n_frames(value=len(self))
            return
        elif n_keyframes == 1:
            f = 0
            kf1 = self._key_frames[0]
        else:
            f = 0
            for kf0, kf1 in pairwise(self._key_frames):
                for s in range(kf1.steps):
                    raw = s / kf1.steps
                    fraction = kf1.ease(raw)
                    self._frame_index[f] = (kf0, kf1, fraction)
                    self._raw_fractions[f] = raw
                    f += 1

        self._frame_index[f] = (kf1, kf1, 0)
        self._raw_fractions[f] = 1.0
        self.events.n_frames(value=len(self))

    def __len__(self) -> int:
        """The total frame count of the animation"""
        return len(self._frame_index)

    def __getitem__(self, key: int) -> ViewerState:
        """Get the interpolated state at frame `key` in the animation."""
        if key < 0:
            key += len(self)
        if key not in self._cache:
            try:
                kf0, kf1, frac = self._frame_index[key]
            except KeyError:
                raise IndexError(
                    f"Frame index ({key}) out of range ({len(self)} frames)"
                )
            raw = self._raw_fractions.get(key, frac)
            if kf0 is kf1 or raw == 0:
                self._cache[key] = kf0.viewer_state
            else:
                self._cache[key] = self._interpolate_state(
                    kf0.viewer_state,
                    kf1.viewer_state,
                    frac,
                    self.state_interpolation_map,
                    raw_fraction=raw,
                )
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)

        return self._cache[key]

    def _interpolate_state(
        self,
        from_state: ViewerState,
        to_state: ViewerState,
        fraction: float,
        state_interpolation_map: Optional[InterpolationMap] = None,
        raw_fraction: Optional[float] = None,
    ):
        """Interpolate a state between two states

        Parameters
        ----------
        from_state : ViewerState
            Description of initial viewer state.
        to_state : ViewerState
            Description of final viewer state.
        fraction : float
            Interpolation fraction, must be between `0` and `1`.
            A value of `0` will return the initial state. A
            value of `1` will return the final state.
        state_interpolation_map : dict
            Dictionary relating state attributes to interpolation functions.

        Returns
        -------
        state : dict
            Description of viewer state.
        """
        from .utils import keys_to_list, nested_get, nested_set

        interp_map = state_interpolation_map or self.state_interpolation_map

        state = {}
        sep = "."

        from_state = asdict(from_state)
        to_state = asdict(to_state)

        # Scene objects are keyed by stable IDs.  Normalize object topology
        # before walking leaves so a section added or removed at a keyframe
        # fades by its explicit ``enabled`` flag while retaining valid
        # geometry for interpolation.  Passing ``None`` into the vector
        # interpolators is both ambiguous and a runtime error.
        from_state["scene"], to_state["scene"] = _normalized_scene_pair(
            from_state.get("scene"), to_state.get("scene")
        )
        from_state["layers"], to_state["layers"] = _normalized_layer_pair(
            from_state.get("layers"), to_state.get("layers")
        )

        # A 2D section cannot rotate through oblique intermediate normals.
        # Keep its orientation with the starting dims order until arrival;
        # source changes also switch their geometry atomically.
        if (raw_fraction if raw_fraction is not None else fraction) < 1:
            source_ids = set()
            for object_id, left in (from_state.get("scene") or {}).items():
                right = to_state["scene"][object_id]
                if left.get("kind") != "ortho_slice":
                    continue
                source_ids.add(left.get("source_id") or left.get("source"))
                source_ids.update(
                    source_id
                    for source_id, channel in (
                        left.get("channels") or {}
                    ).items()
                    if channel.get("enabled", False)
                )
                source_changed = any(
                    left.get(key) != right.get(key)
                    for key in ("source_id", "source_level")
                )
                if source_changed:
                    for key in ("normal", "position", "thickness"):
                        right[key] = deepcopy(left[key])
                elif from_state["dims"].get("ndisplay", 2) == 2:
                    right["normal"] = deepcopy(left["normal"])
            if from_state["dims"].get("ndisplay", 2) == 2:
                for source_id in source_ids:
                    left = from_state["layers"].get(source_id)
                    right = to_state["layers"].get(source_id)
                    if left is None or right is None:
                        continue
                    # A permutation or signed-axis switch cannot be blended
                    # through a rotated/singular source while still in 2D.
                    for key in ("rotate", "shear", "affine"):
                        if key in left:
                            right[key] = deepcopy(left[key])
                    if "scale" in left and "scale" in right:
                        if np.any(
                            np.asarray(left["scale"])
                            * np.asarray(right["scale"])
                            <= 0
                        ):
                            right["scale"] = deepcopy(left["scale"])

        visited = set()
        for keys in keys_to_list(from_state):
            # LUT dictionaries are one discrete colormap value, not numeric
            # leaves. The other endpoint may even be a registered name.
            if (
                len(keys) >= 6
                and keys[0] == "scene"
                and keys[2] == "channels"
                and keys[4:6] == ["appearance", "colormap"]
            ):
                keys = keys[:6]
            path = tuple(keys)
            if path in visited:
                continue
            visited.add(path)
            v0 = nested_get(from_state, keys)
            v1 = nested_get(to_state, keys)

            interp_func = resolve_interpolation(interp_map, sep.join(keys))

            interpolation_fraction = (
                raw_fraction
                if raw_fraction is not None
                and interp_func
                in {Interpolation.STEP_START, Interpolation.STEP_END}
                else fraction
            )
            nested_set(
                state, keys, interp_func(v0, v1, interpolation_fraction)
            )

        # Appearance mode and its payload form one discrete transition.
        # Walking leaves cannot interpolate a custom dictionary into None.
        boundary_fraction = (
            raw_fraction if raw_fraction is not None else fraction
        )
        for object_id, left in (from_state.get("scene") or {}).items():
            right = to_state["scene"][object_id]
            for source_id, channel in (left.get("channels") or {}).items():
                target = right["channels"][source_id]
                if channel.get("appearance_mode", "source") != target.get(
                    "appearance_mode", "source"
                ):
                    selected = channel if boundary_fraction < 1 else target
                    output = state["scene"][object_id]["channels"][source_id]
                    output["appearance_mode"] = selected.get(
                        "appearance_mode", "source"
                    )
                    output["appearance"] = deepcopy(selected.get("appearance"))

        return ViewerState(**state)

    def iter_frames(
        self,
        viewer: napari.viewer.Viewer,
        canvas_only: bool = True,
        scale_factor: float = None,
    ) -> Iterator[np.ndarray]:
        """Iterate over interpolated viewer states, and yield rendered frames."""
        for i, state in enumerate(self):
            frame = state.render(viewer, canvas_only=canvas_only)
            if scale_factor not in (None, 1):
                from scipy import ndimage as ndi

                frame = ndi.zoom(frame, (scale_factor, scale_factor, 1))
                frame = frame.astype(np.uint8)
            yield frame

    def set_movie_frame_index(self, viewer: napari.viewer.Viewer, index: int):
        self[index].apply(viewer)
        self._current_index = index

    @property
    def _current_index(self):
        return self.__current_index

    @_current_index.setter
    def _current_index(self, frame_index):
        if frame_index != self.__current_index:
            self.__current_index = frame_index
            self.events._current_index(value=frame_index)
