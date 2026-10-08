"""Save and restore an animation's keyframes to/from disk.

This lets you close napari and later resume editing an animation.  The saved
file is JSON and contains, for every keyframe, the captured viewer state
(camera, dims, per-layer display settings, optical section), the interpolation
settings (steps / easing) and a thumbnail. Version-2 source records also carry
a stable layer ID, while the current viewer-state compatibility mapping still
uses the display name.

The pixel data of the layers is **not** stored.  Instead the source file path
of each layer is recorded, and on load those layers are re-opened (matched to
keyframe state by stable source ID, with the captured name as a v1 fallback).
Layers that cannot be reloaded (e.g. created programmatically, or whose files
have moved) are skipped; their keyframe settings are simply ignored when
applied.
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from uuid import uuid4
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import imageio
import numpy as np

from .easing import Easing
from .key_frame import KeyFrame
from .layer_registry import (
    LayerRegistry, MANAGED_ROLE_METADATA, OBJECT_ID_METADATA, TRANSIENT_ROLE,
)
from .viewer_state import ViewerState

if TYPE_CHECKING:
    from .animation import Animation

#: Current on-disk schema.  Version 1 files remain readable through the
#: migration path in :func:`load_animation`.
FILE_FORMAT_VERSION = "3"
SUPPORTED_FILE_FORMAT_VERSIONS = {"1", "2", FILE_FORMAT_VERSION}


def _to_builtin(obj):
    """Recursively convert an object into JSON-serialisable builtins.

    Handles the various non-JSON types that show up in napari viewer state:
    numpy arrays/scalars, enums, pint units, namedtuples, etc.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Enum):
        return _to_builtin(obj.value)
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    # pint units, colormaps and any other exotic object -> their string form,
    # which napari's pydantic models can coerce back on apply.
    return str(obj)


def _encode_thumbnail(thumbnail) -> str:
    """Encode an RGBA thumbnail array as a base64 PNG string."""
    if thumbnail is None:
        return ""
    buffer = io.BytesIO()
    imageio.imwrite(
        buffer, np.asarray(thumbnail, dtype=np.uint8), format="png"
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_thumbnail(data: str):
    """Decode a base64 PNG thumbnail string back into an array."""
    if not data:
        return np.zeros((30, 30, 4), dtype=np.uint8)
    raw = base64.b64decode(data.encode("ascii"))
    return np.asarray(imageio.imread(io.BytesIO(raw), format="png"))


def _layer_records(viewer) -> list:
    """Collect (name, path, type) records to allow reloading layer data."""
    records = []
    LayerRegistry(viewer).ensure_unique_ids()
    for layer in viewer.layers:
        if (
            getattr(layer, "metadata", {}).get(MANAGED_ROLE_METADATA) == TRANSIENT_ROLE
            or layer.metadata.get("napari_animation.composite")
        ):
            continue
        source_path = getattr(getattr(layer, "source", None), "path", None)
        try:
            layer_id = LayerRegistry.ensure_id(layer)
        except (TypeError, AttributeError):
            layer_id = None
        records.append(
            {
                "name": layer.name,
                "id": layer_id,
                "source_id": layer_id,
                "path": str(source_path) if source_path else None,
                "type": layer._type_string,
            }
        )
        if layer.metadata.get("napari_animation.luxendo"):
            records[-1]["reader"] = "luxendo"
    return records


def animation_to_dict(animation: "Animation") -> dict:
    """Serialise an :class:`~napari_animation.Animation` to a plain dict."""
    key_frames = []
    for kf in animation.key_frames:
        state = kf.viewer_state
        key_frames.append(
            {
                "name": kf.name,
                "steps": int(kf.steps),
                "ease": kf.ease.name,
                "thumbnail": _encode_thumbnail(kf.thumbnail),
                "viewer_state": {
                    "camera": _to_builtin(state.camera),
                    "dims": _to_builtin(state.dims),
                    "layers": _to_builtin(state.layers),
                    "ortho": _to_builtin(state.ortho),
                    "scene": _to_builtin(state.scene),
                    "view": _to_builtin(state.view),
                },
            }
        )
    return {
        "version": FILE_FORMAT_VERSION,
        "layers": _layer_records(animation.viewer),
        "key_frames": key_frames,
    }


def save_animation(animation: "Animation", path) -> Path:
    """Save an animation's keyframes to ``path`` (a ``.json`` file)."""
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".json")
    data = animation_to_dict(animation)
    # Write beside the destination then atomically replace it.  A failed disk
    # write must never leave a half-valid project that cannot be loaded.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def _bind_layers(animation: "Animation", layer_records, open_missing=False) -> None:
    """Bind saved IDs to live layers and optionally open missing sources."""
    viewer = animation.viewer
    registry = LayerRegistry(viewer)
    registry.ensure_unique_ids()
    seen = set()
    for record in layer_records:
        name = record.get("name")
        source_id = record.get("source_id") or record.get("id")
        if not source_id:
            raise ValueError(f"Layer record {name!r} has no source ID")
        source_id = str(source_id)
        if source_id in seen:
            raise ValueError(f"Duplicate layer source ID {source_id!r}")
        seen.add(source_id)
        # The renderer recreates these by object ID. Never attach a managed
        # identity to a user layer merely because its display name matches.
        if source_id.startswith("napari-animation:"):
            continue
        existing = registry.find(source_id)
        if existing is None and name is not None and name in viewer.layers:
            candidate = viewer.layers[name]
            if not candidate.metadata.get(OBJECT_ID_METADATA):
                existing = candidate
        if existing is not None:
            registry.attach_id(existing, source_id)
            continue
        if not open_missing:
            continue
        path = record.get("path")
        if not path or not Path(path).exists():
            continue
        try:
            if record.get("reader") == "luxendo":
                from .luxendo import add_luxendo_channel, read_luxendo_channel

                new_layers = [
                    add_luxendo_channel(viewer, read_luxendo_channel(path))
                ]
            else:
                new_layers = viewer.open(path)
        except Exception:  # noqa: BLE001 - reloading is best-effort
            continue
        # If a single layer was added, give it the name keyframe state expects.
        if len(new_layers) == 1 and new_layers[0].name != name:
            new_layers[0].name = name
        if len(new_layers) == 1:
            registry.attach_id(new_layers[0], source_id)


def _normalize_source_ids(data: dict) -> None:
    """Convert every frame to the project-wide stable source IDs."""
    by_name = {}
    seen_ids = set()
    for record in data.get("layers", []):
        source_id = record.get("source_id") or record.get("id")
        if source_id:
            source_id = str(source_id)
            record["source_id"] = source_id
            record["id"] = source_id
            if source_id in seen_ids:
                raise ValueError(f"Duplicate layer source ID {source_id!r}")
            seen_ids.add(source_id)
            name = record.get("name")
            if name in by_name and by_name[name] != source_id:
                raise ValueError(
                    f"Duplicate layer name {name!r} has conflicting IDs"
                )
            by_name[record.get("name")] = source_id

    for frame in data.get("key_frames", []):
        state = frame.get("viewer_state", {})
        normalized = {}
        for key, layer_state in (state.get("layers") or {}).items():
            layer_state = dict(layer_state)
            source_id = layer_state.get("source_id")
            if not source_id:
                source_id = by_name.get(layer_state.get("name")) or str(key)
            source_id = str(source_id)
            if source_id in normalized:
                raise ValueError(f"Duplicate layer source ID {source_id!r} in keyframe")
            layer_state["source_id"] = source_id
            normalized[source_id] = layer_state
        state["layers"] = normalized
        for params in (state.get("scene") or {}).values():
            if params.get("kind") == "ortho_slice":
                bound = by_name.get(params.get("source"))
                if not params.get("source_id") and bound:
                    params["source_id"] = bound


def _migrate_v1(data: dict) -> dict:
    """Normalize a version-1 payload without changing its semantics."""
    migrated = dict(data)
    migrated["version"] = "2"
    migrated.setdefault("layers", [])
    migrated.setdefault("key_frames", [])
    for record in migrated["layers"]:
        record.setdefault("id", record.get("source_id"))
        record.setdefault("source_id", record.get("id"))
        if not record.get("source_id"):
            source_id = uuid4().hex
            record["id"] = source_id
            record["source_id"] = source_id
    for key_frame in migrated["key_frames"]:
        viewer_state = key_frame.setdefault("viewer_state", {})
        # Scene was introduced after the first schema; ``None`` is the
        # explicit disabled value used by the current model.
        viewer_state.setdefault("scene", None)
        viewer_state.setdefault("view", None)
    return migrated


def _migrate_v2_sections(data: dict) -> None:
    """Turn one-source sections into one-binding composites in saved state."""
    records = data.get("layers", [])
    first_source = next(
        (record for record in records
         if not str(record.get("source_id", "")).startswith("napari-animation:")),
        None,
    )
    managed_ids = set()
    for frame in data.get("key_frames", []):
        state = frame.get("viewer_state", {})
        layers = state.get("layers") or {}
        for params in (state.get("scene") or {}).values():
            if params.get("kind") != "ortho_slice" or params.get("channels") is not None:
                continue
            section_id = str(params.get("id", ""))
            managed_id = LayerRegistry.managed_id(section_id, "section_3d")
            managed_ids.add(managed_id)
            source_id = params.get("source_id")
            if not source_id and params.get("source") is None and first_source:
                source_id = first_source.get("source_id")
                params["source_id"] = source_id
                params["source"] = first_source.get("name")
            if not source_id and params.get("source"):
                source_id = f"unresolved:{section_id}"
                params["source_id"] = source_id
            if source_id:
                binding = {
                    "source_id": str(source_id),
                    "source_name": params.get("source"),
                    "enabled": True,
                    "source_level": params.get("source_level", 0),
                }
                old_output = layers.get(managed_id, {})
                appearance = {
                    key: old_output[key]
                    for key in ("colormap", "contrast_limits", "gamma", "opacity")
                    if key in old_output
                }
                if appearance:
                    binding["appearance_mode"] = "custom"
                    binding["appearance"] = appearance
                params["channels"] = {str(source_id): binding}
            else:
                params["channels"] = {}
            layers.pop(managed_id, None)
        state["layers"] = layers
    data["layers"] = [
        record for record in records
        if record.get("source_id") not in managed_ids
    ]
    data["version"] = FILE_FORMAT_VERSION


def _validate_keyframe_scene(animation: "Animation", frame: KeyFrame) -> None:
    dims = frame.viewer_state.dims
    ndim = int(dims.get("ndim", animation.viewer.dims.ndim))
    order = tuple(dims.get("order", tuple(range(ndim))))
    if len(order) != ndim or set(order) != set(range(ndim)):
        raise ValueError(f"Invalid dims.order {order!r} for {ndim} dimensions")
    ndisplay = int(dims.get("ndisplay", 2))
    if ndisplay not in {2, 3} or ndisplay > ndim:
        raise ValueError(f"Invalid dims.ndisplay {ndisplay!r} for {ndim} dimensions")
    scene = frame.viewer_state.scene
    active = None
    if isinstance(frame.viewer_state.view, dict):
        active = frame.viewer_state.view.get("active_section_id")
    # SceneRenderer performs strict kind, identity, geometry, active-selection
    # and 2D capability validation without mutating viewer state.
    animation.renderer.preflight(
        scene,
        ndisplay=ndisplay,
        active_section_id=active,
        layer_states=frame.viewer_state.layers,
    )


def _key_frame_from_dict(kf_data: dict) -> KeyFrame:
    vs_data = kf_data["viewer_state"]
    steps = int(kf_data.get("steps", 15))
    if steps < 1:
        raise ValueError("steps must be at least 1")
    ease_name = str(kf_data.get("ease", "LINEAR")).upper()
    if ease_name not in Easing.__members__:
        raise ValueError(f"unknown easing {ease_name!r}")
    viewer_state = ViewerState(
        camera=vs_data["camera"],
        dims=vs_data["dims"],
        layers=vs_data["layers"],
        ortho=vs_data.get("ortho"),
        # absent in files written before scene objects existed
        scene=vs_data.get("scene"),
        view=vs_data.get("view"),
    )
    return KeyFrame(
        viewer_state=viewer_state,
        thumbnail=_decode_thumbnail(kf_data.get("thumbnail", "")),
        steps=steps,
        ease=Easing[ease_name],
        name=kf_data.get("name", "KeyFrame"),
    )


def load_animation(
    animation: "Animation", path, reload_layers: bool = True
) -> None:
    """Load keyframes from ``path`` into ``animation`` (replacing existing).

    Parameters
    ----------
    animation : napari_animation.Animation
        The animation to populate.  Its current keyframes are cleared first.
    path : str or pathlib.Path
        Path to a file previously written by :func:`save_animation`.
    reload_layers : bool
        If ``True`` (default), attempt to re-open the layer data files recorded
        in the saved file (for layers not already present in the viewer).
    """
    path = Path(path)
    # Saved as UTF-8; don't depend on the platform's locale encoding.
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    version = str(data.get("version", "0"))
    if version not in SUPPORTED_FILE_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported animation file version {version!r} "
            f"(expected one of {sorted(SUPPORTED_FILE_FORMAT_VERSIONS)!r})"
        )
    if version == "1":
        data = _migrate_v1(data)

    try:
        _normalize_source_ids(data)
        if version in {"1", "2"}:
            _migrate_v2_sections(data)
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"Invalid animation layer data: {err}") from err

    # Parse the complete payload before touching the live animation.  A
    # malformed keyframe should leave the user's current edit intact.
    try:
        parsed_keyframes = [
            _key_frame_from_dict(kf_data) for kf_data in data["key_frames"]
        ]
        for key_frame in parsed_keyframes:
            _validate_keyframe_scene(animation, key_frame)
    except (KeyError, TypeError, ValueError, IndexError) as err:
        raise ValueError(f"Invalid animation keyframe data: {err}") from err

    old_keyframes = list(animation.key_frames)
    old_active = animation.key_frames.selection.active
    old_selection = set(animation.key_frames.selection)
    old_objects = list(animation.scene)
    old_ortho = animation.ortho_slicer
    old_frame = getattr(animation, "_current_frame", 0)
    old_sequence_index = animation._frames._current_index
    viewer = animation.viewer
    old_layers = [(layer, dict(layer.metadata), layer.name) for layer in viewer.layers]
    ortho, scene, view = animation._capture_state()
    working_state = ViewerState.from_viewer(viewer, ortho=ortho, scene=scene, view=view)
    try:
        with animation.renderer.applying_state():
            _bind_layers(animation, data.get("layers", []), open_missing=reload_layers)
            # Sources opened or rebound above must also pass validation, for
            # every frame, before replacing the user's timeline.
            for key_frame in parsed_keyframes:
                _validate_keyframe_scene(animation, key_frame)
        animation.key_frames.clear()
        animation.key_frames.extend(parsed_keyframes)
        if len(animation.key_frames):
            animation.set_to_keyframe(0)
    except Exception:
        with animation.renderer.applying_state(), animation.key_frames.selection.events.active.blocker():
            animation.key_frames.clear()
            animation.key_frames.extend(old_keyframes)
            animation.key_frames.selection.update(old_selection)
            animation.key_frames.selection.active = old_active
            for layer in list(viewer.layers):
                if not any(layer is old for old, _, _ in old_layers):
                    viewer.layers.remove(layer)
            for layer, metadata, name in old_layers:
                layer.metadata.clear()
                layer.metadata.update(metadata)
                layer.name = name
            animation.scene.clear()
            animation.scene.extend(old_objects)
            animation.scene.restore_snapshot(viewer, working_state.scene)
            animation.ortho_slicer = old_ortho
            animation.active_section_id = view.get("active_section_id")
            animation.renderer.active_section_id = animation.active_section_id
            working_state.apply(viewer)
            animation.apply_working_state()
            viewer.camera.update(working_state.camera)
            animation._current_frame = old_frame
            animation._frames._current_index = old_sequence_index
        raise
