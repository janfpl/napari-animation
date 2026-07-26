"""Save and restore an animation's keyframes to/from disk.

This lets you close napari and later resume editing an animation.  The saved
file is JSON and contains, for every keyframe, the captured viewer state
(camera, dims, per-layer display settings, optical section), the interpolation
settings (steps / easing) and a thumbnail.

The pixel data of the layers is **not** stored.  Instead the source file path
of each layer is recorded, and on load those layers are re-opened (matched to
keyframe state by layer name).  Layers that cannot be reloaded (e.g. created
programmatically, or whose files have moved) are skipped; their keyframe
settings are simply ignored when applied.
"""

from __future__ import annotations

import base64
import io
import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import imageio
import numpy as np

from .easing import Easing
from .key_frame import KeyFrame
from .viewer_state import ViewerState

if TYPE_CHECKING:
    from .animation import Animation

#: Bumped when the on-disk format changes in a backwards-incompatible way.
FILE_FORMAT_VERSION = "1"


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
    for layer in viewer.layers:
        source_path = getattr(getattr(layer, "source", None), "path", None)
        records.append(
            {
                "name": layer.name,
                "path": str(source_path) if source_path else None,
                "type": layer._type_string,
            }
        )
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
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _reload_layers(animation: "Animation", layer_records) -> None:
    """Re-open any saved layers that aren't already present in the viewer."""
    viewer = animation.viewer
    for record in layer_records:
        name = record.get("name")
        if name in viewer.layers:
            continue
        path = record.get("path")
        if not path or not Path(path).exists():
            continue
        try:
            new_layers = viewer.open(path)
        except Exception:  # noqa: BLE001 - reloading is best-effort
            continue
        # If a single layer was added, give it the name keyframe state expects.
        if len(new_layers) == 1 and new_layers[0].name != name:
            new_layers[0].name = name


def _key_frame_from_dict(kf_data: dict) -> KeyFrame:
    vs_data = kf_data["viewer_state"]
    viewer_state = ViewerState(
        camera=vs_data["camera"],
        dims=vs_data["dims"],
        layers=vs_data["layers"],
        ortho=vs_data.get("ortho"),
        # absent in files written before scene objects existed
        scene=vs_data.get("scene"),
    )
    return KeyFrame(
        viewer_state=viewer_state,
        thumbnail=_decode_thumbnail(kf_data.get("thumbnail", "")),
        steps=int(kf_data.get("steps", 15)),
        ease=Easing[kf_data.get("ease", "LINEAR")],
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
    with open(path) as f:
        data = json.load(f)

    version = str(data.get("version", "0"))
    if version != FILE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported animation file version {version!r} "
            f"(expected {FILE_FORMAT_VERSION!r})"
        )

    if reload_layers:
        _reload_layers(animation, data.get("layers", []))

    animation.key_frames.clear()
    for kf_data in data["key_frames"]:
        animation.key_frames.append(_key_frame_from_dict(kf_data))

    # select the first frame so the viewer reflects the loaded animation
    if len(animation.key_frames):
        animation.set_to_keyframe(0)
