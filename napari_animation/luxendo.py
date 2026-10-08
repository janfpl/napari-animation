"""Import processed Luxendo channels using their embedded physical metadata.

Image arrays remain lazy. Dataset references own a shared read-only file lease,
so derived arrays remain usable after their original layer is removed. The last
dataset reference releases the file; closing the animation dock does not close
data still displayed by napari.
"""

from __future__ import annotations

import json
import os
import re
import weakref
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import h5py
import numpy as np

LUXENDO_METADATA = "napari_animation.luxendo"
_PYRAMID = re.compile(r"^Data_(\d+)_(\d+)_(\d+)$")
_COLORS = ("magenta", "green", "cyan", "yellow", "red", "blue")


class ImportCancelled(Exception):
    pass


class _FileLease:
    def __init__(self, file):
        self.finalizer = weakref.finalize(self, file.close)


@dataclass
class LuxendoChannel:
    path: Path
    name: str
    scale: tuple
    arrays: list
    metadata: dict
    contrast_limits: tuple

    def layer_data(self, color="gray"):
        multiscale = len(self.arrays) > 1
        return (
            self.arrays if multiscale else self.arrays[0],
            dict(
                name=self.name,
                scale=self.scale,
                units=("um",) * 3,
                axis_labels=("Z", "Y", "X"),
                rgb=False,
                multiscale=multiscale,
                blending="additive",
                colormap=color,
                contrast_limits=self.contrast_limits,
                metadata={LUXENDO_METADATA: self.metadata},
            ),
        )


@dataclass
class LuxendoImport:
    channels: list
    skipped: list


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise ImportCancelled()


def _decode_metadata(dataset):
    raw = dataset[()]
    if isinstance(raw, np.ndarray):
        if raw.size == 1:
            raw = raw.item()
        elif raw.dtype.kind in "Su":
            raw = raw.tobytes()
        elif raw.dtype.kind == "U":
            raw = "".join(raw.flat)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("metadata must contain a JSON string")
    metadata = json.loads(raw.rstrip("\x00"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must contain a JSON object")
    processing = metadata.get("processingInformation", {})
    if not isinstance(processing, dict):
        raise ValueError("invalid processingInformation metadata")
    voxel = processing.get("voxel_size_um", {})
    try:
        scale = tuple(
            float(voxel[key]) for key in ("depth", "height", "width")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("missing Z/Y/X voxel_size_um metadata") from error
    if not all(np.isfinite(value) and value > 0 for value in scale):
        raise ValueError(
            "voxel sizes must be finite, positive micrometer values"
        )
    return processing, scale


def read_luxendo_channel(path, cancel=None):
    """Read metadata and a bounded contrast sample; keep all levels lazy."""
    _check_cancel(cancel)
    path = Path(path).expanduser().resolve()
    file = h5py.File(path, "r")
    try:
        if "metadata" not in file:
            raise ValueError("missing embedded Luxendo metadata")
        processing, scale = _decode_metadata(file["metadata"])
        native = file.get("Data")
        if not isinstance(native, h5py.Dataset) or native.ndim != 3:
            raise ValueError("expected a 3D Data dataset")
        if native.dtype.kind not in "buif" or min(native.shape) < 1:
            raise ValueError("Data must be a non-empty numeric image")
        levels = [("Data", (1, 1, 1))]
        candidates = []
        for name in file:
            match = _PYRAMID.fullmatch(name)
            if match:
                factors = tuple(int(value) for value in match.groups())
                if min(factors) >= 1 and max(factors) > 1:
                    candidates.append((name, factors))
        previous = native.shape
        for name, factors in sorted(
            candidates, key=lambda entry: np.prod(entry[1])
        ):
            dataset = file[name]
            if (
                not isinstance(dataset, h5py.Dataset)
                or dataset.ndim != 3
                or dataset.dtype != native.dtype
                or min(dataset.shape) < 1
                or not all(a <= b for a, b in zip(dataset.shape, previous))
                or dataset.shape == previous
            ):
                raise ValueError(f"invalid pyramid level {name}")
            levels.append((name, factors))
            previous = dataset.shape

        # A small contiguous ROI avoids a full-volume contrast scan and
        # avoids touching every native HDF5 chunk with strided sampling.
        coarse = file[levels[-1][0]]
        roi = (slice(coarse.shape[0] // 2, coarse.shape[0] // 2 + 1),) + tuple(
            slice(max(0, size // 2 - 64), min(size, size // 2 + 64))
            for size in coarse.shape[1:]
        )
        sample = np.asarray(coarse[roi])
        finite = sample[np.isfinite(sample)]
        low, high = (
            (float(finite.min()), float(finite.max()))
            if finite.size
            else (0.0, 1.0)
        )
        if low == high:
            low, high = (0.0, high) if high > 0 else (low, low + 1.0)
        _check_cancel(cancel)
        lease = _FileLease(file)
        arrays = []
        for name, _factors in levels:
            dataset = file[name]
            dataset._napari_animation_file_lease = lease
            chunks = dataset.chunks or tuple(
                min(size, limit)
                for size, limit in zip(dataset.shape, (16, 256, 256))
            )
            arrays.append(
                da.from_array(
                    dataset,
                    chunks=chunks,
                    asarray=False,
                    lock=True,
                    meta=np.empty((0, 0, 0), dtype=dataset.dtype),
                )
            )
        channel_id = processing.get("channel_id", processing.get("channel"))
        description = str(processing.get("channel_description") or "").strip()
        name = (
            f"ch{channel_id}"
            if channel_id is not None and str(channel_id) != ""
            else ""
        )
        name = " — ".join(value for value in (name, description) if value)
        metadata = {
            "path": str(path),
            "voxel_size_zyx_um": scale,
            "channel": channel_id,
            "channel_description": description,
            "time_point": processing.get("time_point"),
            "stack": processing.get("stack"),
            "pyramid_datasets": [name for name, _ in levels],
            "pyramid_factors_xyz": [factors for _, factors in levels],
        }
        return LuxendoChannel(
            path, name or path.name, scale, arrays, metadata, (low, high)
        )
    except BaseException:
        file.close()
        raise


def scan_luxendo_files(directory, recursive=True):
    """Find per-channel HDF5 files, excluding companion header files."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a folder: {directory}")
    files = []
    for root, _dirs, names in os.walk(directory, followlinks=False):
        for name in names:
            lower = name.lower()
            if (
                lower.endswith(".h5")
                and not lower.startswith("main")
                and not lower.endswith("_bdv.h5")
            ):
                path = Path(root) / name
                if not path.is_symlink():
                    files.append(path)
        if not recursive:
            break
    return sorted(files, key=lambda path: str(path).casefold())


def prepare_luxendo_folder(directory, recursive=True, cancel=None):
    """Prepare valid channels off the GUI thread and report rejected files."""
    channels, skipped = [], []
    for path in scan_luxendo_files(directory, recursive=recursive):
        _check_cancel(cancel)
        try:
            channels.append(read_luxendo_channel(path, cancel=cancel))
        except (OSError, ValueError, TypeError) as error:
            skipped.append(f"{path.name}: {error}")
    _check_cancel(cancel)
    if not channels:
        detail = (
            "\n".join(skipped) or "No per-channel .lux.h5 or .h5 files found."
        )
        raise ValueError(detail)
    from collections import Counter

    counts = Counter(channel.name for channel in channels)
    for channel in channels:
        if counts[channel.name] > 1:
            channel.name = f"{channel.name} ({channel.path.stem})"
    return LuxendoImport(channels, skipped)


def add_luxendo_channel(viewer, channel, index=0):
    """Install and auto-contrast one channel on the GUI thread."""
    from napari.layers._source import layer_source

    data, kwargs = channel.layer_data(_COLORS[index % len(_COLORS)])
    with layer_source(path=str(channel.path)):
        layer = viewer.add_image(data, **kwargs)
    try:
        # Match napari's Auto contrast action using the displayed image slice.
        layer.reset_contrast_limits()
    except Exception:
        viewer.layers.remove(layer)
        raise
    return layer


def loaded_luxendo_paths(viewer):
    return {
        os.path.normcase(str(Path(info["path"]).resolve()))
        for layer in viewer.layers
        if (info := layer.metadata.get(LUXENDO_METADATA)) and info.get("path")
    }


def import_luxendo_folder(viewer, directory, recursive=True):
    """Script API; the dock performs preparation in a background worker."""
    import warnings

    plan = prepare_luxendo_folder(directory, recursive=recursive)
    existing = loaded_luxendo_paths(viewer)
    added = []
    try:
        for channel in plan.channels:
            if os.path.normcase(str(channel.path)) not in existing:
                added.append(add_luxendo_channel(viewer, channel, len(added)))
                existing.add(os.path.normcase(str(channel.path)))
    except Exception:
        for layer in reversed(added):
            viewer.layers.remove(layer)
        raise
    if plan.skipped:
        warnings.warn(
            "Skipped Luxendo files:\n" + "\n".join(plan.skipped), stacklevel=2
        )
    return added
