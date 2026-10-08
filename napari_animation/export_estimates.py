"""Fast, explicitly approximate export sizes for the save dialog.

This is a planning heuristic, not a codec prediction or a size guarantee.
No frames are rendered or encoded. H.264 assumes 0.10 bits/pixel/frame at
quality 5, doubles per six CRF units, and reports a 0.5x–2x range. Other video
formats get a broader range; GIF and PNG use quality-independent ranges.
Real motion, noise, detail and container overhead can exceed these ranges.
"""

from dataclasses import dataclass
from math import ceil
from typing import Optional

from .animation import H264_SUFFIXES, VIDEO_SUFFIXES, _compat_output_params


@dataclass(frozen=True)
class ExportEstimate:
    resolution: tuple
    duration: float
    size_range: Optional[tuple]
    quality_applies: bool


def estimate_export(
    frame_size, frame_count, fps, scale, quality, suffix, video_block_size=16,
):
    """Estimate output using an input (width, height) in screenshot pixels."""
    suffix = suffix.lower()
    quality_applies = suffix in VIDEO_SUFFIXES
    duration = frame_count / fps
    # scipy.ndimage.zoom rounds the requested dimensions before resampling.
    width, height = (int(round(value * scale)) for value in frame_size)
    if width <= 0 or height <= 0:
        return ExportEstimate((width, height), duration, None, quality_applies)
    if quality_applies:
        width, height = (
            video_block_size * ceil(value / video_block_size)
            for value in (width, height)
        )
    pixels = width * height * frame_count
    if suffix in H264_SUFFIXES:
        crf = int(_compat_output_params(quality)[3])
        bits_per_pixel = 0.10 * 2 ** ((20 - crf) / 6)
        low, high = bits_per_pixel / 2, bits_per_pixel * 2
    elif quality_applies:
        bits_per_pixel = 0.10 * 2 ** ((quality - 5) / 3)
        low, high = bits_per_pixel / 4, bits_per_pixel * 8
    elif suffix == ".gif":
        low, high = 1, 8
    elif suffix in {"", ".png"}:
        low, high = 3, 32
    else:
        return ExportEstimate((width, height), duration, None, quality_applies)
    size_range = (pixels * low / 8, pixels * high / 8) if frame_count else None
    return ExportEstimate((width, height), duration, size_range, quality_applies)


def format_size_range(size_range):
    """Format decimal file-size units without suggesting false precision."""
    low, high = size_range
    for divisor, unit in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if high >= divisor:
            return f"{low / divisor:.1f}–{high / divisor:.1f} {unit}"
    return f"{low:.0f}–{high:.0f} bytes"
