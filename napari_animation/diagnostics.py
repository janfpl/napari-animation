"""Diagnostics for "the render finished but the video is wrong".

The render pipeline can report complete success and still produce a file that
will not play, or that plays but shows nothing. The failure is almost always
one of a small number of things, and none of them are visible in an ordinary
success log:

* **Frames changed size mid-render.** Video codecs need every frame to be the
  same size. If the napari window is resized while rendering, later frames
  differ from the first and the encoder produces garbage.
* **Frames are blank.** A screenshot that comes back a single flat colour
  usually means the canvas had nothing to read -- the file encodes fine and
  plays as a solid rectangle.
* **Frames are all identical.** The animation state never actually changed,
  so the movie is technically valid but looks frozen.
* **The container was never finalised.** The file exists but is unreadable, or
  is a few hundred bytes of header.

:class:`RenderDiagnostics` samples each frame cheaply, then reads the finished
file back off disk and compares what came out against what went in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("napari_animation")

#: A written file smaller than this is almost certainly a bare container
#: header with no usable video in it.
SUSPICIOUS_BYTES = 2048

#: Frames are subsampled to roughly this many pixels per axis before their
#: statistics are taken, so diagnostics cost the same on a 4K canvas as on a
#: small one.
SAMPLE_AXIS = 64


def _sample(frame: np.ndarray) -> np.ndarray:
    """Return a cheap, evenly spaced subsample of a frame."""
    row_step = max(1, frame.shape[0] // SAMPLE_AXIS)
    col_step = max(1, frame.shape[1] // SAMPLE_AXIS)
    return frame[::row_step, ::col_step]


class RenderDiagnostics:
    """Collects evidence about what the render actually produced.

    Parameters
    ----------
    enabled : bool
        When ``False`` every method is a no-op, so this can be left wired in
        without cost.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.frame_count = 0
        self.first_shape: Optional[tuple] = None
        self.first_dtype = None
        self.shape_changes: List[tuple] = []
        self.blank_frames: List[int] = []
        self.duplicate_frames = 0
        self.value_range = [None, None]
        self._previous_sample: Optional[np.ndarray] = None
        self.notes: List[str] = []

    # ------------------------------------------------------------- capture

    def describe_environment(self) -> None:
        """Log the encoder versions and executable actually being used."""
        if not self.enabled:
            return
        try:
            import imageio

            self.notes.append(f"imageio {imageio.__version__}")
        except Exception:  # noqa: BLE001
            pass
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.notes.append(
                f"imageio-ffmpeg {imageio_ffmpeg.__version__} ({exe})"
            )
        except Exception as err:  # noqa: BLE001
            self.notes.append(f"imageio-ffmpeg unavailable: {err}")

    def record_frame(self, index: int, frame: np.ndarray) -> None:
        """Sample one rendered frame. Must never raise."""
        if not self.enabled:
            return
        try:
            self._record_frame(index, frame)
        except (
            Exception
        ) as err:  # noqa: BLE001 - diagnostics never break a render
            logger.debug("frame diagnostics failed at %d: %s", index, err)

    def _record_frame(self, index: int, frame: np.ndarray) -> None:
        self.frame_count += 1

        if self.first_shape is None:
            self.first_shape = tuple(frame.shape)
            self.first_dtype = frame.dtype
        elif tuple(frame.shape) != self.first_shape:
            # only record the first few; a resize mid-render affects every
            # subsequent frame and the list would otherwise be enormous
            if len(self.shape_changes) < 5:
                self.shape_changes.append((index, tuple(frame.shape)))

        sample = _sample(frame)
        # ignore alpha: napari screenshots are RGBA and a fully opaque alpha
        # channel would mask an otherwise blank image
        if sample.ndim == 3 and sample.shape[2] == 4:
            sample = sample[..., :3]

        low, high = float(sample.min()), float(sample.max())
        if self.value_range[0] is None:
            self.value_range = [low, high]
        else:
            self.value_range[0] = min(self.value_range[0], low)
            self.value_range[1] = max(self.value_range[1], high)

        if low == high:
            if len(self.blank_frames) < 10:
                self.blank_frames.append(index)

        if self._previous_sample is not None:
            if self._previous_sample.shape == sample.shape and np.array_equal(
                self._previous_sample, sample
            ):
                self.duplicate_frames += 1
        self._previous_sample = sample

    # -------------------------------------------------------- verification

    def verify_output(self, path, expected_frames: int) -> None:
        """Read the finished file back and compare it with what was written."""
        if not self.enabled:
            return
        try:
            self._verify_output(path, expected_frames)
        except Exception as err:  # noqa: BLE001
            self.notes.append(f"could not verify output file: {err}")

    def _verify_output(self, path, expected_frames: int) -> None:
        path = Path(path)
        if not path.exists():
            self.notes.append(f"OUTPUT MISSING: {path} was not created")
            return

        size = path.stat().st_size
        self.notes.append(f"output file: {size:,} bytes")
        if size < SUSPICIOUS_BYTES:
            self.notes.append(
                f"WARNING: {size} bytes is too small to contain "
                f"{expected_frames} frames -- the encoder probably wrote only "
                "a container header"
            )

        import imageio.v2 as iio

        try:
            reader = iio.get_reader(str(path))
        except Exception as err:  # noqa: BLE001
            self.notes.append(
                f"WARNING: the written file could not be reopened ({err}); "
                "it is not a readable video"
            )
            return

        try:
            shapes = set()
            read_back = 0
            for read_frame in reader.iter_data():
                read_back += 1
                shapes.add(tuple(read_frame.shape))
        finally:
            reader.close()

        self.notes.append(
            f"read back {read_back} frame(s) of {sorted(shapes)}"
        )
        if read_back == 0:
            self.notes.append("WARNING: the file contains no decodable frames")
        elif read_back < expected_frames:
            self.notes.append(
                f"WARNING: wrote {expected_frames} frames but only "
                f"{read_back} decoded -- the container may be truncated"
            )

        encoded = next(iter(shapes), None)
        if encoded and self.first_shape:
            canvas = self.first_shape[:2]
            if tuple(encoded[:2]) != tuple(canvas):
                self.notes.append(
                    f"note: canvas {canvas[1]}x{canvas[0]} was encoded as "
                    f"{encoded[1]}x{encoded[0]}; ffmpeg pads frames up to a "
                    "multiple of 16, which adds a blank strip. Size the napari "
                    "window so both dimensions are multiples of 16 to avoid it"
                )

    # -------------------------------------------------------------- report

    def report(self) -> str:
        """Return a human-readable summary of everything observed."""
        if not self.enabled:
            return ""

        lines = ["napari-animation render diagnostics:"]

        if self.first_shape is not None:
            channels = self.first_shape[2] if len(self.first_shape) > 2 else 1
            lines.append(
                f"  frames rendered : {self.frame_count} @ "
                f"{self.first_shape[1]}x{self.first_shape[0]} "
                f"({channels} channels, {self.first_dtype})"
            )
            lines.append(
                f"  pixel range     : {self.value_range[0]:g} .. "
                f"{self.value_range[1]:g}"
            )

        if self.shape_changes:
            lines.append(
                "  FRAME SIZE CHANGED MID-RENDER -- this corrupts the video."
            )
            for index, shape in self.shape_changes:
                lines.append(
                    f"      frame {index}: {shape[1]}x{shape[0]} "
                    f"(expected {self.first_shape[1]}x{self.first_shape[0]})"
                )
            lines.append(
                "      Do not resize or interact with the napari window while "
                "rendering."
            )

        if self.blank_frames:
            shown = ", ".join(str(i) for i in self.blank_frames)
            lines.append(
                f"  BLANK FRAMES    : {shown}"
                f"{' ...' if len(self.blank_frames) >= 10 else ''} "
                "(a single flat colour -- nothing was drawn on the canvas)"
            )

        if self.frame_count > 1:
            duplicates = self.duplicate_frames
            if duplicates == self.frame_count - 1:
                lines.append(
                    "  EVERY FRAME IS IDENTICAL -- the viewer state never "
                    "changed, so the movie will look frozen. Check that the "
                    "keyframes differ from one another."
                )
            elif duplicates:
                lines.append(
                    f"  identical frames: {duplicates} of "
                    f"{self.frame_count - 1} transitions produced no visible "
                    "change"
                )

        for note in self.notes:
            lines.append(f"  {note}")

        return "\n".join(lines)
