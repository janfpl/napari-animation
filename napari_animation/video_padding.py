"""Prepare video frames without changing the pixels rendered by napari."""

import numpy as np

VIDEO_BLOCK_SIZE = 16


def pad_video_frame(frame, block_size=VIDEO_BLOCK_SIZE):
    """Pad the right and bottom edges with opaque black pixels.

    Imageio otherwise asks FFmpeg to *scale* each non-multiple-of-16 frame.
    Returning the original array when it is already aligned avoids a copy.
    """
    height, width = frame.shape[:2]
    padded_height = ((height + block_size - 1) // block_size) * block_size
    padded_width = ((width + block_size - 1) // block_size) * block_size
    if (padded_height, padded_width) == (height, width):
        return frame

    padded = np.zeros(
        (padded_height, padded_width, *frame.shape[2:]), dtype=frame.dtype
    )
    if frame.ndim == 3 and frame.shape[2] == 4:
        padded[..., 3] = (
            np.iinfo(frame.dtype).max
            if np.issubdtype(frame.dtype, np.integer)
            else 1
        )
    padded[:height, :width] = frame
    return padded
