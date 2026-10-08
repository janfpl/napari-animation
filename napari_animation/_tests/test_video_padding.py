"""Video padding must preserve the captured pixels and opaque black border."""

import numpy as np

from napari_animation.video_padding import pad_video_frame


def test_pad_video_frame_preserves_pixels_and_adds_opaque_black():
    frame = np.arange(18 * 17 * 4, dtype=np.uint8).reshape(18, 17, 4)
    padded = pad_video_frame(frame)

    assert padded.shape == (32, 32, 4)
    np.testing.assert_array_equal(padded[:18, :17], frame)
    assert np.all(padded[18:, :, :3] == 0)
    assert np.all(padded[:, 17:, :3] == 0)
    assert np.all(padded[18:, :, 3] == 255)
    assert np.all(padded[:, 17:, 3] == 255)
    assert pad_video_frame(padded) is padded
