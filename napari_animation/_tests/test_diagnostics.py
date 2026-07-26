"""Tests for render diagnostics.

Each test drives one of the failure modes that a successful-looking render can
hide, and checks the report actually names it.
"""

import numpy as np
import pytest

from napari_animation.diagnostics import RenderDiagnostics


def _frame(value=128, shape=(64, 48), channels=4):
    frame = np.full((*shape, channels), value, dtype=np.uint8)
    if channels == 4:
        frame[..., 3] = 255
    return frame


def _varied_frame(index, shape=(64, 48)):
    """A frame whose content genuinely differs from its neighbours."""
    rng = np.random.default_rng(index)
    frame = rng.integers(0, 255, (*shape, 4), dtype=np.uint8)
    frame[..., 3] = 255
    return frame


def test_reports_frame_geometry():
    diagnostics = RenderDiagnostics()
    for i in range(5):
        diagnostics.record_frame(i, _varied_frame(i))

    report = diagnostics.report()

    assert "frames rendered : 5 @ 48x64" in report
    assert "4 channels" in report


def test_detects_frame_size_changing_mid_render():
    """The single most likely cause of a corrupt video."""
    diagnostics = RenderDiagnostics()
    diagnostics.record_frame(0, _varied_frame(0, shape=(64, 48)))
    diagnostics.record_frame(1, _varied_frame(1, shape=(64, 48)))
    diagnostics.record_frame(2, _varied_frame(2, shape=(72, 48)))

    report = diagnostics.report()

    assert "FRAME SIZE CHANGED MID-RENDER" in report
    assert "frame 2: 48x72" in report
    assert "resize" in report.lower()


def test_detects_blank_frames():
    diagnostics = RenderDiagnostics()
    diagnostics.record_frame(0, _varied_frame(0))
    diagnostics.record_frame(1, _frame(0))  # flat black

    report = diagnostics.report()

    assert "BLANK FRAMES" in report
    assert "nothing was drawn" in report


def test_opaque_alpha_does_not_mask_a_blank_frame():
    """Alpha is 255 on a blank napari screenshot; it must be ignored."""
    diagnostics = RenderDiagnostics()
    diagnostics.record_frame(0, _frame(0, channels=4))

    assert "BLANK FRAMES" in diagnostics.report()


def test_detects_a_frozen_animation():
    diagnostics = RenderDiagnostics()
    for i in range(6):
        diagnostics.record_frame(i, _frame(128))

    report = diagnostics.report()

    assert "EVERY FRAME IS IDENTICAL" in report
    assert "keyframes differ" in report


def test_counts_partial_duplicate_frames():
    diagnostics = RenderDiagnostics()
    diagnostics.record_frame(0, _varied_frame(0))
    diagnostics.record_frame(1, _varied_frame(0))  # no change
    diagnostics.record_frame(2, _varied_frame(2))

    report = diagnostics.report()

    assert "identical frames: 1 of 2" in report
    assert "EVERY FRAME IS IDENTICAL" not in report


def test_clean_render_reports_no_warnings():
    diagnostics = RenderDiagnostics()
    for i in range(5):
        diagnostics.record_frame(i, _varied_frame(i))

    report = diagnostics.report()

    assert "WARNING" not in report
    assert "BLANK" not in report
    assert "IDENTICAL" not in report


def test_disabled_diagnostics_are_inert():
    diagnostics = RenderDiagnostics(enabled=False)
    diagnostics.record_frame(0, _varied_frame(0))

    assert diagnostics.report() == ""
    assert diagnostics.frame_count == 0


def test_record_frame_never_raises():
    """Diagnostics must not be able to break a long render."""
    diagnostics = RenderDiagnostics()

    diagnostics.record_frame(0, "not an array")

    assert diagnostics.report()  # still produces something


# ------------------------------------------------------- output verification


def test_missing_output_is_reported(tmp_path):
    diagnostics = RenderDiagnostics()
    diagnostics.verify_output(tmp_path / "never_written.mp4", 10)

    assert "OUTPUT MISSING" in diagnostics.report()


def test_tiny_output_is_flagged(tmp_path):
    path = tmp_path / "truncated.mp4"
    path.write_bytes(b"\x00" * 64)

    diagnostics = RenderDiagnostics()
    diagnostics.verify_output(path, 46)

    report = diagnostics.report()
    assert "too small" in report
    assert "could not be reopened" in report or "no decodable frames" in report


def test_round_trip_of_a_real_video(tmp_path):
    """A good render reads back with the frame count that was written."""
    imageio = pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")

    path = tmp_path / "good.mp4"
    diagnostics = RenderDiagnostics()
    writer = imageio.get_writer(str(path), fps=10, quality=5)
    frames = 12
    for i in range(frames):
        frame = _varied_frame(i, shape=(64, 48))
        diagnostics.record_frame(i, frame)
        writer.append_data(frame)
    writer.close()

    diagnostics.verify_output(path, frames)
    report = diagnostics.report()

    assert f"read back {frames} frame(s)" in report
    assert "WARNING" not in report


def test_animate_emits_diagnostics(tmp_path, capsys):
    """The whole pipeline is wired up, not just the diagnostics class."""
    pytest.importorskip("imageio_ffmpeg")
    from napari.components import ViewerModel

    from napari_animation import Animation

    class _HeadlessViewer(ViewerModel):
        # a screenshot that reflects the camera, so frames genuinely differ
        def screenshot(self, *args, **kwargs):
            frame = np.zeros((64, 48, 4), dtype=np.uint8)
            frame[..., 3] = 255
            shift = int(self.camera.angles[2]) % 40
            frame[:, shift : shift + 4, 0] = 255
            return frame

    viewer = _HeadlessViewer()
    viewer.add_image(np.random.random((4, 16, 16)), name="img")
    animation = Animation(viewer)
    viewer.camera.angles = (0, 0, 0)
    animation.capture_keyframe()
    viewer.camera.angles = (0, 0, 90)
    animation.capture_keyframe(steps=8)

    animation.animate(str(tmp_path / "movie.mp4"), fps=10, perf_log=False)

    out = capsys.readouterr().out
    assert "render diagnostics" in out
    assert "frames rendered : 9 @ 48x64" in out
    assert "read back 9 frame(s)" in out


def test_animate_can_disable_diagnostics(tmp_path, capsys):
    pytest.importorskip("imageio_ffmpeg")
    from napari.components import ViewerModel

    from napari_animation import Animation

    class _HeadlessViewer(ViewerModel):
        def screenshot(self, *args, **kwargs):
            return np.zeros((64, 48, 4), dtype=np.uint8)

    viewer = _HeadlessViewer()
    viewer.add_image(np.random.random((4, 16, 16)), name="img")
    animation = Animation(viewer)
    animation.capture_keyframe()
    animation.capture_keyframe(steps=4)

    animation.animate(
        str(tmp_path / "movie.mp4"),
        fps=10,
        perf_log=False,
        diagnose=False,
    )

    assert "render diagnostics" not in capsys.readouterr().out


def test_output_stream_parameters_are_reported(tmp_path):
    """ "Decodes but won't play" is a codec problem, so name the codec."""
    imageio = pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")

    path = tmp_path / "probe.mp4"
    writer = imageio.get_writer(str(path), fps=10, quality=5)
    for i in range(6):
        writer.append_data(_varied_frame(i, shape=(64, 48)))
    writer.close()

    diagnostics = RenderDiagnostics()
    diagnostics.verify_output(path, 6)
    report = diagnostics.report()

    assert "Stream #0:0" in report
    assert "h264" in report
    assert "yuv420p" in report


def test_compat_encoding_is_applied_and_announced(tmp_path, capsys):
    """mp4 output must not depend on the local ffmpeg's default codec."""
    pytest.importorskip("imageio_ffmpeg")
    from napari.components import ViewerModel

    from napari_animation import Animation
    from napari_animation.animation import COMPAT_ENCODER

    class _HeadlessViewer(ViewerModel):
        def screenshot(self, *args, **kwargs):
            frame = np.zeros((64, 48, 4), dtype=np.uint8)
            frame[..., 3] = 255
            frame[:, int(self.camera.angles[2]) % 40] = 255
            return frame

    viewer = _HeadlessViewer()
    viewer.add_image(np.random.random((4, 16, 16)), name="img")
    animation = Animation(viewer)
    viewer.camera.angles = (0, 0, 0)
    animation.capture_keyframe()
    viewer.camera.angles = (0, 0, 30)
    animation.capture_keyframe(steps=6)

    animation.animate(str(tmp_path / "movie.mp4"), fps=10, perf_log=False)

    out = capsys.readouterr().out
    assert "compatibility mode" in out
    assert COMPAT_ENCODER["pixelformat"] in out
    # and the file really is in that pixel format
    assert "yuv420p" in out


def test_macro_block_padding_is_explained(tmp_path):
    """Sizes not divisible by 16 get padded; say so rather than leaving a
    mystery black strip."""
    imageio = pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")

    path = tmp_path / "padded.mp4"
    diagnostics = RenderDiagnostics()
    writer = imageio.get_writer(str(path), fps=10, quality=5)
    for i in range(4):
        # 546 is not a multiple of 16, matching the reported canvas height
        frame = _varied_frame(i, shape=(546, 448))
        diagnostics.record_frame(i, frame)
        writer.append_data(frame)
    writer.close()

    diagnostics.verify_output(path, 4)
    report = diagnostics.report()

    assert "was encoded as" in report
    assert "multiple of 16" in report
