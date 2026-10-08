"""Tests for the (threaded) animate/render pipeline.

These avoid a real OpenGL context by stubbing ``viewer.screenshot`` and using a
``ViewerModel`` (which has real camera/dims/layers but no Qt canvas).
"""

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_animation import Animation, KeyFrame, ViewerState


@pytest.fixture
def stub_animation(monkeypatch):
    viewer = ViewerModel()
    viewer.add_image(np.random.random((6, 20, 20)), name="img")

    # stub screenshot so no GL context is needed
    def fake_screenshot(self, *args, **kwargs):
        return np.zeros((64, 64, 4), dtype=np.uint8)

    monkeypatch.setattr(
        type(viewer), "screenshot", fake_screenshot, raising=False
    )

    animation = Animation(viewer)
    thumb = np.zeros((30, 30, 4), dtype=np.uint8)
    for i in range(2):
        animation.key_frames.append(
            KeyFrame(
                viewer_state=ViewerState.from_viewer(viewer),
                thumbnail=thumb,
                steps=3,
            )
        )
        viewer.camera.zoom *= 2
    return animation


def test_overwrite_keyframe_preserves_steps_and_name(stub_animation):
    animation = stub_animation
    animation.key_frames[0].name = "first"
    animation.key_frames[0].steps = 7
    old_state = animation.key_frames[0].viewer_state

    # change the view, then overwrite keyframe 0 with it
    animation.viewer.camera.zoom *= 5
    animation.overwrite_keyframe(0)

    kf = animation.key_frames[0]
    assert kf.name == "first"  # name preserved
    assert kf.steps == 7  # interpolation settings preserved
    assert kf.viewer_state != old_state  # captured state updated
    assert len(animation.key_frames) == 2  # overwrite, not insert


def test_animate_writes_mp4(tmp_path, stub_animation):
    out = tmp_path / "movie.mp4"
    stub_animation.animate(str(out), fps=10)
    assert out.exists()
    assert out.stat().st_size > 0


def test_animate_restores_the_working_view(tmp_path, stub_animation):
    viewer = stub_animation.viewer
    viewer.camera.zoom = 7.5
    viewer.dims.set_point(0, 4)
    before = (viewer.camera.zoom, tuple(viewer.dims.point))

    stub_animation.animate(str(tmp_path / "movie.mp4"), diagnose=False)

    assert viewer.camera.zoom == pytest.approx(before[0])
    assert tuple(viewer.dims.point) == pytest.approx(before[1])


def test_animate_with_prefetch_runs(tmp_path, stub_animation, capsys):
    out = tmp_path / "movie.mp4"
    stub_animation.animate(str(out), prefetch=2, prefetch_workers=2)
    assert out.exists()
    assert "Prefetching up to 2 frame(s)" in capsys.readouterr().out


def test_animate_prefetch_disabled(tmp_path, stub_animation, capsys):
    out = tmp_path / "movie.mp4"
    stub_animation.animate(str(out), prefetch=0, dask_cache=None)
    assert out.exists()
    out_text = capsys.readouterr().out
    assert "Prefetching" not in out_text
    assert "dask cache" not in out_text


def test_animate_dask_cache_enabled(tmp_path, stub_animation, capsys):
    out = tmp_path / "movie.mp4"
    stub_animation.animate(str(out), dask_cache=10**7)
    assert out.exists()
    assert "dask cache enabled" in capsys.readouterr().out


def test_animate_writes_png_folder(tmp_path, stub_animation, capsys):
    out = tmp_path / "frames"  # no extension -> folder of PNGs
    stub_animation.animate(str(out))
    pngs = list((tmp_path / "frames").glob("*.png"))
    # 2 keyframes, second has steps=3 -> 3 + 1 frames
    assert len(pngs) == 4
    # the user is told up-front that PNGs (not a video) are being written
    assert "No video file extension" in capsys.readouterr().out


def test_animate_writes_render_log(tmp_path, stub_animation):
    # video -> log sits next to the file
    out = tmp_path / "movie.mp4"
    stub_animation.animate(str(out))
    assert (tmp_path / "movie.render_log.txt").exists()
    log = (tmp_path / "movie.render_log.txt").read_text()
    assert "performance summary" in log and "frames:" in log

    # folder -> log sits inside the folder
    folder_out = tmp_path / "frames"
    stub_animation.animate(str(folder_out))
    assert (tmp_path / "frames" / "render_log.txt").exists()


def test_animate_loud_fallback_when_writer_fails(
    tmp_path, stub_animation, monkeypatch, capsys
):
    import imageio

    def boom(*args, **kwargs):
        raise ValueError("no ffmpeg here")

    monkeypatch.setattr(imageio, "get_writer", boom)
    out = tmp_path / "movie.mp4"
    stub_animation.animate(str(out))

    captured = capsys.readouterr().out
    assert "WARNING" in captured  # the fallback is no longer silent
    # PNGs were written to a sibling folder instead
    assert list((tmp_path / "movie").glob("*.png"))


def test_animate_perf_report_printed(tmp_path, stub_animation, capsys):
    out = tmp_path / "movie.mp4"
    stub_animation.animate(str(out), perf_log=True)
    report = capsys.readouterr().out
    assert "performance summary" in report
    assert "screenshot" in report and "encode/write" in report


def test_failed_render_closes_writer_and_restores_scene(tmp_path, stub_animation, monkeypatch):
    import imageio

    from napari_animation.scene import OrthoSlice

    animation = stub_animation
    section = animation.add_scene_object(OrthoSlice(source="img", position=(4, 5, 6)))
    animation.set_active_section(section.id)
    before = animation.scene.to_dict()
    animation.viewer.camera.zoom = 7.5
    closed = []

    class Writer:
        def append_data(self, frame):
            pass

        def close(self):
            closed.append(True)

    def fail_screenshot(*args, **kwargs):
        raise RuntimeError("screenshot failed")

    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: Writer())
    monkeypatch.setattr(type(animation.viewer), "screenshot", fail_screenshot)
    with pytest.raises(RuntimeError, match="screenshot failed"):
        animation.animate(str(tmp_path / "failed.mp4"), diagnose=False, prefetch=0)
    assert closed == [True]
    assert animation.scene.to_dict() == before
    assert animation.viewer.camera.zoom == pytest.approx(7.5)


def test_estimated_resolution_matches_encoded_video(
    tmp_path, stub_animation, monkeypatch, caplog
):
    import imageio

    from napari_animation.export_estimates import estimate_export

    monkeypatch.setattr(
        type(stub_animation.viewer), "screenshot",
        lambda *args, **kwargs: np.zeros((63, 79, 4), dtype=np.uint8),
    )
    path = tmp_path / "estimate-resolution.mp4"
    stub_animation.animate(
        str(path), fps=20, quality=5, scale_factor=0.5,
        diagnose=False, prefetch=0, dask_cache=None,
    )
    expected = estimate_export(
        (79, 63), len(stub_animation._frames), 20, 0.5, 5, ".mp4"
    )
    reader = imageio.get_reader(path)
    try:
        assert tuple(reader.get_meta_data()["size"]) == expected.resolution
    finally:
        reader.close()
    assert "resizing from" not in caplog.text


@pytest.mark.parametrize("suffix", [".mp4", ".MP4", ".Mp4"])
def test_preset_video_dimensions_ignore_extension_case(
    tmp_path, stub_animation, monkeypatch, caplog, suffix
):
    import imageio.v2 as imageio

    pytest.importorskip("imageio_ffmpeg")

    def screenshot(self, *, canvas_only, size, flash):
        return np.full((*size, 4), 255, dtype=np.uint8)

    monkeypatch.setattr(type(stub_animation.viewer), "screenshot", screenshot)
    path = tmp_path / f"preset{suffix}"
    stub_animation.animate(
        str(path), render_size=(18, 30), diagnose=False, perf_log=False,
        prefetch=0, dask_cache=None,
    )
    with imageio.get_reader(path) as reader:
        assert reader.get_data(0).shape == (18, 30, 3)
    assert "resizing from" not in caplog.text


def test_failed_capture_saves_partial_performance(tmp_path, stub_animation, monkeypatch):
    import json
    import napari_animation.animation as module

    def fail(*args, **kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(module, "capture_export_frame", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        stub_animation.animate(str(tmp_path / "failed.mp4"), diagnose=False)
    data = json.loads((tmp_path / "failed.render_perf.json").read_text())
    assert data["metadata"]["status"] == "failed"
    assert data["phases"]["screenshot"]["count"] == 1
    assert data["phases"]["restore"]["count"] == 1
    assert data["counters"].get("frames_written", 0) == 0


def test_successful_capture_writes_structured_performance(tmp_path, stub_animation):
    import json
    output = tmp_path / "frames"
    stub_animation.animate(str(output), diagnose=False)
    data = json.loads((output / "render_perf.json").read_text())
    assert data["metadata"]["status"] == "complete"
    assert data["counters"]["frames_written"] == len(stub_animation._frames)
    assert data["phases"]["queue.put_wait"]["count"] == len(stub_animation._frames)
    assert data["frames_per_second"] > 0
