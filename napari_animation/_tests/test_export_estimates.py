import pytest

from napari_animation.export_estimates import estimate_export


def test_all_quality_levels_keep_resolution_and_increase_size():
    estimates = [
        estimate_export((1919, 1079), 300, 20, 1, level, ".mp4")
        for level in range(1, 11)
    ]
    assert {entry.resolution for entry in estimates} == {(1920, 1088)}
    assert all(entry.duration == 15 for entry in estimates)
    assert all(
        left.size_range[0] < right.size_range[0]
        and left.size_range[1] < right.size_range[1]
        for left, right in zip(estimates, estimates[1:])
    )


def test_scale_frames_and_fps_have_consistent_effects():
    base = estimate_export((1920, 1088), 300, 20, 1, 5, ".mp4")
    half = estimate_export((1920, 1088), 300, 20, 0.5, 5, ".mp4")
    longer = estimate_export((1920, 1088), 600, 20, 1, 5, ".mp4")
    faster = estimate_export((1920, 1088), 300, 40, 1, 5, ".mp4")
    assert half.resolution == (960, 544)
    assert half.size_range == pytest.approx(
        tuple(v / 4 for v in base.size_range)
    )
    assert longer.size_range == pytest.approx(
        tuple(v * 2 for v in base.size_range)
    )
    assert faster.duration == base.duration / 2
    assert faster.size_range == base.size_range


@pytest.mark.parametrize("suffix", [".gif", "", ".png"])
def test_non_video_formats_do_not_apply_quality_or_video_padding(suffix):
    low = estimate_export((101, 53), 20, 10, 1, 1, suffix)
    high = estimate_export((101, 53), 20, 10, 1, 10, suffix)
    assert low.resolution == (101, 53)
    assert low.size_range == high.size_range
    assert not low.quality_applies


def test_missing_or_invalid_output_has_no_size_estimate():
    assert estimate_export((800, 600), 0, 20, 1, 5, ".mp4").size_range is None
    assert (
        estimate_export((10, 10), 10, 20, 0.001, 5, ".mp4").size_range is None
    )
    assert (
        estimate_export((800, 600), 20, 20, 1, 5, ".unknown").size_range
        is None
    )
