"""The GUI export must not livelock when the viewer rewrites applied state.

Regression for an export that stalled at 72%: SLERP returned camera angles
in a different 360-degree branch (-141.6) than the vispy camera reads back
(218.4), so the post-apply signature check re-applied one frame forever.
"""

import itertools

import numpy as np
import pytest
from qtpy.QtCore import QEventLoop, QTimer

from napari_animation import Animation, KeyFrame, ViewerState
from napari_animation._qt.export_controller import (
    MAX_FRAME_REAPPLIES,
    QtExportController,
    _state_differences,
)
from napari_animation.interpolation import slerp

# The camera of the stalled animation, as napari reported it.
ANGLES = (-179.3598668993643, 51.435184875369686, 218.35298737956663)


def test_slerp_keeps_the_endpoints_angle_branch():
    for fraction in (0.0, 0.01, 0.15, 0.5, 1.0):
        np.testing.assert_allclose(
            slerp(ANGLES, ANGLES, fraction), ANGLES, atol=1e-9
        )


def test_slerp_stays_continuous_across_the_wrap():
    a, b = (0.0, 0.0, 170.0), (0.0, 0.0, 190.0)
    steps = [slerp(a, b, f)[2] for f in np.linspace(0, 1, 11)]
    np.testing.assert_allclose(steps, np.linspace(170, 190, 11), atol=1e-9)


def test_signature_compares_angles_as_rotations():
    applied = {"camera": {"angles": [ANGLES[0], ANGLES[1], ANGLES[2] - 360]}}
    readback = {"camera": {"angles": [ANGLES[0] + 3e-14, *ANGLES[1:]]}}
    assert list(_state_differences(applied, readback)) == []

    turned = {"camera": {"angles": [ANGLES[0], ANGLES[1], ANGLES[2] + 1]}}
    assert list(_state_differences(applied, turned)) == ["camera.angles"]


def test_signature_tolerates_float_noise_only():
    applied = {"dims": {"point": [697.0, 307.2, 1024.0], "ndisplay": 3}}
    noisy = {
        "dims": {"point": [697.0, 307.20000000000005, 1024.0], "ndisplay": 3}
    }
    moved = {"dims": {"point": [697.0, 308.0, 1024.0], "ndisplay": 3}}
    assert list(_state_differences(applied, noisy)) == []
    assert list(_state_differences(applied, moved)) == ["dims.point[1]"]


def _export(animation, tmp_path, monkeypatch):
    captured = []

    class Writer:
        def append_data(self, frame):
            captured.append(frame)

        def close(self):
            pass

    controller = QtExportController(
        animation,
        {
            "path": str(tmp_path / "drift.mp4"),
            "prefetch": 0,
            "dask_cache": None,
            "diagnose": False,
        },
    )
    monkeypatch.setattr(controller, "_open_writer", Writer)
    loop = QEventLoop()
    results = []
    controller.finished.connect(
        lambda path, error: (results.append(error), loop.quit())
    )
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(20000)
    controller.start()
    loop.exec_()
    return controller, results, captured


@pytest.fixture
def rotating_animation(make_napari_viewer):
    viewer = make_napari_viewer(ndisplay=3, show=True)
    viewer.add_image(np.random.default_rng(0).random((8, 16, 16)))
    animation = Animation(viewer)
    viewer.camera.angles = ANGLES
    for z in (2.0, 6.0):
        viewer.dims.point = (z, 8.0, 8.0)
        animation.key_frames.append(
            KeyFrame(
                ViewerState.from_viewer(viewer),
                np.zeros((30, 30, 4), np.uint8),
                steps=4,
            )
        )
    # authored in the branch napari reads back, which SLERP used to leave
    for key_frame in animation.key_frames:
        key_frame.viewer_state.camera["angles"] = ANGLES
    yield animation
    animation.close()


def test_export_finishes_with_out_of_branch_camera_angles(
    rotating_animation, tmp_path, monkeypatch
):
    # With large data a canvas paint always lands between apply and the
    # signature check, and VispyCamera.on_draw writes its own Euler branch
    # back into the model. Force that paint so small test data sees it too.
    camera = rotating_animation.viewer.window._qt_viewer.canvas.camera
    await_capture = QtExportController._await_capture

    def paint_then_await(self, generation):
        camera.on_draw(None)
        await_capture(self, generation)

    monkeypatch.setattr(QtExportController, "_await_capture", paint_then_await)
    controller, results, captured = _export(
        rotating_animation, tmp_path, monkeypatch
    )
    assert results == [None]
    assert len(captured) == len(rotating_animation._frames)
    counters = controller._perf.counters
    assert counters.get("frame.drift_accepted", 0) == 0


def test_export_caps_reapplies_when_viewer_keeps_drifting(
    rotating_animation, tmp_path, monkeypatch
):
    ticks = itertools.count()
    monkeypatch.setattr(
        QtExportController,
        "_viewer_signature",
        lambda self: {"tick": float(next(ticks))},
    )
    controller, results, captured = _export(
        rotating_animation, tmp_path, monkeypatch
    )
    n_frames = len(rotating_animation._frames)
    assert results == [None]
    assert len(captured) == n_frames
    counters = controller._perf.counters
    assert counters["frame.drift_accepted"] == n_frames
    assert counters["frame.reapplied"] == n_frames * MAX_FRAME_REAPPLIES
