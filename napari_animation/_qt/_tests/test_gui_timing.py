import csv

import numpy as np
from napari.components import ViewerModel
from qtpy.QtCore import Qt

from napari_animation._qt import AnimationWidget
from napari_animation._qt.gui_timing import FIELDS, debug_enabled


class _HeadlessViewer(ViewerModel):
    def screenshot(self, *args, **kwargs):
        return np.zeros((60, 60, 4), dtype=np.uint8)


def _widget(qtbot):
    viewer = _HeadlessViewer()
    viewer.add_image(np.random.random((8, 16, 16)), name="img")
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def test_debug_mode_is_off_by_default(qtbot, monkeypatch):
    monkeypatch.delenv("NAPARI_ANIMATION_DEBUG", raising=False)
    assert not debug_enabled()
    assert _widget(qtbot)._gui_timer is None


def test_debug_mode_times_a_click(qtbot, monkeypatch, tmp_path):
    path = tmp_path / "timing.csv"
    monkeypatch.setenv("NAPARI_ANIMATION_DEBUG", "profile")
    monkeypatch.setenv("NAPARI_ANIMATION_DEBUG_CSV", str(path))
    widget = _widget(qtbot)
    timer = widget._gui_timer
    button = widget.keyframesListControlWidget.captureButton

    qtbot.mouseClick(button, Qt.LeftButton)
    qtbot.waitUntil(lambda: not timer.busy, timeout=10000)

    assert len(widget.animation.key_frames) == 1
    (record,) = timer.records
    assert record["action"].startswith("captureButton")
    assert record["status"] == "ok"
    assert 0 <= record["handler_ms"] <= record["painted_ms"]
    # the capture applies the new keyframe, an instrumented phase
    assert "state.apply" in record["top_phases"]
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(FIELDS)
    assert rows[0]["action"] == record["action"]
    assert list(tmp_path.glob("timing-001-*.prof"))


def test_clicks_outside_the_dock_are_ignored(qtbot, monkeypatch, tmp_path):
    from qtpy.QtWidgets import QPushButton

    monkeypatch.setenv("NAPARI_ANIMATION_DEBUG", "1")
    monkeypatch.setenv("NAPARI_ANIMATION_DEBUG_CSV", str(tmp_path / "t.csv"))
    widget = _widget(qtbot)
    other = QPushButton("elsewhere")
    qtbot.addWidget(other)
    other.show()
    qtbot.mouseClick(other, Qt.LeftButton)
    assert not widget._gui_timer.busy
    assert widget._gui_timer.records == []
