"""Click every button in the animation dock on a 3D keyframe and time it.

usage: python benchmarks/time_gui_buttons.py animation.json out.csv [--keyframe N]
File dialogs are answered with Cancel (Load Keyframes gets the JSON path),
the Save Animation dialog is cancelled after it is built, and any other
modal window is closed. The source JSON is never written.
"""

import argparse
import os
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("animation", type=Path)
p.add_argument("csv", type=Path)
p.add_argument("--keyframe", type=int, default=3)
args = p.parse_args()
os.environ.setdefault("NAPARI_ANIMATION_DEBUG", "1")
os.environ["NAPARI_ANIMATION_DEBUG_CSV"] = str(args.csv)
if args.csv.exists():
    args.csv.unlink()

import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("napari_animation").setLevel(logging.INFO)

import napari
from qtpy.QtCore import Qt, QTimer
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QAbstractButton,
    QApplication,
    QFileDialog,
)

from napari_animation._qt import AnimationWidget
from napari_animation._qt import savedialog_widget

viewer = napari.Viewer()
viewer.window._qt_window.resize(1600, 1000)
widget = AnimationWidget(viewer)
viewer.window.add_dock_widget(widget, area="right")
timer = widget._gui_timer
assert timer is not None, "debug mode did not install the GUI timer"

# -- dialogs ----------------------------------------------------------------
QFileDialog.getOpenFileName = staticmethod(
    lambda *a, **k: (str(args.animation), "")
)
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
savedialog_widget.SaveDialogWidget.exec_ = lambda self: 0
closed = []


def close_modals():
    modal = QApplication.activeModalWidget()
    if modal is not None:
        closed.append(type(modal).__name__)
        modal.close()


modal_timer = QTimer()
modal_timer.timeout.connect(close_modals)
modal_timer.start(200)


def wait_idle(limit=180):
    end = time.perf_counter() + limit
    while time.perf_counter() < end:
        QApplication.processEvents()
        if not timer.busy:
            break
        time.sleep(0.002)
    # let deferred refinements (e.g. 300 ms section refine) land
    t = time.perf_counter() + 0.4
    while time.perf_counter() < t:
        QApplication.processEvents()
        time.sleep(0.005)


def click(button):
    QTest.mouseClick(
        button, Qt.LeftButton, Qt.NoModifier, button.rect().center()
    )
    wait_idle()


def click_row(row):
    view = widget.keyframesListWidget
    rect = view.visualRect(view.model().index(row, 0))
    QTest.mouseClick(
        view.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center()
    )
    wait_idle()


def reset_to_keyframe():
    widget.animation.key_frames.selection.active = None
    widget.animation.set_to_keyframe(args.keyframe)
    end = time.perf_counter() + 60
    while time.perf_counter() < end and timer._busy():
        QApplication.processEvents()
        time.sleep(0.005)
    for _ in range(20):
        QApplication.processEvents()


QApplication.processEvents()
click(widget.loadKeyframesButton)
print("loaded", len(widget.animation.key_frames), "keyframes", flush=True)
for row in range(len(widget.animation.key_frames)):
    click_row(row)
click_row(args.keyframe)
print("ndisplay", viewer.dims.ndisplay, flush=True)

LAST = {"deleteButton", "removeButton", "loadKeyframesButton"}

buttons = [
    b
    for b in widget.findChildren(QAbstractButton)
    if b.isVisibleTo(widget) and b.isEnabled()
]


def attr(button):
    return timer.describe(button)[0].split(" [")[0]


buttons.sort(key=lambda b: attr(b) in LAST)
for button in buttons:
    if attr(button) == "loadKeyframesButton":
        continue
    if not (button.isVisibleTo(widget) and button.isEnabled()):
        continue
    reset_to_keyframe()
    label = timer.describe(button)[0]
    print("click", label, flush=True)
    click(button)
    if button.isCheckable():
        # restore toggles so later buttons see the same scene
        print("click again", label, flush=True)
        click(button)

print("modals closed:", closed, flush=True)
rows = sorted(
    (r for r in timer.records if r["painted_ms"] != ""),
    key=lambda r: -float(r["painted_ms"]),
)
print("\nslowest actions (click -> painted):")
for r in rows[:15]:
    print(f"{r['painted_ms']:>9} ms  {r['action']}  [{r['top_phases']}]")
timer.close()
viewer.close()
