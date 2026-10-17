"""Debug-mode timing of GUI actions, from click to a settled canvas.

Set ``NAPARI_ANIMATION_DEBUG=1`` before opening the dock to time every
button, checkbox and list click inside it. Each action records three
moments, all measured from the mouse release that triggers ``clicked``:

``handler``
    the click's Qt slots have returned and the event loop runs again;
``settled``
    background section projections are done and every visible layer has
    loaded its slice;
``painted``
    the canvas has drawn the settled state.

Nested phases instrumented with :func:`napari_animation.perf.render_phase`
are recorded per action, so the log also names where the time went. With
``NAPARI_ANIMATION_DEBUG=profile`` each action is also run under cProfile:
the functions with the most own time and the plugin code paths with the
most inclusive time go in the row, and the full profile is saved next to
the CSV. Rows go to the ``napari_animation`` logger and to a
CSV file: ``NAPARI_ANIMATION_DEBUG_CSV`` if set, otherwise a timestamped file
in ``~/napari-animation-debug``.
"""

from __future__ import annotations

import cProfile
import csv
import logging
import pstats
import re
import os
import time
from datetime import datetime
from pathlib import Path

from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QApplication,
    QWidget,
)

from ..perf import PerfLogger, _active_perf

logger = logging.getLogger("napari_animation")

DEBUG_ENV = "NAPARI_ANIMATION_DEBUG"
CSV_ENV = "NAPARI_ANIMATION_DEBUG_CSV"

#: Give up waiting for an action to settle after this long.
SETTLE_TIMEOUT = 120.0
#: Polling interval while waiting for background work.
POLL_MS = 10

FIELDS = (
    "timestamp",
    "action",
    "widget",
    "handler_ms",
    "settled_ms",
    "painted_ms",
    "status",
    "modal",
    "top_phases",
    "top_functions",
    "plugin_calls",
)


def debug_enabled() -> bool:
    return os.environ.get(DEBUG_ENV, "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def profile_enabled() -> bool:
    return os.environ.get(DEBUG_ENV, "").strip().lower() == "profile"


_PACKAGE = str(Path(__file__).resolve().parents[1])


def _where(filename, line, name) -> str:
    if filename == "~":
        return name
    return f"{Path(filename).name}:{line}({name})"


def _top_functions(profile, limit=4):
    """Summarize where a profiled action spent its time.

    Returns the functions with the most own time (often native calls such
    as an OpenGL upload) and the plugin functions with the most inclusive
    time, outermost first, which name the code path that led there.
    """
    stats = pstats.Stats(profile).stats
    own = sorted(stats.items(), key=lambda item: item[1][2], reverse=True)
    plugin = sorted(
        (
            item
            for item in stats.items()
            if item[0][0].startswith(_PACKAGE)
            and not item[0][0].endswith("gui_timing.py")
        ),
        key=lambda item: item[1][3],
        reverse=True,
    )
    return (
        "; ".join(
            f"{_where(*key)} {value[2] * 1000:.0f}ms"
            for key, value in own[:limit]
        ),
        "; ".join(
            f"{_where(*key)} {value[3] * 1000:.0f}ms"
            for key, value in plugin[:limit]
        ),
    )


def _default_csv_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path.home() / "napari-animation-debug" / f"gui-timing-{stamp}.csv"


def _widget_path(widget, root) -> str:
    """A readable path such as ``SceneWidget/QGroupBox/addSliceButton``."""
    parts = []
    node = widget
    while node is not None and node is not root:
        name = node.objectName() or type(node).__name__
        parts.append(name)
        node = node.parentWidget()
    return "/".join(reversed(parts))


def _button_attribute_names(root):
    """Map widgets to the attribute names the plugin gave them."""
    names = {}
    for owner in [root, *root.findChildren(QWidget)]:
        for key, value in vars(owner).items():
            if isinstance(value, QWidget) and not key.startswith("__"):
                names.setdefault(id(value), key)
    return names


def _label(widget) -> str:
    text = ""
    if isinstance(widget, QAbstractButton):
        text = widget.text() or widget.toolTip()
    return " ".join(text.replace("&", "").split())


class GuiActionTimer(QObject):
    """Time clicks inside ``root`` (see the module docstring)."""

    def __init__(self, viewer, root, csv_path=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.root = root
        csv_path = csv_path or os.environ.get(CSV_ENV)
        self.csv_path = Path(csv_path) if csv_path else _default_csv_path()
        self.records = []
        self._pending = None
        self._generation = 0
        self._names = {}
        self._csv_ready = False
        self._profile = profile_enabled()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        logger.info("GUI timing enabled; writing %s", self.csv_path)

    def close(self):
        if self._pending is not None:
            self._finish("interrupted")
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

    # -- action detection --------------------------------------------------

    def _action_widget(self, obj):
        if not isinstance(obj, QWidget):
            return None
        if obj is not self.root and not self.root.isAncestorOf(obj):
            return None
        if isinstance(obj, QAbstractButton):
            return obj if obj.isEnabled() else None
        parent = obj.parentWidget()
        if isinstance(parent, QAbstractItemView) and obj is parent.viewport():
            return parent
        return None

    def eventFilter(self, obj, event):
        if (
            event.type() == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
        ):
            widget = self._action_widget(obj)
            if widget is not None:
                self.begin(widget)
        return False

    # -- one action ---------------------------------------------------------

    def describe(self, widget):
        if id(widget) not in self._names:
            self._names = _button_attribute_names(self.root)
        name = self._names.get(id(widget))
        label = _label(widget)
        if isinstance(widget, QAbstractItemView):
            index = widget.currentIndex()
            label = f"row {index.row()}" if index.isValid() else "list"
        action = " ".join(part for part in (name, f"[{label}]") if part)
        return action or type(widget).__name__, _widget_path(widget, self.root)

    def begin(self, widget, action=None):
        """Start timing an action (also callable from scripts)."""
        if self._pending is not None:
            self._finish("interrupted")
        described, path = self.describe(widget)
        perf = PerfLogger(enabled=True)
        perf.start()
        token = _active_perf.set((perf, None))
        profile = None
        if self._profile:
            profile = cProfile.Profile()
            try:
                profile.enable()
            except ValueError:  # another profiler is active
                profile = None
        self._generation += 1
        self._pending = {
            "generation": self._generation,
            "action": action or described,
            "widget": path,
            "start": time.perf_counter(),
            "perf": perf,
            "token": token,
            "handler": None,
            "settled": None,
            "modal": False,
            "profile": profile,
        }
        generation = self._generation
        QTimer.singleShot(0, lambda: self._handler_done(generation))

    def _current(self, generation):
        pending = self._pending
        if pending is None or pending["generation"] != generation:
            return None
        return pending

    def _handler_done(self, generation):
        pending = self._current(generation)
        if pending is None or pending["handler"] is not None:
            return
        pending["handler"] = time.perf_counter()
        self._poll_settled(generation)

    def _busy(self) -> bool:
        from ..scene.processing import service_for_viewer

        service = service_for_viewer(self.viewer)
        if (
            service._active is not None
            or service._pending is not None
            or service._batch_pending
        ):
            return True
        slicer = getattr(self.viewer, "_layer_slicer", None)
        tasks = getattr(slicer, "_layers_to_task", None)
        if tasks and any(not task.done() for task in tasks.values()):
            return True
        return any(
            not getattr(layer, "loaded", True)
            for layer in self.viewer.layers
            if layer.visible
        )

    def _poll_settled(self, generation):
        pending = self._current(generation)
        if pending is None:
            return
        if QApplication.activeModalWidget() is not None:
            pending["modal"] = True
        if time.perf_counter() - pending["start"] > SETTLE_TIMEOUT:
            self._finish("timeout")
            return
        if self._busy():
            QTimer.singleShot(POLL_MS, lambda: self._poll_settled(generation))
            return
        pending["settled"] = time.perf_counter()
        canvas = getattr(
            getattr(getattr(self.viewer, "window", None), "_qt_viewer", None),
            "canvas",
            None,
        )
        scene_canvas = getattr(canvas, "_scene_canvas", None)
        if scene_canvas is None:
            self._finish("ok")
            return

        def drawn(event=None):
            scene_canvas.events.draw.disconnect(drawn)
            # Draw events fire before the frame is swapped; the next loop
            # turn is when the user sees it.
            if self._current(generation) is not None:
                QTimer.singleShot(
                    0,
                    lambda: self._current(generation) and self._finish("ok"),
                )

        scene_canvas.events.draw.connect(drawn)
        pending["drawn"] = (scene_canvas, drawn)
        scene_canvas.update()

    def _finish(self, status):
        pending, self._pending = self._pending, None
        if pending is None:
            return
        now = time.perf_counter()
        drawn = pending.get("drawn")
        if drawn is not None and status != "ok":
            scene_canvas, callback = drawn
            try:
                scene_canvas.events.draw.disconnect(callback)
            except Exception:  # noqa: BLE001 - already disconnected
                pass
        try:
            _active_perf.reset(pending["token"])
        except ValueError:
            # Reset from a different context; clear rather than leak.
            _active_perf.set((None, None))
        perf = pending["perf"]
        perf.stop()
        profile = pending["profile"]
        top_functions = plugin_calls = ""
        if profile is not None:
            profile.disable()
            top_functions, plugin_calls = _top_functions(profile)
            slug = re.sub(r"[^A-Za-z0-9]+", "-", pending["action"]).strip("-")
            stem = f"{len(self.records) + 1:03d}-{slug[:40]}"
            try:
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                profile.dump_stats(
                    str(
                        self.csv_path.with_name(
                            f"{self.csv_path.stem}-{stem}.prof"
                        )
                    )
                )
            except OSError as error:
                logger.warning("Could not save GUI profile: %s", error)

        def ms(value):
            return (
                ""
                if value is None
                else round((value - pending["start"]) * 1000.0, 1)
            )

        phases = perf.snapshot().get("phases", {})
        top = sorted(
            phases.items(),
            key=lambda item: item[1]["total_seconds"],
            reverse=True,
        )[:5]
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": pending["action"],
            "widget": pending["widget"],
            "handler_ms": ms(pending["handler"]),
            "settled_ms": ms(pending["settled"]),
            "painted_ms": ms(now) if status == "ok" else "",
            "status": status,
            "modal": pending["modal"],
            "top_phases": "; ".join(
                f"{name} {data['total_seconds'] * 1000:.0f}ms"
                f" x{data['count']}"
                for name, data in top
            ),
            "top_functions": top_functions,
            "plugin_calls": plugin_calls,
        }
        self.records.append(record)
        logger.info(
            "GUI %s: handler %s ms, settled %s ms, painted %s ms (%s)%s",
            record["action"],
            record["handler_ms"],
            record["settled_ms"],
            record["painted_ms"],
            status,
            f" -- {record['top_phases']}" if record["top_phases"] else "",
        )
        self._write(record)

    def _write(self, record):
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self._csv_ready and not self.csv_path.exists()
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(record)
            self._csv_ready = True
        except OSError as error:
            logger.warning("Could not write GUI timing CSV: %s", error)

    @property
    def busy(self) -> bool:
        """Whether an action is still being timed."""
        return self._pending is not None
