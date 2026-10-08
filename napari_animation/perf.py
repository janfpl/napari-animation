"""Lightweight performance logging for animation rendering.

The :class:`PerfLogger` accumulates wall-clock time spent in named phases of
the render/encode pipeline (interpolation, applying viewer state, taking the
screenshot, encoding/writing) so it is easy to see which step is rate
limiting. It is intentionally dependency-free and cheap when disabled.
"""

from __future__ import annotations

import logging
import json
import os
import threading
import time
from collections import OrderedDict, deque
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

#: Module logger; configure napari_animation's logger to see perf output.
logger = logging.getLogger("napari_animation")


_active_perf = ContextVar("animation_perf", default=(None, None))


@contextmanager
def render_phase(name):
    """Instrument a nested operation only inside an active export timer."""
    perf, frame = _active_perf.get()
    if perf is None:
        yield
    else:
        with perf.timer(name, frame=frame):
            yield


def timed_render(name):
    def decorate(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            with render_phase(name):
                return method(*args, **kwargs)
        return wrapped
    return decorate


class PerfLogger:
    """Thread-safe inclusive timings, with bounded samples and event history.

    Nested operations and concurrent workers overlap: phase totals must never
    be added to estimate elapsed time. Percentages use measured wall time.
    """

    def __init__(self, enabled=True, log=logger, sample_limit=10000,
                 event_limit=100000):
        self.enabled = enabled
        self._logger = log
        self._lock = threading.RLock()
        self._sample_limit = sample_limit
        self._events = deque(maxlen=event_limit)
        self._totals = OrderedDict()
        self._counts = OrderedDict()
        self._samples = {}
        self._maxima = {}
        self._wall_start = None
        self._wall_stop = None
        self.metadata = {}
        self.counters = {}

    def start(self):
        if self.enabled:
            self._wall_start = time.perf_counter()
            self._wall_stop = None

    def stop(self):
        if self.enabled and self._wall_stop is None:
            self._wall_stop = time.perf_counter()

    def reset(self):
        with self._lock:
            self._totals.clear()
            self._counts.clear()
            self._samples.clear()
            self._maxima.clear()
            self._events.clear()
            self.metadata.clear()
            self.counters.clear()
            self._wall_start = self._wall_stop = None

    def count(self, name, amount=1):
        if self.enabled:
            with self._lock:
                self.counters[name] = self.counters.get(name, 0) + amount

    def record(self, name, dt, *, frame=None, start=None):
        if not self.enabled:
            return
        with self._lock:
            self._totals[name] = self._totals.get(name, 0.0) + dt
            self._counts[name] = self._counts.get(name, 0) + 1
            self._samples.setdefault(name, deque(maxlen=self._sample_limit)).append(dt)
            self._maxima[name] = max(dt, self._maxima.get(name, 0.0))
            self._events.append({
                "phase": name, "seconds": dt, "frame": frame,
                "start_seconds": (start - self._wall_start)
                if start is not None and self._wall_start is not None else None,
                "thread": threading.current_thread().name,
            })

    @contextmanager
    def timer(self, name, *, frame=None):
        if not self.enabled:
            yield
            return
        if frame is None:
            frame = _active_perf.get()[1]
        token = _active_perf.set((self, frame))
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start, frame=frame, start=start)
            _active_perf.reset(token)

    def snapshot(self):
        with self._lock:
            wall = ((self._wall_stop or time.perf_counter()) - self._wall_start
                    if self._wall_start is not None else 0.0)
            phases = {}
            for name, total in self._totals.items():
                samples = sorted(self._samples[name])

                def percentile(fraction):
                    pos = (len(samples) - 1) * fraction
                    low = int(pos)
                    return samples[low] + (samples[min(low + 1, len(samples) - 1)]
                                           - samples[low]) * (pos - low)
                phases[name] = {
                    "total_seconds": total, "count": self._counts[name],
                    "mean_ms": total / self._counts[name] * 1000,
                    "p50_ms": percentile(.5) * 1000,
                    "p95_ms": percentile(.95) * 1000,
                    "max_ms": self._maxima[name] * 1000,
                    "wall_percent": 100 * total / wall if wall else 0,
                    "retained_samples": len(samples),
                }
            frames = self.counters.get("frames_written", 0)
            return {
                "schema_version": 1, "wall_seconds": wall,
                "frames_per_second": frames / wall if wall else 0,
                "metadata": dict(self.metadata), "counters": dict(self.counters),
                "phases": phases, "events": list(self._events),
                "events_dropped": sum(self._counts.values()) - len(self._events),
                "semantics": "Inclusive wall timings overlap across nested phases and threads. "
                             "Do not sum phase percentages. Quantiles use the most recent retained samples.",
            }

    def report(self):
        data = self.snapshot()
        lines = ["napari-animation performance summary:", data["semantics"]]
        for name, phase in data["phases"].items():
            lines.append(
                f"  {name:<28} total={phase['total_seconds']:8.3f}s  "
                f"mean={phase['mean_ms']:8.2f}ms  p95={phase['p95_ms']:8.2f}ms  "
                f"max={phase['max_ms']:8.2f}ms  n={phase['count']}  "
                f"wall={phase['wall_percent']:.1f}%"
            )
        lines.append(f"  wall-clock total={data['wall_seconds']:.3f}s; "
                     f"throughput={data['frames_per_second']:.3f} frames/s")
        lines.append(f"  counters: {data['counters']}")
        return "\n".join(lines)

    def log_report(self, level=logging.INFO):
        text = self.report()
        if self.enabled and self._totals:
            self._logger.log(level, text)
        return text

    def save(self, output):
        """Write a machine-readable sidecar; logging must not fail an export."""
        if not self.enabled:
            return None
        output = Path(output)
        path = (output.with_name(output.stem + ".render_perf.json")
                if output.suffix else output / "render_perf.json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.snapshot(), indent=2, default=str) + "\n",
                            encoding="utf-8")
            return path
        except OSError as error:
            self._logger.warning("Could not write performance JSON: %s", error)
            return None


class InteractionTrace:
    """Opt-in callback and heartbeat trace for a real Qt interaction session.

    Set ``NAPARI_ANIMATION_PERF_TRACE`` to a JSON output path before opening
    the dock.  Timings are gathered across GUI and projection/encoder threads.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._events = []
        self._heartbeat = None
        self._last_beat = None

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get("NAPARI_ANIMATION_PERF_TRACE"))

    @contextmanager
    def timer(self, name):
        if self.path is None:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._events.append({
                    "phase": name,
                    "start": start,
                    "seconds": elapsed,
                    "thread": threading.get_ident(),
                })

    def start_heartbeat(self):
        if self.path is None or self._heartbeat is not None:
            return
        try:
            from qtpy.QtCore import QTimer
            from qtpy.QtWidgets import QApplication

            if QApplication.instance() is None:
                return
        except ImportError:
            return
        self._heartbeat = QTimer()
        self._heartbeat.setInterval(50)
        self._last_beat = time.perf_counter()

        def beat():
            now = time.perf_counter()
            gap = now - self._last_beat
            self._last_beat = now
            if gap > 0.1:
                with self._lock:
                    self._events.append({
                        "phase": "qt_heartbeat_gap",
                        "start": now - gap,
                        "seconds": gap,
                        "thread": threading.get_ident(),
                    })

        self._heartbeat.timeout.connect(beat)
        self._heartbeat.start()

    def save(self):
        if self.path is None:
            return
        if self._heartbeat is not None:
            self._heartbeat.stop()
        with self._lock:
            events = list(self._events)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"events": events}, indent=2))


def trace_interaction(name):
    """Time an Animation controller method when tracing is enabled."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with self._trace.timer(name):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate
