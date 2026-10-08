"""Event-loop-driven movie export for the Qt Save button.

State application and screenshots stay on the GUI thread.  A bounded queue
hands resize, diagnostics, encoding and container finalization to one worker.
No queue wait or thread join runs inside a Qt callback.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import imageio
from qtpy.QtCore import QObject, QTimer, Signal, Qt
from qtpy.QtWidgets import QProgressDialog

from ..diagnostics import RenderDiagnostics
from ..perf import PerfLogger
from ..frame_sequence import FrameSequence
from ..key_frame import KeyFrameList
from ..ortho_slicer import OrthoSlicer
from ..prefetch import SlicePrefetcher, dask_cache_context
from ..render_capture import ExportCaptureSession, capture_export_frame
from ..scene.processing import service_for_viewer
from ..viewer_state import ViewerState
from ..video_padding import pad_video_frame

logger = logging.getLogger("napari_animation")


class QtExportController(QObject):
    """Export one frame per event-loop turn, waiting for matching data."""

    finished = Signal(object, object)  # output path, error

    def __init__(self, animation, options, parent=None):
        super().__init__(parent)
        self.animation = animation
        self.options = dict(options)
        self._perf = PerfLogger(enabled=self.options.get("perf_log", True))
        self._backpressure_start = None
        self._data_wait_start = None
        self._queue = queue.Queue(maxsize=4)
        self._writer = None
        self._writer_thread = None
        self._write_error = None
        self._diagnostic_report = ""
        self._pending_frame = None
        self._index = 0
        self._cancelled = False
        self._generation = 0
        self._finalizing = False
        self._done = False
        self._cache_cm = None
        self._capture_session = None
        self._prefetcher = None
        self._progress = None
        self._diagnostics = RenderDiagnostics(
            enabled=bool(self.options.get("diagnose", True))
        )

    def start(self):
        render_size = self.options.get("render_size")
        if render_size is not None:
            if not self.options.get("canvas_only", True):
                raise ValueError("Render size requires Canvas only.")
            if len(render_size) != 2 or any(value <= 0 for value in render_size):
                raise ValueError("Render size must be (height, width) with positive values.")
        animation = self.animation
        self._perf.start()
        self._perf.metadata.update(self.options, export_path="qt")
        animation._last_render_perf = self._perf
        animation._validate_animation()
        # Copy only authored states, then interpolate lazily with the normal
        # bounded cache. Editing the live timeline cannot change this export.
        keyframes = KeyFrameList()
        keyframes.extend(
            replace(frame, viewer_state=deepcopy(frame.viewer_state))
            for frame in animation.key_frames
        )
        self._frames = FrameSequence(keyframes)
        self._frames.state_interpolation_map = dict(animation._frames.state_interpolation_map)
        self._total = len(self._frames)
        self._path = Path(self.options["path"])
        self._folder = self._path.absolute().parent / self._path.stem
        self._png = not bool(self._path.suffix)
        self._working_ortho, self._working_scene, self._working_view = (
            animation._capture_state()
        )
        self._working_state = ViewerState.from_viewer(
            animation.viewer,
            ortho=self._working_ortho,
            scene=self._working_scene,
            view=self._working_view,
        )
        self._source_bindings = self._viewer_sources()
        self._cache_cm = dask_cache_context(self.options.get("dask_cache", "auto"))
        self._cache_cm.__enter__()
        self._prefetcher = SlicePrefetcher(
            animation.viewer, self._frames,
            depth=self.options.get("prefetch", 4),
            workers=self.options.get("prefetch_workers", 4),
        )
        self._perf.metadata.update(
            prefetch_enabled=self._prefetcher.enabled,
            prefetch_skip_reason=self._prefetcher.skip_reason,
        )
        self._writer_thread = threading.Thread(
            target=self._consume, name="napari-animation-qt-writer", daemon=True
        )
        self._previous_exact = getattr(animation.viewer, "_napari_animation_export_exact", False)
        animation.viewer._napari_animation_export_exact = True
        service_for_viewer(animation.viewer).retain_sections(set())
        window = getattr(animation.viewer, "window", None)
        qt_window = getattr(window, "_qt_window", None)
        if qt_window is not None:
            qt_viewer = getattr(window, "_qt_viewer", None)
            if qt_viewer is not None:
                qt_viewer.dims.stop()
            self._progress = QProgressDialog(
                "Exporting animation…", "Cancel", 0, self._total, qt_window
            )
            self._progress.setWindowTitle("Export animation")
            self._progress.setWindowModality(Qt.WindowModal)
            self._progress.setAutoClose(False)
            self._progress.setAutoReset(False)
            self._progress.canceled.connect(self.cancel)
            self._progress.show()
        self._capture_session = ExportCaptureSession(animation.viewer, render_size)
        self._capture_session.__enter__()
        self._writer_thread.start()
        QTimer.singleShot(0, self._step)

    def _open_writer(self):
        from ..animation import COMPAT_ENCODER, H264_SUFFIXES, VIDEO_SUFFIXES
        from ..animation import _compat_output_params

        suffix = self._path.suffix.lower()
        fps = self.options.get("fps", 20)
        quality = self.options.get("quality", 5)
        fmt = self.options.get("format")
        if suffix in VIDEO_SUFFIXES:
            if self.options.get("compat", True) and suffix in H264_SUFFIXES:
                kwargs = {"fps": fps, **COMPAT_ENCODER}
                kwargs["output_params"] = _compat_output_params(quality)
            else:
                kwargs = {"fps": fps, "quality": quality}
            if self.options.get("render_size") is not None:
                kwargs["macro_block_size"] = 2
        else:
            kwargs = {"fps": fps}
        return imageio.get_writer(str(self._path), format=fmt, **kwargs)

    def _consume(self):
        from imageio.v2 import imwrite
        from ..animation import VIDEO_SUFFIXES

        pad_for_video = self._path.suffix.lower() in VIDEO_SUFFIXES

        try:
            with self._perf.timer("diagnostics.environment"):
                self._diagnostics.describe_environment()
            if not self._png:
                try:
                    with self._perf.timer("writer.open"):
                        self._writer = self._open_writer()
                except Exception as error:  # documented PNG fallback
                    logger.warning(
                        "Video writer unavailable (%s); saving PNGs", error
                    )
                    self._png = True
            if self._png:
                self._folder.mkdir(parents=True, exist_ok=True)
                for old in self._folder.glob("*.png"):
                    old.unlink()
            while True:
                with self._perf.timer("writer.queue_wait"):
                    item = self._queue.get()
                try:
                    if item is None:
                        break
                    index, frame = item
                    scale = self.options.get("scale_factor")
                    if scale not in (None, 1):
                        from scipy import ndimage as ndi
                        import numpy as np

                        with self._perf.timer("scale", frame=index):
                            frame = ndi.zoom(
                                frame, (scale, scale, 1)
                            ).astype(np.uint8)
                    with self._perf.timer("diagnostics.frame", frame=index):
                        self._diagnostics.record_frame(index, frame)
                    with self._perf.timer("encode/write", frame=index):
                        if self._writer is not None:
                            if pad_for_video:
                                block_size = 2 if self.options.get("render_size") else 16
                                frame = pad_video_frame(frame, block_size=block_size)
                            self._writer.append_data(frame)
                        else:
                            imwrite(
                                self._folder / f"{self._path.stem}_{index}.png",
                                frame,
                            )
                    self._perf.count("frames_written")
                finally:
                    self._queue.task_done()
        except Exception as error:  # handed back by the timer-driven GUI poll
            self._write_error = error
        finally:
            if self._writer is not None:
                try:
                    with self._perf.timer("finalize"):
                        self._writer.close()
                except Exception as error:
                    if self._write_error is None:
                        self._write_error = error
            if self._write_error is None and not self._cancelled:
                try:
                    if not self._png:
                        # Verification decodes the whole video and launches
                        # ffmpeg. Keep it with encoding, off the Qt thread.
                        with self._perf.timer("diagnostics.verify"):
                            self._diagnostics.verify_output(
                                str(self._path), self._total
                            )
                    self._diagnostic_report = self._diagnostics.report()
                except Exception as error:  # diagnostics cannot fail export
                    logger.warning("Export diagnostics failed: %s", error)

    def _later(self, callback, delay=0):
        QTimer.singleShot(delay, callback)

    def _viewer_signature(self):
        from ..io import _to_builtin

        state = ViewerState.from_viewer(self.animation.viewer)
        return _to_builtin((state.camera, state.dims, state.layers))

    def _viewer_sources(self):
        from ..layer_registry import OBJECT_ID_METADATA

        return tuple(
            (id(layer), id(layer.data)) for layer in self.animation.viewer.layers
            if not layer.metadata.get(OBJECT_ID_METADATA)
        )

    def _validate_sources(self):
        if self._viewer_sources() != self._source_bindings:
            raise RuntimeError("Source layers changed during export; restart the export.")

    def _apply_frame(self):
        self._validate_sources()
        with self._perf.timer("interpolate", frame=self._index):
            state = self._frames[self._index]
        with self._perf.timer("apply", frame=self._index):
            state.apply(self.animation.viewer)
        with self._perf.timer("state.signature", frame=self._index):
            self._applied_signature = self._viewer_signature()

    def _await_capture(self, generation):
        self._data_wait_start = time.perf_counter()
        service_for_viewer(self.animation.viewer).when_idle(
            lambda: self._later(lambda: self._capture(generation), 0)
        )

    def _step(self):
        if self._done:
            return
        if self._cancelled or self._write_error is not None:
            self._finish(self._write_error or RuntimeError("Export cancelled"))
            return
        if self._pending_frame is not None:
            try:
                self._queue.put_nowait(self._pending_frame)
            except queue.Full:
                if self._backpressure_start is None:
                    self._backpressure_start = time.perf_counter()
                self._perf.count("queue.full_polls")
                self._later(self._step, 15)
                return
            if self._backpressure_start is not None:
                self._perf.record("queue.backpressure", time.perf_counter() - self._backpressure_start,
                                  frame=self._index, start=self._backpressure_start)
                self._backpressure_start = None
            self._pending_frame = None
            self._index += 1
            if self._progress is not None:
                self._progress.setValue(self._index)
                # Modal progress updates can process a queued Cancel event.
                if self._done or self._cancelled:
                    return
        if self._index >= self._total:
            self._end_writer()
            return
        with self._perf.timer("prefetch.schedule", frame=self._index):
            self._prefetcher.advance(self._index)
        self._generation += 1
        generation = self._generation
        try:
            self._apply_frame()
        except Exception as error:
            self._finish(error)
            return
        self._await_capture(generation)

    def _capture(self, generation):
        if self._done or self._cancelled or generation != self._generation:
            return
        if self._write_error is not None:
            self._finish(self._write_error)
            return
        # A queued editor callback or script can still change model state
        # while the modal progress window is open. Restore and await this
        # frame again before capturing any such change.
        try:
            self._validate_sources()
            with self._perf.timer("state.signature", frame=self._index):
                signature = self._viewer_signature()
            if signature != self._applied_signature:
                self._perf.count("frame.reapplied")
                self._apply_frame()
                self._await_capture(generation)
                return
        except Exception as error:
            self._finish(error)
            return
        service = service_for_viewer(self.animation.viewer)
        if service.last_error is not None:
            self._finish(service.last_error)
            return
        # Native napari async slicing is separate from section projection.
        # Wait for the visible layer requests before taking this frame.
        if any(
            not getattr(layer, "loaded", True)
            for layer in self.animation.viewer.layers if layer.visible
        ):
            self._perf.count("data.not_loaded_polls")
            self._later(lambda: self._capture(generation), 15)
            return
        if self._data_wait_start is not None:
            self._perf.record("data.await_ready", time.perf_counter() - self._data_wait_start,
                              frame=self._index, start=self._data_wait_start)
            self._data_wait_start = None
        try:
            with self._perf.timer("screenshot", frame=self._index):
                frame = capture_export_frame(
                    self.animation.viewer,
                    canvas_only=self.options.get("canvas_only", True),
                    render_size=self.options.get("render_size"),
                )
        except Exception as error:
            self._finish(error)
            return
        self._perf.count("frames_captured")
        self._pending_frame = (self._index, frame)
        self._later(self._step)

    def _end_writer(self):
        if self._done:
            return
        if self._write_error is not None:
            self._finish(self._write_error)
            return
        if not self._finalizing:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                self._later(self._end_writer, 15)
                return
            self._finalizing = True
        if self._writer_thread.is_alive():
            self._later(self._end_writer, 20)
            return
        self._finish(self._write_error)

    def cancel(self):
        self._cancelled = True
        self._generation += 1
        if self._prefetcher is not None:
            self._prefetcher.shutdown()
        # The worker can be inside FFmpeg; let the normal timer poll finalize
        # without blocking the window-close or cancel callback.
        self._later(self._step)

    def _finish(self, error):
        if self._done:
            return
        if self._finalizing is False:
            # On an early failure, stop producing and let the writer close.
            self._cancelled = True
            self._generation += 1
            if self._writer_thread is None or not self._writer_thread.is_alive():
                self._finalizing = True
            else:
                try:
                    self._queue.put_nowait(None)
                    self._finalizing = True
                except queue.Full:
                    self._later(lambda: self._finish(error), 15)
                    return
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._later(lambda: self._finish(error), 20)
            return
        self._done = True
        if self._prefetcher is not None:
            self._prefetcher.shutdown()
            self._prefetcher = None
        if self._cache_cm is not None:
            self._cache_cm.__exit__(None, None, None)
            self._cache_cm = None
        if self._capture_session is not None:
            try:
                with self._perf.timer("capture.session_cleanup"):
                    self._capture_session.close()
            except Exception as capture_error:
                if error is None:
                    error = capture_error
            self._capture_session = None
        self.animation.viewer._napari_animation_export_exact = self._previous_exact
        try:
            with self._perf.timer("restore"):
                self._working_state.apply(self.animation.viewer)
                self.animation.scene.restore_snapshot(
                    self.animation.viewer, self._working_scene
                )
                self.animation.ortho_slicer = OrthoSlicer.from_dict(
                    self._working_ortho
                )
                active = (self._working_view or {}).get("active_section_id")
                self.animation.active_section_id = active
                self.animation.renderer.active_section_id = active
                self.animation.renderer.apply_working(ortho=self._working_ortho)
                self.animation.viewer.camera.update(self._working_state.camera)
        except Exception as restore_error:
            if error is None:
                error = restore_error
        output = self._folder if self._png else self._path
        if error is not None and str(error) != "Export cancelled":
            logger.error(
                "Export failed at frame %s/%s (%s)",
                self._index + 1, self._total, output,
                exc_info=(type(error), error, error.__traceback__),
            )
        if error is None and self._diagnostic_report:
            logger.info("%s", self._diagnostic_report)
        if self._progress is not None:
            self._progress.close()
            self._progress.deleteLater()
            self._progress = None
        self._perf.metadata.update(status="complete" if error is None else "failed",
                                   error=str(error) if error else None,
                                   expected_frames=self._total,
                                   output_shape=self._diagnostics.first_shape,
                                   blank_frames=list(self._diagnostics.blank_frames),
                                   sampled_pixel_range=list(self._diagnostics.value_range))
        self._perf.stop()
        if self._perf.enabled:
            report = self._perf.log_report()
            if self._diagnostic_report:
                report = self._diagnostic_report + "\n\n" + report
            self.animation._write_render_log(output, self._total, report)
            self._perf.save(output)
        self.finished.emit(output, error)
