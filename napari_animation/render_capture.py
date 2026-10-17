"""Capture export frames at a requested pixel size and matching camera scale."""

from contextlib import ExitStack
from math import ceil

from .native_clipping import sync_native_clipping
from .perf import render_phase


def _exact_frame(frame, render_size, rounding=0):
    """Trim only device-pixel rounding at the edges; never resample a frame."""
    height, width = render_size
    extra_height, extra_width = (
        frame.shape[0] - height,
        frame.shape[1] - width,
    )
    if not (0 <= extra_height <= rounding and 0 <= extra_width <= rounding):
        raise RuntimeError(
            f"Requested a {width} × {height} render, but the canvas returned "
            f"{frame.shape[1]} × {frame.shape[0]} pixels."
        )
    if not (extra_height or extra_width):
        return frame
    top, left = extra_height // 2, extra_width // 2
    return frame[top : top + height, left : left + width].copy()


class ExportCaptureSession:
    """Hold the native export viewport across frames, restoring it once.

    Frame states still use the editor's authored camera zoom. Each capture
    temporarily scales that zoom against the original physical canvas size.
    Slicer synchronization remains per frame so newly created layers are
    included, and Qt callbacks never run under a persistent force_sync scope.
    """

    def __init__(self, viewer, render_size=None, canvas_only=True):
        self.viewer = viewer
        self.render_size = render_size
        self.canvas_only = canvas_only
        self.native = None
        self._paused = None
        self._closed = False

    def _pause_canvas_repaints(self, window):
        """Stop on-screen repaints; each capture renders the canvas itself.

        Every applied frame schedules a Qt repaint of the canvas, and the
        screenshot then renders the same scene again offscreen. For large
        volumes each render is a substantial GPU pass, so the on-screen one
        doubled the cost of a frame. A full-window capture needs the widgets
        painted and keeps them.
        """
        native = getattr(getattr(window, "_qt_viewer", None), "canvas", None)
        native = getattr(native, "native", None)
        if not self.canvas_only or native is None:
            return
        if not hasattr(native, "setUpdatesEnabled"):
            return
        if native.updatesEnabled():
            native.setUpdatesEnabled(False)
            self._paused = native

    def _resume_canvas_repaints(self):
        native, self._paused = self._paused, None
        if native is not None:
            native.setUpdatesEnabled(True)
            native.update()

    def __enter__(self):
        window = getattr(self.viewer, "window", None)
        if self.render_size is None or window is None:
            if window is not None:
                self._pause_canvas_repaints(window)
            return self
        if (
            getattr(self.viewer, "_napari_animation_capture_session", None)
            is not None
        ):
            raise RuntimeError("An export capture session is already active")
        self.canvas = window._qt_viewer.canvas
        native = self.canvas.native
        self.old_size = (native.width(), native.height())
        self.ratio = float(native.devicePixelRatioF())
        if min(self.old_size) <= 0 or self.ratio <= 0:
            raise RuntimeError(
                "Cannot export from a canvas with an empty size."
            )
        self.native = native
        self.viewer._napari_animation_capture_session = self
        self._pause_canvas_repaints(window)
        # Defer resizing until the first capture, after applying its state.
        return self

    def close(self):
        self._resume_canvas_repaints()
        if self._closed or self.native is None:
            return
        self._closed = True
        delattr(self.viewer, "_napari_animation_capture_session")
        zoom, center = self.viewer.camera.zoom, self.viewer.camera.center
        with render_phase("capture.restore_session"):
            try:
                self.native.resize(*self.old_size)
            finally:
                self.viewer.camera.center = center
                self.viewer.camera.zoom = zoom
                self.canvas.on_draw(None)

    def __exit__(self, *exc):
        self.close()


def capture_export_frame(viewer, *, canvas_only=True, render_size=None):
    """Render at (height, width), awaiting data for the export viewport.

    Qt widget dimensions are logical pixels. Round them up using the native
    widget's floating-point device-pixel ratio, then trim the excess physical
    edge pixels. This avoids napari's version-dependent screenshot(size=...)
    conversion and preserves the actual source samples at fractional scaling.
    """
    sync_native_clipping(viewer)
    if render_size is None:
        with render_phase("capture.framebuffer"):
            return viewer.screenshot(canvas_only=canvas_only)
    if not canvas_only:
        raise ValueError("Render size requires canvas_only=True.")

    window = getattr(viewer, "window", None)
    if window is None:
        # Non-Qt viewer adapters own their rendering and must honor this size.
        return _exact_frame(
            viewer.screenshot(canvas_only=True, size=render_size, flash=False),
            render_size,
        )

    canvas = window._qt_viewer.canvas
    native = canvas.native
    session = getattr(viewer, "_napari_animation_capture_session", None)
    if session is not None and session.render_size != render_size:
        raise ValueError("Capture size differs from the active export session")
    old_size = (
        session.old_size if session else (native.width(), native.height())
    )
    ratio = float(native.devicePixelRatioF())
    if session is not None and ratio != session.ratio:
        raise RuntimeError(
            "Display scaling changed during export; restart the export."
        )
    if min(old_size) <= 0 or ratio <= 0:
        raise RuntimeError("Cannot export from a canvas with an empty size.")
    height, width = render_size
    old_zoom, old_center = viewer.camera.zoom, viewer.camera.center
    scale = min(
        width / (old_size[0] * ratio),
        height / (old_size[1] * ratio),
    )
    slicer = getattr(viewer, "_layer_slicer", None)
    with ExitStack() as stack:
        if slicer is not None:
            stack.enter_context(slicer.force_sync())
            with render_phase("capture.slice_wait"):
                slicer.wait_until_idle(timeout=5)

            def materialize(layer):
                with render_phase("capture.materialize"):
                    slicer.submit(layers=[layer], dims=viewer.dims, force=True)
                    # Some napari versions update slice input but skip reading
                    # data on the synchronous force=True path. Materialize and
                    # publish the actual slice before the framebuffer is read.
                    layer.set_view_slice()
                    layer.events.set_data()

            def reload_synchronously(event):
                materialize(event.layer)

            for layer in viewer.layers:
                if not layer.visible:
                    continue
                stack.enter_context(
                    layer.events.reload.blocker(viewer._on_layer_reload)
                )
                layer.events.reload.connect(reload_synchronously)
                stack.callback(
                    layer.events.reload.disconnect, reload_synchronously
                )
        try:
            target_size = (ceil(width / ratio), ceil(height / ratio))
            with render_phase("capture.resize"):
                if (native.width(), native.height()) != target_size:
                    native.resize(*target_size)
            with render_phase("capture.camera_scale"):
                viewer.camera.zoom = old_zoom * scale
            # This draw updates multiscale levels/tiles for the target view.
            # Any reload it triggers must finish before reading the framebuffer.
            with render_phase("capture.draw"):
                canvas.on_draw(None)
            if slicer is not None:
                pending = [
                    layer
                    for layer in viewer.layers
                    if layer.visible and not layer.loaded
                ]
                if pending:
                    # A completed async future can still have its Qt delivery
                    # queued. Apply these slices synchronously, without pumping
                    # the event loop or accepting edits during capture.
                    for layer in pending:
                        materialize(layer)
            with render_phase("capture.framebuffer"):
                frame = viewer.screenshot(canvas_only=True, flash=False)
            return _exact_frame(frame, render_size, rounding=ceil(ratio))
        finally:
            with render_phase("capture.restore_viewport"):
                if session is None:
                    native.resize(*old_size)
                viewer.camera.center = old_center
                viewer.camera.zoom = old_zoom
                if session is None:
                    canvas.on_draw(None)
