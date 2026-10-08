import logging
import os
import queue
import threading
from dataclasses import replace
from itertools import count
from pathlib import Path

import imageio
import numpy as np
from napari.utils.io import imsave
from napari.utils.events import EmitterGroup
from tqdm import tqdm

from .diagnostics import RenderDiagnostics
from .easing import Easing
from .frame_sequence import FrameSequence
from .key_frame import KeyFrame, KeyFrameList
from .layer_registry import LayerRegistry
from .ortho_slicer import OrthoSlicer, order_for_view
from .perf import InteractionTrace, PerfLogger, trace_interaction
from .prefetch import (
    SlicePrefetcher, acquire_interactive_cache, dask_cache_context,
    release_interactive_cache,
)
from .render_capture import ExportCaptureSession, capture_export_frame
from .scene import Scene
from .scene.renderer import SceneRenderer
from .viewer_state import ViewerState
from .video_padding import pad_video_frame

logger = logging.getLogger("napari_animation")

#: File extensions handled by the imageio-ffmpeg video writer.
VIDEO_SUFFIXES = (".mov", ".avi", ".mpg", ".mpeg", ".mp4", ".mkv", ".wmv")

#: Containers for which H.264 is the sensible codec, and where forcing the
#: most widely supported settings is worth doing.
H264_SUFFIXES = (".mp4", ".mov", ".mkv")

#: Encoder settings chosen for playback compatibility rather than for size.
#:
#: * ``yuv420p`` because OS-bundled decoders reject 4:2:2 and 4:4:4 H.264.
#: * ``-profile:v high`` because setting the pixel format is *not* sufficient:
#:   some ffmpeg builds still select the "High 4:4:4 Predictive" profile for a
#:   yuv420p stream, and Windows Media Player and Photos cannot decode it. The
#:   profile has to be pinned explicitly.
#: * ``-crf`` because ``-qscale:v`` -- what imageio's ``quality`` argument
#:   becomes -- is an MJPEG-era control that libx264 honours only loosely, and
#:   is implicated in the odd profile selection above. CRF is x264's intended
#:   rate control.
#: * ``+faststart`` moves the moov atom to the front of the file, without which
#:   many players, and every Explorer/Finder preview, refuse to open it.
COMPAT_ENCODER = {
    "codec": "libx264",
    "pixelformat": "yuv420p",
}


def _compat_output_params(quality):
    """ffmpeg parameters for a maximally playable H.264 stream."""
    # imageio quality is 0-10 (higher is better); CRF is 0-51 (lower is
    # better). Map so the default quality of 5 lands on a visually lossless
    # CRF of 20.
    crf = int(round(30 - 2 * float(quality)))
    crf = max(0, min(51, crf))
    return [
        "-profile:v",
        "high",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
    ]


class Animation:
    """Make animations using the napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        napari viewer.

    Attributes
    ----------
    key_frames : list of KeyFrame
        List of key-frames in the animation.
    """

    def __init__(self, viewer):
        from ._dask_cache import protect_napari_cache

        protect_napari_cache()
        self.viewer = viewer
        self._closed = False
        self._viewer_destroyed = None
        self._trace = InteractionTrace.from_environment()
        self._trace.start_heartbeat()
        viewer._napari_animation_trace = self._trace
        try:
            from qtpy.QtWidgets import QApplication

            gui_viewer = (
                QApplication.instance() is not None and hasattr(viewer, "window")
            )
        except ImportError:
            gui_viewer = False
        self._interactive_cache_acquired = (
            acquire_interactive_cache() if gui_viewer else False
        )
        self.layer_registry = LayerRegistry(viewer)

        self.key_frames = KeyFrameList()

        self.key_frames.events.removed.connect(self._on_keyframe_removed)

        self._keyframe_counter = count()  # track number of frames created

        self._frames = FrameSequence(self.key_frames)

        # FrameSequence must receive structural list events before the
        # selection callback below.  A replacement emits ``changed`` while the
        # list is mutating; selecting the replacement before rebuilding the
        # frame index makes navigation briefly point at the removed object.
        self.key_frames.events.changed.connect(self._on_keyframe_changed)

        self.key_frames.selection.events.active.connect(
            self._on_active_keyframe_changed
        )

        # Imaris-style optical-section slicer; its parameters are captured into
        # each keyframe so the optical section can be animated.
        self.ortho_slicer = OrthoSlicer()

        # Clipping planes and other 3D scene objects. Their parameters are
        # captured into every keyframe, so each can be reshaped and switched
        # on or off independently at each keyframe.
        self.scene = Scene()
        self.renderer = SceneRenderer(
            viewer, scene=self.scene, registry=self.layer_registry
        )
        self.active_section_id = None
        self._navigating = False
        self._last_navigation_selection = None
        self._current_frame = None
        self._last_dims_point = tuple(viewer.dims.point)
        self.events = EmitterGroup(source=self, state_applied=None)
        from .resolution import ResolutionSettings

        self.resolution = ResolutionSettings(self)
        emitter = getattr(self.viewer.dims.events, "current_step", None)
        self._dims_emitter = emitter
        if emitter is not None:
            emitter.connect(self._on_dims_point_changed)
        if gui_viewer:
            qt_window = getattr(getattr(viewer, "window", None), "_qt_window", None)
            if qt_window is not None:
                self._viewer_destroyed = qt_window.destroyed
                self._viewer_destroyed.connect(self.close)

    def _active_section(self, object_id=None):
        wanted = self.active_section_id if object_id is None else object_id
        if wanted is not None:
            return next(
                (
                    obj
                    for obj in self.scene
                    if obj.id == wanted and obj.kind == "ortho_slice"
                ),
                None,
            )
        return next(
            (
                obj
                for obj in self.scene
                if obj.kind == "ortho_slice" and obj.enabled
            ),
            None,
        )

    def _section_order(self, section):
        world_axis = int(np.argmax(np.abs(np.asarray(section.normal, dtype=float))))
        axis = self.viewer.dims.ndim - 3 + world_axis
        current = list(self.viewer.dims.order)
        current.remove(axis)
        leading = [value for value in current if value < self.viewer.dims.ndim - 3]
        spatial = [value for value in current if value >= self.viewer.dims.ndim - 3]
        return tuple(leading + [axis] + spatial)

    def _sync_active_section_from_dims(self):
        section = self._active_section()
        if section is None or not section.enabled or self.viewer.dims.ndisplay >= 3:
            return
        world_axis = int(np.argmax(np.abs(np.asarray(section.normal, dtype=float))))
        global_axis = self.viewer.dims.ndim - 3 + world_axis
        position = list(section.position)
        position[world_axis] = float(self.viewer.dims.point[global_axis])
        section.position = tuple(position)

    def _on_dims_point_changed(self, event=None):
        previous = self._last_dims_point
        current = tuple(self.viewer.dims.point)
        self._last_dims_point = current
        if self.renderer.applying:
            return
        section = self._active_section()
        if section is None:
            return
        world_axis = int(np.argmax(np.abs(np.asarray(section.normal, dtype=float))))
        axis = self.viewer.dims.ndim - 3 + world_axis
        if len(previous) == len(current) and previous[axis] != current[axis]:
            self._sync_active_section_from_dims()
        self.renderer.apply_working(
            ortho=self.ortho_slicer.to_dict() if self.ortho_slicer.enabled else None
        )

    def _capture_state(self):
        """Current ortho-slicer and scene parameters to record on a keyframe."""
        ortho = (
            self.ortho_slicer.to_dict() if self.ortho_slicer.enabled else None
        )
        # objects backed by a real layer read their geometry back out of it
        # first, so a plane repositioned outside this plugin is captured as
        # the user sees it
        self.scene.sync_from_viewer(self.viewer)
        scene = self.scene.to_dict() if len(self.scene) else None
        view = {"active_section_id": self.active_section_id}
        return ortho, scene, view

    def remove_scene_object(self, scene_object):
        """Remove a scene object, taking any layer it owns with it.

        The object's parameters are left in already-captured keyframes; they
        are simply ignored, so removing an object cannot corrupt an animation
        that was built around it.
        """
        scene_object.remove_layer(self.viewer)
        self.scene.remove(scene_object)
        if self.active_section_id == scene_object.id:
            self.active_section_id = None
            self.renderer.active_section_id = None
        ortho, _, _ = self._capture_state()
        self.renderer.apply_working(ortho=ortho)

    def add_scene_object(self, scene_object, backfill: bool = True):
        """Add a scene object (e.g. a clipping plane) to the animation.

        Parameters
        ----------
        scene_object : SceneObject
            The object to add.
        backfill : bool
            Whether to record the object in the keyframes captured so far. By
            default it is, so a newly added clipping plane is visible across
            the whole animation and the user switches it *off* where it is not
            wanted -- rather than it silently disappearing whenever an older
            keyframe is selected.

        Returns
        -------
        SceneObject
            The object that was added.
        """
        proposed = self.scene.to_dict()
        if scene_object.id in proposed:
            raise ValueError(f"Scene object ID {scene_object.id!r} already exists")
        if scene_object.kind == "ortho_slice":
            # Resolve identity before copying this object into older frames.
            scene_object.source_layer(self.viewer)
        proposed[scene_object.id] = scene_object.to_dict()
        self.renderer.preflight(proposed)
        self.scene.append(scene_object)
        if backfill:
            params = scene_object.to_dict()
            for key_frame in self.key_frames:
                state = key_frame.viewer_state
                scene = dict(state.scene or {})
                scene[scene_object.id] = dict(params)
                key_frame.viewer_state = replace(state, scene=scene)
            self._frames._rebuild_frame_index()
        ortho, _, _ = self._capture_state()
        if self.scene:
            self.renderer.apply_working(ortho=ortho)
        return scene_object

    def set_object_enabled(
        self, keyframe_index: int, object_id: str, enabled: bool
    ):
        """Switch a scene object on or off at a single keyframe.

        This is the per-keyframe toggle: the object's geometry is untouched,
        only whether it contributes to that keyframe.
        """
        key_frame = self.key_frames[keyframe_index]
        state = key_frame.viewer_state
        scene = {k: dict(v) for k, v in (state.scene or {}).items()}
        if object_id not in scene:
            raise KeyError(
                f"No scene object {object_id!r} in keyframe {keyframe_index}"
            )
        scene[object_id]["enabled"] = bool(enabled)
        key_frame.viewer_state = replace(state, scene=scene)
        # the interpolation cache holds states built from the old value
        self._frames._rebuild_frame_index()

    def update_scene_object(self, object_id: str, **changes):
        """Update one live scene object and render the working state.

        Qt editors and scripts can use this command instead of mutating an
        object and separately remembering to repaint.  The object remains the
        canonical editable instance; captured keyframes are unchanged until
        :meth:`capture_keyframe` or :meth:`overwrite_keyframe` is called.
        """
        scene_object = next(
            (candidate for candidate in self.scene if candidate.id == object_id),
            None,
        )
        if scene_object is None:
            raise KeyError(f"Unknown scene object {object_id!r}")
        if "id" in changes or "kind" in changes:
            raise ValueError("scene object identity and kind are immutable")
        valid = set(scene_object.__dataclass_fields__) - {"id"}
        unknown = set(changes) - valid
        if unknown:
            raise AttributeError(
                f"Unknown {scene_object.kind} field(s): {sorted(unknown)!r}"
            )
        if "source" in changes and "source_id" not in changes:
            source_name = changes["source"]
            source_layer = (
                self.viewer.layers[source_name]
                if source_name and source_name in self.viewer.layers
                else None
            )
            changes["source_id"] = (
                self.layer_registry.source_id(source_layer)
                if source_layer is not None
                else None
            )
        # Validate a detached candidate before mutating the live object or
        # touching layers. A rejected editor value must remain editable.
        changes = {
            name: tuple(value) if name == "targets" else value
            for name, value in changes.items()
        }
        candidate = self.scene.to_dict()
        candidate[object_id].update(changes)
        self.renderer.preflight(candidate)
        for name, value in changes.items():
            if name == "targets":
                value = tuple(value)
            setattr(scene_object, name, value)
        self.apply_working_state()
        return scene_object

    def set_active_section(self, object_id=None):
        """Choose which enabled section owns the true 2D canvas."""
        obj = None
        if object_id is not None:
            obj = next(
                (candidate for candidate in self.scene if candidate.id == object_id),
                None,
            )
            if obj is None or obj.kind != "ortho_slice":
                raise KeyError(f"Unknown ortho section {object_id!r}")
        old_id = self.active_section_id
        old_order = tuple(self.viewer.dims.order)
        with self.renderer.applying_state():
            return self._set_active_section_guarded(object_id, obj, old_id, old_order)

    def _set_active_section_guarded(self, object_id, obj, old_id, old_order):
        try:
            self.renderer.preflight(
                ndisplay=self.viewer.dims.ndisplay,
                active_section_id=object_id,
            )
            if obj is not None and self.viewer.dims.ndisplay < 3:
                self.viewer.dims.order = self._section_order(obj)
            self.active_section_id = object_id
            self.renderer.active_section_id = object_id
            self.apply_working_state()
        except Exception:
            self.viewer.dims.order = old_order
            self.active_section_id = old_id
            self.renderer.active_section_id = old_id
            raise
        return object_id

    def set_keyframe_timing(self, keyframe_index, *, steps=None, ease=None):
        """Update one keyframe's timing and invalidate evaluated frames.

        Timing values are part of the timeline structure rather than viewer
        state. Keeping this mutation in ``Animation`` prevents the Qt widgets
        from changing a keyframe without rebuilding the frame index.
        """
        if not isinstance(keyframe_index, int):
            raise TypeError("keyframe_index must be an integer")
        key_frame = self.key_frames[keyframe_index]
        if steps is not None:
            if isinstance(steps, bool) or not isinstance(steps, int):
                raise TypeError("steps must be an integer")
            if steps < 1:
                raise ValueError("steps must be at least 1")
            key_frame.steps = steps
        if ease is not None:
            if not isinstance(ease, Easing):
                raise TypeError("ease must be an Easing member")
            key_frame.ease = ease
        self._frames._rebuild_frame_index()
        return key_frame

    # ------------------------------------------------- view mode and sweeps

    def fit_view(self):
        """Fit the camera to the data, across napari versions."""
        for name in ("fit_to_view", "reset_view"):
            method = getattr(self.viewer, name, None)
            if callable(method):
                method()
                return

    @trace_interaction("apply_working_state")
    def apply_working_state(self) -> None:
        """Apply the editable scene and ortho controls as one composition.

        Qt controls call this helper instead of invoking the legacy ortho
        slicer directly.  The compositor therefore remains the sole writer
        of clipping planes and an interactive edit cannot erase scene-owned
        planes.
        """
        with self.renderer.applying_state():
            self._apply_working_state_guarded()

    def _apply_working_state_guarded(self):
        self.renderer.preflight()
        ortho = (
            self.ortho_slicer.to_dict() if self.ortho_slicer.enabled else None
        )
        self.ortho_slicer.apply_state(
            self.viewer,
            ortho,
            # With no scene objects the legacy slicer is the only owner of
            # clipping planes. Once a scene exists, the compositor folds its
            # contribution in with user-authored planes.
            assign_clipping_planes=not bool(self.scene),
        )
        # An empty scene has no compositor state to apply; leave any planes
        # owned by the legacy slicer or another plugin untouched.
        if self.scene:
            self.renderer.apply_working(ortho=ortho)
        else:
            self.renderer.restore_view_overrides()
            self.renderer._reconcile([], None)

    @trace_interaction("snap_2d")
    def snap_2d(self, view="XY", capture=True, steps=15, ease=Easing.LINEAR):
        """Snap the canvas to a flat 2D view of a named plane.

        Parameters
        ----------
        view : str
            ``"XY"``, ``"XZ"`` or ``"YZ"``. The remaining axis becomes the
            dims slider, which is what a scroll-through then sweeps.
        capture : bool
            Whether to record a keyframe once snapped.
        """
        with self.renderer.applying_state():
            self._snap_2d_guarded(view)
        if capture:
            self.capture_keyframe(steps=steps, ease=ease)

    def _snap_2d_guarded(self, view):
        old_dims = self.viewer.dims.dict()
        old_active = self.active_section_id
        section = self._active_section()
        active_id = section.id if section is not None else self.active_section_id
        try:
            self.renderer.preflight(ndisplay=2, active_section_id=active_id)
            self.viewer.dims.order = (
                self._section_order(section)
                if section is not None
                else order_for_view(view, self.viewer.dims.ndim)
            )
            self.viewer.dims.ndisplay = 2
            self.active_section_id = active_id
            self.renderer.active_section_id = active_id
            self.apply_working_state()
            self.fit_view()
        except Exception:
            self.viewer.dims.update(old_dims)
            self.active_section_id = old_active
            self.renderer.active_section_id = old_active
            raise
    @trace_interaction("snap_3d")
    def snap_3d(self, capture=True, steps=15, ease=Easing.LINEAR):
        """Return the canvas to the 3D view."""
        with self.renderer.applying_state():
            self._snap_3d_guarded()
        if capture:
            self.capture_keyframe(steps=steps, ease=ease)

    def _snap_3d_guarded(self):
        self.renderer.preflight(ndisplay=3)
        self.viewer.dims.ndisplay = 3
        # Margins are a 2D thick-slice view override.  Leaving them authored
        # while entering 3D makes the next captured frame carry a stale slab
        # even though the 3D section is rendered by its plane layer.
        OrthoSlicer._reset_margins(self.viewer)
        self.apply_working_state()
        self.fit_view()
    def _axis_limits(self, axis):
        """World-coordinate ``(start, stop)`` of one dims axis."""
        axis_range = self.viewer.dims.range[axis]
        start = float(getattr(axis_range, "start", axis_range[0]))
        stop = float(getattr(axis_range, "stop", axis_range[1]))
        return start, stop

    @trace_interaction("add_scroll_through")
    def add_scroll_through(
        self,
        axis=None,
        start=None,
        stop=None,
        steps=30,
        enter_steps=1,
        ease=Easing.LINEAR,
    ):
        """Add two keyframes that sweep the dims slider along one axis.

        This is the plain slice-by-slice scroll-through: in 2D it walks the
        stack. In 3D it only affects axes outside the three displayed spatial
        axes (such as time); use :meth:`add_slice_sweep` to move a visible 3D
        section through a volume. Only ``dims.point`` differs between the two
        keyframes, together with the matching position of an active 2D section.
        Camera and layer settings stay as they are now.

        Parameters
        ----------
        axis : int, optional
            Axis to scroll. Defaults to the current slider axis, falling back
            to the third-from-last (Z) axis when everything is displayed.
        start, stop : float, optional
            World coordinates to sweep between. Default to the full extent of
            the data along ``axis``.
        steps : int
            Number of interpolation steps across the sweep -- its duration.
        enter_steps : int
            Steps used to reach the *start* of the sweep from the previous
            keyframe. The default of 1 makes the sweep begin immediately;
            raise it to glide into position first.

        Returns
        -------
        tuple
            The two keyframes that were added.
        """
        dims = self.viewer.dims
        if axis is None:
            not_displayed = tuple(dims.not_displayed)
            axis = (
                int(not_displayed[0])
                if not_displayed
                else max(0, dims.ndim - 3)
            )
        axis = int(axis)

        limits = self._axis_limits(axis)
        start = limits[0] if start is None else float(start)
        stop = limits[1] if stop is None else float(stop)

        from copy import deepcopy
        from dataclasses import replace

        from .viewer_state import ViewerState

        ortho, scene, view = self._capture_state()
        base = ViewerState.from_viewer(
            self.viewer, ortho=ortho, scene=scene, view=view
        )
        snapshots = []
        for value in (start, stop):
            state_dims = deepcopy(base.dims)
            points = list(state_dims["point"])
            points[axis] = value
            state_dims["point"] = tuple(points)
            state_scene = deepcopy(base.scene)
            section = self._active_section()
            if section is not None and section.enabled and dims.ndisplay < 3:
                world_axis = int(np.argmax(np.abs(section.normal)))
                if axis == dims.ndim - 3 + world_axis:
                    position = list(state_scene[section.id]["position"])
                    position[world_axis] = value
                    state_scene[section.id]["position"] = tuple(position)
            snapshots.append(replace(base, dims=state_dims, scene=state_scene))
        return self._insert_endpoint_snapshots(
            snapshots, enter_steps=enter_steps, steps=steps, ease=ease
        )

    @trace_interaction("add_slice_sweep")
    def add_slice_sweep(
        self,
        scene_object,
        start=None,
        stop=None,
        steps=30,
        enter_steps=1,
        ease=Easing.LINEAR,
    ):
        """Add two keyframes that sweep an ortho slice along its own normal.

        The 3D counterpart of :meth:`add_scroll_through`: rather than moving a
        dims slider, this slides the slab through the volume in whatever
        direction it faces, including oblique. Combine it with different
        camera angles on each keyframe to get the Imaris effect of a section
        travelling through the volume while the view orbits.

        Parameters
        ----------
        scene_object : OrthoSlice
            The slice to sweep.
        start, stop : float, optional
            Distances along the normal, in world units. Default to the extent
            of the source data projected onto the normal, so the slab travels
            from one face of the volume to the other.

        Returns
        -------
        tuple
            The two keyframes that were added.
        """
        scene_object.sync_from_viewer(self.viewer)
        low, high, centre, normal = self._sweep_geometry(scene_object)
        start = low if start is None else float(start)
        stop = high if stop is None else float(stop)

        def position_at(distance):
            offset = distance - float(np.dot(centre, normal))
            return tuple(centre + offset * normal)

        from copy import deepcopy
        from dataclasses import replace

        from .viewer_state import ViewerState

        ortho, scene, view = self._capture_state()
        base = ViewerState.from_viewer(
            self.viewer, ortho=ortho, scene=scene, view=view
        )
        snapshots = []
        for distance in (start, stop):
            state_scene = deepcopy(base.scene)
            state_scene[scene_object.id]["position"] = position_at(distance)
            snapshots.append(replace(base, scene=state_scene))
        first, last = self._insert_endpoint_snapshots(
            snapshots, enter_steps=enter_steps, steps=steps, ease=ease
        )
        # Selection restores this final position into the editable object.
        return first, last

    def _insert_endpoint_snapshots(self, snapshots, *, enter_steps, steps, ease):
        """Insert a pair without rendering or screenshotting the first endpoint."""
        from .key_frame import KeyFrame

        active = self.key_frames.selection.active
        position = self.key_frames.index(active) + 1 if active else len(self.key_frames)
        frames = []
        for state, duration in zip(snapshots, (enter_steps, steps)):
            frame = KeyFrame(
                viewer_state=state,
                thumbnail=np.zeros((30, 30, 4), dtype=np.uint8),
                steps=duration,
                ease=ease,
                name=self._next_keyframe_name(),
            )
            frame._thumbnail_pending = True
            frames.append(frame)
        with self.key_frames.selection.events.active.blocker():
            for offset, frame in enumerate(frames):
                self.key_frames.insert(position + offset, frame)
            self.key_frames.selection.active = frames[-1]
        self.set_movie_frame_index(self._keyframe_frame_index(position + 1))
        return tuple(frames)

    def _sweep_geometry(self, scene_object):
        """Extent of the source data projected onto a slice's normal."""
        normal = np.asarray(scene_object.normal, dtype=float)
        magnitude = np.linalg.norm(normal)
        if magnitude == 0:
            raise ValueError("Cannot sweep a slice with a zero normal")
        normal = normal / magnitude

        layers = scene_object.source_layers(self.viewer)
        if not layers:
            raise ValueError("Ortho slice has no source layer to sweep across")

        extents = [np.asarray(layer.extent.world, dtype=float)[:, -3:]
                   for layer in layers]
        low_corner = np.min([extent[0] for extent in extents], axis=0)
        high_corner = np.max([extent[1] for extent in extents], axis=0)
        centre = (low_corner + high_corner) / 2.0

        # project every corner of the bounding box onto the normal, so an
        # oblique slice still travels the full depth of the data
        corners = np.array(
            [
                [
                    high_corner[i] if (index >> i) & 1 else low_corner[i]
                    for i in range(3)
                ]
                for index in range(8)
            ]
        )
        distances = corners @ normal
        return (
            float(distances.min()),
            float(distances.max()),
            centre,
            normal,
        )

    def set_voxel_size(self, layer_name, scale, units=None):
        """Correct the physical voxel spacing of a layer.

        Use this when a file's metadata is wrong -- most commonly a TIFF whose
        Z spacing does not match the microscope's step size, which leaves the
        volume stretched or squashed along Z.

        The spacing is applied to the live layer *and* rewritten into every
        keyframe already captured. Without that, replaying an older keyframe
        would restore the wrong spacing, because ``scale`` is part of the
        captured layer state. Voxel size describes the data rather than the
        animation, so it is corrected everywhere rather than being keyframed.

        Parameters
        ----------
        layer_name : str
            Name of the layer to correct.
        scale : sequence of float
            Physical size of one voxel. Applied to the layer's trailing axes,
            so a 3-element ``(z, y, x)`` works for a layer that also has
            leading time or channel axes.
        units : str or sequence of str, optional
            Physical unit for the spacing, e.g. ``"um"``. Applied to the same
            trailing axes.
        """
        if layer_name not in self.viewer.layers:
            raise KeyError(f"No layer named {layer_name!r}")

        layer = self.viewer.layers[layer_name]
        scale = [float(value) for value in scale]
        if any(value == 0 for value in scale):
            raise ValueError("Voxel size must be non-zero on every axis")

        new_scale = list(layer.scale)
        new_scale[-len(scale) :] = scale[-len(new_scale) :]
        layer.scale = new_scale

        new_units = None
        if units is not None:
            if isinstance(units, str):
                units = [units] * len(scale)
            try:
                new_units = list(layer.units)
                new_units[-len(units) :] = list(units)[-len(new_units) :]
                layer.units = new_units
            except (AttributeError, ValueError, TypeError):
                # older napari, or a unit string it will not accept
                new_units = None
            if new_units is not None:
                # Section layers are derived views of this source. Keep the
                # live 3D plane and 2D projection in the same unit system so
                # correcting voxel size does not introduce a units warning.
                for section in self.scene:
                    if section.kind != "ortho_slice":
                        continue
                    for target in section._channel_sections(self.viewer):
                        if target.source_layer(self.viewer) is not layer:
                            continue
                        for role in ("section_3d", "section_2d"):
                            managed = self.layer_registry.find_managed(
                                target.id, role
                            )
                            if managed is not None:
                                managed.units = tuple(layer.units[-managed.ndim:])

        self._rewrite_keyframe_layer_state(
            layer_name, scale=new_scale, units=new_units
        )
        return tuple(new_scale)

    def invalidate_source(self, layer_name_or_id):
        """Refresh sections after an in-place edit of a source array.

        A hidden napari source layer does not emit ``set_data`` when its
        ``refresh()`` method is called.  This explicit hook invalidates the
        projection cache and requests the current section again in that case.
        Replacing ``layer.data`` already emits the normal data event.
        """
        layer = self.layer_registry.find(layer_name_or_id)
        if layer is None:
            layer = self.layer_registry.find(name=layer_name_or_id)
        if layer is None:
            raise KeyError(f"No layer {layer_name_or_id!r}")
        layer.events.data(value=layer.data)
        self.apply_working_state()

    def _rewrite_keyframe_layer_state(self, layer_name, **values):
        """Overwrite one layer's captured properties in every keyframe."""
        changed = False
        for key_frame in self.key_frames:
            state = key_frame.viewer_state
            if layer_name not in state.layers:
                continue
            layers = {k: dict(v) for k, v in state.layers.items()}
            state_key = layer_name
            if state_key not in layers:
                state_key = next(
                    key
                    for key, layer_state in layers.items()
                    if layer_state.get("name") == layer_name
                )
            for key, value in values.items():
                if value is not None and key in layers[state_key]:
                    layers[state_key][key] = list(value)
            key_frame.viewer_state = replace(state, layers=layers)
            changed = True
        if changed:
            self._frames._rebuild_frame_index()

    def set_layer_visible(
        self, keyframe_index: int, layer_name: str, visible: bool
    ):
        """Show or hide a napari layer at a single keyframe."""
        key_frame = self.key_frames[keyframe_index]
        state = key_frame.viewer_state
        if layer_name not in state.layers:
            raise KeyError(
                f"No layer {layer_name!r} in keyframe {keyframe_index}"
            )
        layers = {k: dict(v) for k, v in state.layers.items()}
        state_key = layer_name
        if state_key not in layers:
            state_key = next(
                key
                for key, layer_state in layers.items()
                if layer_state.get("name") == layer_name
            )
        layers[state_key]["visible"] = bool(visible)
        key_frame.viewer_state = replace(state, layers=layers)
        self._frames._rebuild_frame_index()

    def _next_keyframe_name(self):
        """Continue automatic numbering after loaded or renamed keyframes."""
        number = next(self._keyframe_counter)
        prefix = "Key Frame "
        for frame in self.key_frames:
            suffix = frame.name.removeprefix(prefix)
            if (
                frame.name.startswith(prefix)
                and suffix.isascii()
                and suffix.isdecimal()
            ):
                number = max(number, int(suffix) + 1)
        self._keyframe_counter = count(number + 1)
        return f"{prefix}{number}"

    def capture_keyframe(
        self, steps=15, ease=Easing.LINEAR, insert=True, position: int = None
    ):
        """Record current key-frame

        Parameters
        ----------
        steps : int
            Number of interpolation steps between last keyframe and captured one.
        ease : callable, optional
            If provided this method should make from `[0, 1]` to `[0, 1]` and will
            be used as an easing function for the transition between the last state
            and captured one.
        insert : bool
            If captured key-frame should insert into current list or replace the current
            keyframe.
        position : int, optional
            If provided, place new frame at this index. By default, inserts at current
            active frame.
        """

        if position is None:
            active_keyframe = self.key_frames.selection.active
            if active_keyframe:
                position = self.key_frames.index(active_keyframe)
            else:
                if insert:
                    position = -1
                else:
                    raise ValueError("No selected keyframe to replace !")

        ortho, scene, view = self._capture_state()
        new_frame = KeyFrame.from_viewer(
            self.viewer,
            steps=steps,
            ease=ease,
            ortho=ortho,
            scene=scene,
            view=view,
        )
        new_frame.name = self._next_keyframe_name()

        if insert:
            self.key_frames.insert(position + 1, new_frame)
        else:
            self.key_frames[position] = new_frame

    def overwrite_keyframe(self, index: int):
        """Replace the key-frame at ``index`` with the current viewer state.

        The target key-frame's interpolation settings (``steps`` and ``ease``)
        and name are preserved; only the captured viewer state and thumbnail
        are updated to match the current view.

        Parameters
        ----------
        index : int
            Index of the key-frame to overwrite.
        """
        existing = self.key_frames[index]
        ortho, scene, view = self._capture_state()
        captured = KeyFrame.from_viewer(
            self.viewer,
            steps=existing.steps,
            ease=existing.ease,
            ortho=ortho,
            scene=scene,
            view=view,
        )
        # Update the existing key-frame in place (keeping its identity, name,
        # steps and ease) rather than replacing the list item -- replacing
        # would emit a `changed` event before the frame-sequence cache is
        # rebuilt, leaving navigation pointing at a stale key-frame.
        existing.viewer_state = captured.viewer_state
        existing.thumbnail = captured.thumbnail

        # refresh the interpolation cache, then notify listeners (e.g. the list
        # widget thumbnail) that this key-frame changed.
        self._frames._rebuild_frame_index()
        self.key_frames.events.changed(
            index=index, old_value=existing, value=existing
        )

    def set_to_keyframe(self, frame: int):
        """Set the viewer to a given key-frame

        Parameters
        ----------
        frame : int
            Key-frame index to visualize
        """
        self.key_frames.selection.active = self.key_frames[frame]

    def _validate_animation(self):
        if len(self.key_frames) < 2:
            raise ValueError(
                f"Must have at least 2 key frames, received {len(self.key_frames)}"
            )

    def set_key_frame_index(self, index: int):
        frame_index = self._keyframe_frame_index(index)
        self.set_movie_frame_index(frame_index)

    @trace_interaction("set_movie_frame_index")
    def set_movie_frame_index(self, index: int):
        """Set state to a specific frame in the final movie."""
        if self._navigating:
            # Updating the selected row during navigation emits the same
            # selection callback. The outer call owns the render transaction.
            return
        self._navigating = True
        try:
            if index < 0:
                index += len(self._frames)

            key_frame = self._frames._frame_index[index][0]

            # to prevent active callback again
            if self.key_frames.selection.active != key_frame:
                self.key_frames.selection.active = key_frame

            self._frames.set_movie_frame_index(self.viewer, index)
            # Keep the editable scene and legacy slicer synchronized with the
            # snapshot that was just rendered.  Without this, selecting a
            # loaded frame changed pixels but the next capture still read the
            # previously selected scene controls.
            state = self._frames[index]
            self.scene.restore_snapshot(self.viewer, state.scene)
            self.ortho_slicer = OrthoSlicer.from_dict(state.ortho)
            active = (
                state.view.get("active_section_id")
                if isinstance(state.view, dict)
                else None
            )
            self.active_section_id = active
            self.renderer.active_section_id = active
            # ViewerState.apply has already rendered the snapshot. Give the
            # restored *live* scene objects ownership of their managed layer
            # listeners and last-written plane state without rendering again.
            for obj in self.scene:
                if obj.kind != "ortho_slice":
                    continue
                for target in obj._channel_sections(self.viewer):
                    layer = self.layer_registry.find_managed(
                        target.id, "section_3d"
                    )
                    if layer is None:
                        continue
                    target._bind_layer_events(layer, self.viewer)
                    plane = layer.plane
                    target._applied_plane = {
                        "position": tuple(plane.position),
                        "normal": tuple(plane.normal),
                        "thickness": float(plane.thickness),
                    }
                    target._applied_visible = bool(layer.visible)
                    target._applied_rendering = layer.rendering
                    if obj.channels is not None:
                        source = target.source_layer(self.viewer)
                        if source is not None:
                            obj._bind_composite_appearance_events(
                                self.viewer, source
                            )
            self._current_frame = index
            self._last_navigation_selection = self.key_frames.selection.active
            self.events.state_applied()
            if getattr(key_frame, "_thumbnail_pending", False):
                self._queue_thumbnail_refresh(key_frame)

        except KeyError:
            return
        finally:
            self._navigating = False

    def _queue_thumbnail_refresh(self, key_frame):
        """Capture a generated endpoint only after its exact section is ready."""
        try:
            from qtpy.QtCore import QTimer
            from qtpy.QtWidgets import QApplication
        except ImportError:
            return
        if QApplication.instance() is None or not hasattr(self.viewer, "window"):
            return
        from .scene.processing import service_for_viewer
        from .utils import make_thumbnail

        def capture():
            if self.key_frames.selection.active is not key_frame:
                return
            if not getattr(key_frame, "_thumbnail_pending", False):
                return
            try:
                key_frame.thumbnail = make_thumbnail(
                    self.viewer.screenshot(canvas_only=True, flash=False)
                )
            except Exception as error:  # thumbnail must not break navigation
                logger.warning("Could not capture keyframe thumbnail: %s", error)
                return
            key_frame._thumbnail_pending = False
            row = self.key_frames.index(key_frame)
            self.key_frames.events.changed(
                index=row, old_value=key_frame, value=key_frame
            )

        service_for_viewer(self.viewer).when_idle(
            lambda: QTimer.singleShot(0, capture)
        )

    def animate(
        self,
        path,
        fps=20,
        quality=5,
        format=None,
        canvas_only=True,
        scale_factor=None,
        perf_log=True,
        prefetch=4,
        prefetch_workers=4,
        dask_cache="auto",
        diagnose=True,
        compat=True,
        render_size=None,
    ):
        """Create a movie based on key-frames
        Parameters
        -------
        path : str
            path to use for saving the movie (can also be a path). Extension
            should be one of .gif, .mp4, .mov, .avi, .mpg, .mpeg, .mkv, .wmv
            If no extension is provided, images are saved as a folder of PNGs
        fps : int
            frames per second
        quality: float
            number from 1 (lowest quality) to 9
            only applies to non-gif extensions
        format: str
            The format to use to write the file. By default imageio selects the appropriate
            for you based on the filename.
        canvas_only : bool
            If True include just includes the canvas, otherwise include the full napari
            viewer.
        scale_factor : float
            Resize the captured frame after rendering. Use ``render_size``
            to rerender source detail at a chosen output resolution.
        perf_log : bool
            If True (default), time each phase of the render pipeline
            including nested state/capture work, writer backpressure, cleanup,
            and diagnostics. Save a text summary and bounded frame/thread
            events in a ``.render_perf.json`` sidecar. Timings are inclusive;
            nested phases and concurrent encoding overlap.
        prefetch : int
            Number of upcoming frames whose data slices are read ahead on
            background threads to warm the cache (overlapping disk reads). Set
            to 0 to disable. Most useful when rendering lazily-loaded data
            (e.g. dask/HDF5/Imaris) where the ``apply`` phase dominates.
            Increase together with ``prefetch_workers`` if your storage serves
            reads in parallel (SSD/NVMe/RAID).
        prefetch_workers : int
            Number of background threads used for prefetching.
        dask_cache : str, int or None
            Opportunistic dask cache used during rendering so prefetched
            (decompressed) chunks are reused by the main-thread read instead of
            being recomputed, and frames sharing a chunk are free. ``"auto"``
            (default) sizes it from available RAM; pass a byte count for a
            fixed size, or ``None``/0 to disable.
        diagnose : bool
            If True (default), sample every rendered frame and read the
            finished file back off disk, then report anything that would make
            the output unplayable or blank: frames that changed size
            mid-render, frames that are a single flat colour, frames that never
            change, a truncated container, or a file too small to hold the
            frames written.
        compat : bool
            If True (default), encode ``.mp4``/``.mov``/``.mkv`` with the most
            widely playable settings -- H.264 in ``yuv420p`` with the moov atom
            moved to the front of the file -- rather than leaving the codec and
            pixel format to whatever the installed ffmpeg happens to default
            to. This is what makes the output open in OS-bundled players such
            as Windows Media Player and Photos. Set to False to let imageio
            choose.
        render_size : tuple[int, int] or None
            Render each canvas frame at (height, width), independent of the
            viewer window size. Requires ``canvas_only=True``. This rerenders
            the scene instead of resizing an existing screenshot.

        Notes
        -----
        Frames are rendered on the calling (main) thread -- napari needs its
        single OpenGL context there -- while encoding/writing to disk and
        prefetching the next frames' data run on background threads, so they
        overlap. This keeps the disk and writer busy while a frame renders and
        avoids a silent pause at the end.
        """
        if render_size is not None:
            if not canvas_only:
                raise ValueError("Render size requires canvas_only=True.")
            if len(render_size) != 2 or any(value <= 0 for value in render_size):
                raise ValueError("Render size must be (height, width) with positive values.")
            if scale_factor not in (None, 1):
                raise ValueError("Render size and scale_factor cannot be combined.")
        self._validate_animation()
        working_ortho, working_scene, working_view = self._capture_state()
        working_state = ViewerState.from_viewer(
            self.viewer,
            ortho=working_ortho,
            scene=working_scene,
            view=working_view,
        )

        perf = PerfLogger(enabled=perf_log)
        perf.start()
        perf.metadata.update(path=str(path), export_path="script", fps=fps,
                             render_size=render_size, prefetch=prefetch,
                             prefetch_workers=prefetch_workers,
                             dask_cache=dask_cache, diagnose=diagnose)
        self._last_render_perf = perf

        # Evidence about what the render actually produced, for the case where
        # everything reports success but the file is unplayable or blank.
        diagnostics = RenderDiagnostics(enabled=diagnose)
        with perf.timer("diagnostics.environment"):
            diagnostics.describe_environment()

        # create path object
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        folder_path = path_obj.absolute().parent.joinpath(path_obj.stem)

        # if path has no extension, save as folder of PNG
        save_as_folder = suffix == ""
        if save_as_folder:
            # This is a common surprise: a path without a video extension
            # produces a folder of PNGs rather than a movie. Say so clearly.
            msg = (
                f"No video file extension in {path!r}: saving a folder of PNG "
                "frames instead of a video. For a movie, give a path ending in "
                ".mp4, .mov, .gif, .avi, .mkv, etc."
            )
            logger.warning(msg)
            print(f"NOTE: {msg}")

        # try to create an ffmpeg writer. If not available, fall back to a
        # folder of PNGs -- but make that fallback *loud*, since silently
        # producing PNGs instead of the requested video is confusing.
        writer = None
        if not save_as_folder:
            try:
                # gif doesn't accept the quality parameter
                if suffix in VIDEO_SUFFIXES:
                    if compat and suffix in H264_SUFFIXES:
                        # CRF replaces quality here, so don't pass both
                        writer_kwargs = {"fps": fps, **COMPAT_ENCODER}
                        writer_kwargs["output_params"] = _compat_output_params(
                            quality
                        )
                        settings = ", ".join(
                            f"{k}={v}" for k, v in writer_kwargs.items()
                        )
                        print(f"Encoder (compatibility mode): {settings}")
                        logger.info("Encoder settings: %s", settings)
                    else:
                        writer_kwargs = {"fps": fps, "quality": quality}
                    if render_size is not None:
                        writer_kwargs["macro_block_size"] = 2
                    writer = imageio.get_writer(
                        path, format=format, **writer_kwargs
                    )
                else:
                    writer = imageio.get_writer(path, fps=fps, format=format)
            except Exception as err:  # noqa: BLE001 - report and fall back
                save_as_folder = True
                msg = (
                    f"Could not create a video writer for {path!r} ({err}). "
                    "Saving a folder of PNG files instead."
                )
                logger.warning(msg)
                print(f"WARNING: {msg}")

        if save_as_folder:
            # if movie is saved as series of PNG, create a folder
            if folder_path.is_dir():
                for f in folder_path.glob("*.png"):
                    os.remove(f)
            else:
                folder_path.mkdir(exist_ok=True)

        n_frames = len(self._frames)
        perf.metadata["expected_frames"] = n_frames

        # -- define how a single (index, frame) pair is written to disk --
        if writer is not None:
            pad_for_video = suffix in VIDEO_SUFFIXES

            def write_frame(item):
                _, frame = item
                if pad_for_video:
                    frame = pad_video_frame(
                        frame, block_size=2 if render_size is not None else 16
                    )
                writer.append_data(frame)

        else:

            def write_frame(item):
                ind, frame = item
                fname = folder_path / f"{path_obj.stem}_{ind}.png"
                imsave(fname, frame)

        # -- background writer thread (encoding overlaps rendering) --
        frame_queue: "queue.Queue" = queue.Queue(maxsize=8)
        write_errors = []

        def _consumer():
            while True:
                item = frame_queue.get()
                try:
                    if item is None:
                        return
                    with perf.timer("encode/write", frame=item[0]):
                        write_frame(item)
                    perf.count("frames_written")
                except Exception as err:  # noqa: BLE001
                    write_errors.append(err)
                finally:
                    frame_queue.task_done()

        writer_thread = threading.Thread(
            target=_consumer, name="napari-animation-writer", daemon=True
        )
        writer_thread.start()

        target = path if writer is not None else folder_path
        print(f"Rendering {n_frames} frames -> {target}")
        logger.info("Rendering %d frames -> %s", n_frames, target)

        # opportunistic dask cache so prefetched chunks are reused by the
        # main-thread read (kept active for the whole render below).
        cache_cm = dask_cache_context(dask_cache)
        cache_bytes = cache_cm.__enter__()
        if cache_bytes:
            print(f"dask cache enabled: {cache_bytes / 1e9:.1f} GB")

        # warm the cache for upcoming frames' data slices on background threads
        prefetcher = SlicePrefetcher(
            self.viewer,
            self._frames,
            depth=prefetch,
            workers=prefetch_workers,
        )
        perf.metadata.update(
            prefetch_enabled=prefetcher.enabled,
            prefetch_skip_reason=prefetcher.skip_reason,
        )
        if prefetcher.enabled:
            print(
                f"Prefetching up to {prefetcher.depth} frame(s) ahead on "
                f"{prefetch_workers} thread(s)"
            )

        render_error = None
        try:
            with ExportCaptureSession(self.viewer, render_size), tqdm(total=n_frames) as pbar:
                for ind in range(n_frames):
                    if write_errors:
                        break
                    # kick off reads for the look-ahead window
                    with perf.timer("prefetch.schedule", frame=ind):
                        prefetcher.advance(ind)
                    with perf.timer("interpolate", frame=ind):
                        state = self._frames[ind]
                    from .scene.processing import exact_render

                    with exact_render(self.viewer):
                        with perf.timer("apply", frame=ind):
                            state.apply(self.viewer)
                        with perf.timer("screenshot", frame=ind):
                            frame = capture_export_frame(
                                self.viewer, canvas_only=canvas_only,
                                render_size=render_size,
                            )
                    if scale_factor not in (None, 1):
                        from scipy import ndimage as ndi

                        with perf.timer("scale", frame=ind):
                            frame = ndi.zoom(
                                frame, (scale_factor, scale_factor, 1)
                            ).astype(np.uint8)
                    # sample the frame exactly as it will be encoded
                    with perf.timer("diagnostics.frame", frame=ind):
                        diagnostics.record_frame(ind, frame)
                    with perf.timer("queue.put_wait", frame=ind):
                        frame_queue.put((ind, frame))
                    perf.count("frames_captured")
                    pbar.update(1)
        except BaseException as error:
            render_error = error
            raise
        finally:
            # signal the writer to finish and wait for the queue to drain
            with perf.timer("pipeline.drain"):
                prefetcher.shutdown()
                frame_queue.put(None)
                print("Waiting for encoder to finish writing queued frames...")
                writer_thread.join()
            cache_cm.__exit__(None, None, None)
            if writer is not None:
                print("Finalizing video container (muxing)...")
                logger.info("Finalizing video container for %s", path)
                try:
                    with perf.timer("finalize"):
                        writer.close()
                except Exception as err:  # noqa: BLE001
                    # Preserve an earlier render/write error, but still
                    # restore the working editor if finalization fails.
                    write_errors.append(err)
            # Rendering evaluates detached frame snapshots. Always return the
            # editor to the exact working state the user had before export.
            with perf.timer("restore"):
                working_state.apply(self.viewer)
                self.scene.restore_snapshot(self.viewer, working_scene)
                self.ortho_slicer = OrthoSlicer.from_dict(working_ortho)
                active = (working_view or {}).get("active_section_id")
                self.active_section_id = active
                self.renderer.active_section_id = active
                self.renderer.apply_working(ortho=working_ortho)
                self.viewer.camera.update(working_state.camera)

            if render_error is not None:
                perf.metadata.update(status="failed", error=str(render_error))
                perf.stop()
                perf.save(Path(target))

        if write_errors:
            perf.metadata["status"] = "failed"
            perf.stop()
            perf.save(Path(target))
            raise write_errors[0]

        if writer is not None:
            print(f"Saved animation to {path}")
            logger.info("Saved animation to %s", path)
            output = path_obj
            # only a real video file can be decoded back; a PNG folder can't
            with perf.timer("diagnostics.verify"):
                diagnostics.verify_output(path, n_frames)
        else:
            print(f"Saved {n_frames} PNG frames to {folder_path}")
            logger.info("Saved %d PNG frames to %s", n_frames, folder_path)
            output = folder_path

        diagnostics_report = diagnostics.report()
        if diagnostics_report:
            print(diagnostics_report)
            logger.info("%s", diagnostics_report)

        perf.metadata.update(status="complete", output_shape=diagnostics.first_shape,
                             blank_frames=list(diagnostics.blank_frames),
                             sampled_pixel_range=list(diagnostics.value_range))
        perf.stop()
        perf.save(output)
        if perf_log:
            report = perf.report()
            print(report)
            perf.log_report()
            if diagnostics_report:
                report = f"{diagnostics_report}\n\n{report}"
            # also write the report to a file the user can find and report back
            log_path = self._write_render_log(output, n_frames, report)
            if log_path is not None:
                print(f"Performance log written to: {log_path}")

    @staticmethod
    def _write_render_log(output: Path, n_frames: int, report: str):
        """Write the render performance report next to the output.

        For a video file ``movie.mp4`` the log is ``movie.render_log.txt``
        alongside it; for a folder of PNGs it is ``render_log.txt`` inside the
        folder. Returns the log path, or ``None`` if it could not be written.
        """
        from datetime import datetime

        output = Path(output)
        if output.suffix:  # a file (video)
            log_path = output.with_name(output.stem + ".render_log.txt")
        else:  # a folder of PNGs
            log_path = output / "render_log.txt"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().isoformat(timespec="seconds")
            with open(log_path, "w") as f:
                f.write(
                    f"napari-animation render log\n"
                    f"timestamp: {timestamp}\n"
                    f"output: {output}\n"
                    f"frames: {n_frames}\n\n"
                    f"{report}\n"
                )
            return log_path
        except OSError as err:  # don't fail the render over a log file
            logger.warning("Could not write render log: %s", err)
            return None

    def save_keyframes(self, path):
        """Save the keyframes to a file so the animation can be resumed later.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file. A ``.json`` extension is added if missing.

        Returns
        -------
        pathlib.Path
            The path the keyframes were written to.
        """
        from .io import save_animation

        return save_animation(self, path)

    def load_keyframes(self, path, reload_layers: bool = True):
        """Load keyframes previously written by :meth:`save_keyframes`.

        Existing keyframes are replaced. When ``reload_layers`` is ``True``
        (default), layer data files recorded in the saved file are re-opened
        for any layers not already present in the viewer.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a previously saved keyframes file.
        reload_layers : bool
            Whether to re-open recorded layer data files.
        """
        from .io import load_animation

        load_animation(self, path, reload_layers=reload_layers)

    def close(self):
        """Release interactive caches and worker resources owned by this dock."""
        if self._closed:
            return
        self._closed = True
        self.resolution.close()
        from .native_clipping import release_native_clipping
        release_native_clipping(self.viewer)
        if self._viewer_destroyed is not None:
            try:
                self._viewer_destroyed.disconnect(self.close)
            except (RuntimeError, TypeError):
                pass
            self._viewer_destroyed = None
        service = getattr(self.viewer, "_napari_animation_projection_service", None)
        if service is not None:
            service.shutdown()
            if getattr(self.viewer, "_napari_animation_projection_service", None) is service:
                delattr(self.viewer, "_napari_animation_projection_service")
        if self._dims_emitter is not None:
            try:
                self._dims_emitter.disconnect(self._on_dims_point_changed)
            except (TypeError, ValueError):
                pass
            self._dims_emitter = None
        if self._interactive_cache_acquired:
            release_interactive_cache()
            self._interactive_cache_acquired = False
        if getattr(self.viewer, "_napari_animation_renderer", None) is self.renderer:
            for layer in self.viewer.layers:
                listener = getattr(
                    layer, "_napari_animation_source_listener", None
                )
                if listener is not None:
                    callback, emitters = listener
                    for emitter in emitters:
                        try:
                            emitter.disconnect(callback)
                        except (TypeError, ValueError):
                            pass
                    delattr(layer, "_napari_animation_source_listener")
        self._trace.save()

    def _keyframe_frame_index(self, keyframe_index):
        """Gets the frame index of the keyframe corresponding to keyframe_index."""
        # Get all steps leading to keyframe.
        steps_to_keyframe = [
            kf.steps for kf in self.key_frames[1 : keyframe_index + 1]
        ]
        frame_index = np.sum(steps_to_keyframe) if steps_to_keyframe else 0
        return int(frame_index)

    def _on_keyframe_removed(self, event):
        self.key_frames.selection.active = None

    def _on_keyframe_changed(self, event):
        self.key_frames.selection.active = event.value

    def _on_active_keyframe_changed(self, event):
        active_keyframe = event.value
        if active_keyframe:
            keyframe_index = self.key_frames.index(active_keyframe)
            frame_index = self._keyframe_frame_index(keyframe_index)
            if (
                frame_index == self._current_frame
                and active_keyframe is self._last_navigation_selection
            ):
                # napari can emit a second active-selection notification for
                # one click after the first callback has already rendered it.
                return
            self.set_movie_frame_index(frame_index)
