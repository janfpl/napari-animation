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
from tqdm import tqdm

from .diagnostics import RenderDiagnostics
from .easing import Easing
from .frame_sequence import FrameSequence
from .key_frame import KeyFrame, KeyFrameList
from .ortho_slicer import OrthoSlicer
from .perf import PerfLogger
from .prefetch import SlicePrefetcher, dask_cache_context
from .scene import Scene

logger = logging.getLogger("napari_animation")

#: File extensions handled by the imageio-ffmpeg video writer.
VIDEO_SUFFIXES = (".mov", ".avi", ".mpg", ".mpeg", ".mp4", ".mkv", ".wmv")


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
        self.viewer = viewer

        self.key_frames = KeyFrameList()

        self.key_frames.events.removed.connect(self._on_keyframe_removed)

        self.key_frames.events.changed.connect(self._on_keyframe_changed)

        self.key_frames.selection.events.active.connect(
            self._on_active_keyframe_changed
        )

        self._keyframe_counter = count()  # track number of frames created

        self._frames = FrameSequence(self.key_frames)

        # Imaris-style optical-section slicer; its parameters are captured into
        # each keyframe so the optical section can be animated.
        self.ortho_slicer = OrthoSlicer()

        # Clipping planes and other 3D scene objects. Their parameters are
        # captured into every keyframe, so each can be reshaped and switched
        # on or off independently at each keyframe.
        self.scene = Scene()

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
        return ortho, scene

    def remove_scene_object(self, scene_object):
        """Remove a scene object, taking any layer it owns with it.

        The object's parameters are left in already-captured keyframes; they
        are simply ignored, so removing an object cannot corrupt an animation
        that was built around it.
        """
        scene_object.remove_layer(self.viewer)
        self.scene.remove(scene_object)
        ortho, _ = self._capture_state()
        self.scene.apply(self.viewer, ortho=ortho)

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
        self.scene.append(scene_object)
        if backfill:
            params = scene_object.to_dict()
            for key_frame in self.key_frames:
                state = key_frame.viewer_state
                scene = dict(state.scene or {})
                scene[scene_object.id] = dict(params)
                key_frame.viewer_state = replace(state, scene=scene)
            self._frames._rebuild_frame_index()
        ortho, _ = self._capture_state()
        self.scene.apply(self.viewer, ortho=ortho)
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
        layers[layer_name]["visible"] = bool(visible)
        key_frame.viewer_state = replace(state, layers=layers)
        self._frames._rebuild_frame_index()

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

        ortho, scene = self._capture_state()
        new_frame = KeyFrame.from_viewer(
            self.viewer, steps=steps, ease=ease, ortho=ortho, scene=scene
        )
        new_frame.name = f"Key Frame {next(self._keyframe_counter)}"

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
        ortho, scene = self._capture_state()
        captured = KeyFrame.from_viewer(
            self.viewer,
            steps=existing.steps,
            ease=existing.ease,
            ortho=ortho,
            scene=scene,
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

    def set_movie_frame_index(self, index: int):
        """Set state to a specific frame in the final movie."""
        try:
            if index < 0:
                index += len(self._frames)

            key_frame = self._frames._frame_index[index][0]

            # to prevent active callback again
            if self.key_frames.selection.active != key_frame:
                self.key_frames.selection.active = key_frame

            self._frames.set_movie_frame_index(self.viewer, index)
            self._current_frame = index

        except KeyError:
            return

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
            Rescaling factor for the image size. Only used without
            viewer (with_viewer = False).
        perf_log : bool
            If True (default), time each phase of the render pipeline
            (interpolate / apply / screenshot / encode) and log a summary so
            the rate-limiting step is visible.
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

        Notes
        -----
        Frames are rendered on the calling (main) thread -- napari needs its
        single OpenGL context there -- while encoding/writing to disk and
        prefetching the next frames' data run on background threads, so they
        overlap. This keeps the disk and writer busy while a frame renders and
        avoids a silent pause at the end.
        """
        self._validate_animation()

        perf = PerfLogger(enabled=perf_log)
        perf.start()

        # Evidence about what the render actually produced, for the case where
        # everything reports success but the file is unplayable or blank.
        diagnostics = RenderDiagnostics(enabled=diagnose)
        diagnostics.describe_environment()

        # create path object
        path_obj = Path(path)
        folder_path = path_obj.absolute().parent.joinpath(path_obj.stem)

        # if path has no extension, save as folder of PNG
        save_as_folder = path_obj.suffix == ""
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
                if path_obj.suffix in VIDEO_SUFFIXES:
                    writer = imageio.get_writer(
                        path, fps=fps, quality=quality, format=format
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

        # -- define how a single (index, frame) pair is written to disk --
        if writer is not None:

            def write_frame(item):
                _, frame = item
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
                    with perf.timer("encode/write"):
                        write_frame(item)
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
        if prefetcher.enabled:
            print(
                f"Prefetching up to {prefetcher.depth} frame(s) ahead on "
                f"{prefetch_workers} thread(s)"
            )

        try:
            with tqdm(total=n_frames) as pbar:
                for ind in range(n_frames):
                    if write_errors:
                        break
                    # kick off reads for the look-ahead window
                    prefetcher.advance(ind)
                    with perf.timer("interpolate"):
                        state = self._frames[ind]
                    with perf.timer("apply"):
                        state.apply(self.viewer)
                    with perf.timer("screenshot"):
                        frame = self.viewer.screenshot(canvas_only=canvas_only)
                    if scale_factor not in (None, 1):
                        from scipy import ndimage as ndi

                        with perf.timer("scale"):
                            frame = ndi.zoom(
                                frame, (scale_factor, scale_factor, 1)
                            ).astype(np.uint8)
                    # sample the frame exactly as it will be encoded
                    diagnostics.record_frame(ind, frame)
                    frame_queue.put((ind, frame))
                    pbar.update(1)
        finally:
            # signal the writer to finish and wait for the queue to drain
            prefetcher.shutdown()
            frame_queue.put(None)
            print("Waiting for encoder to finish writing queued frames...")
            writer_thread.join()
            cache_cm.__exit__(None, None, None)

        if write_errors:
            if writer is not None:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001
                    pass
            raise write_errors[0]

        if writer is not None:
            # ffmpeg finalizes/muxes the container on close; this can take a
            # while for large movies, so make it visible rather than a hang.
            print("Finalizing video container (muxing)...")
            logger.info("Finalizing video container for %s", path)
            with perf.timer("finalize"):
                writer.close()
            print(f"Saved animation to {path}")
            logger.info("Saved animation to %s", path)
            output = path_obj
            # only a real video file can be decoded back; a PNG folder can't
            diagnostics.verify_output(path, n_frames)
        else:
            print(f"Saved {n_frames} PNG frames to {folder_path}")
            logger.info("Saved %d PNG frames to %s", n_frames, folder_path)
            output = folder_path

        diagnostics_report = diagnostics.report()
        if diagnostics_report:
            print(diagnostics_report)
            logger.info("%s", diagnostics_report)

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
            self.set_key_frame_index(keyframe_index)
