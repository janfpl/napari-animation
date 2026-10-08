"""Benchmark a saved animation with real Qt/OpenGL capture and sidecar timings.

Run from the repository using its napari environment. Source JSON is read-only.
Each invocation creates a new output directory, avoiding existing movie files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("animation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=7,
                        help="Evenly spaced original frames; 0 renders all frames")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--prefetch-sequence", type=str,
                        help="Comma-separated prefetch depths, one full export each")
    parser.add_argument("--timeout", type=int, default=300, help="Qt benchmark deadline in seconds")
    parser.add_argument("--qt", action="store_true", help="Use the Save-button export path")
    parser.add_argument("--legacy-viewport", action="store_true", help="A/B reference: resize and restore on every frame")
    parser.add_argument("--eager-clip-planes", action="store_true", help="A/B reference: allocate full clip-mode backing volumes")
    parser.add_argument("--eager-prefetch-full-volume", action="store_true",
                        help="A/B reference: read ahead already materialized 3D volumes")
    parser.add_argument("--hidden", action="store_true", help="Diagnostic only: skip native window initialization")
    parser.add_argument("--lossless", action="store_true", help="Save first/middle/last unencoded frames for equality checks")
    parser.add_argument("--cache-stats", action="store_true",
                        help="Count Dask cache hits and distinct inserted task keys")
    parser.add_argument("--profile", action="store_true", help="Profile load and native initialization")
    parser.add_argument("--no-diagnose", action="store_true")
    args = parser.parse_args()
    if args.samples < 0 or args.repeat < 1 or min(args.height, args.width) < 1:
        parser.error("Invalid sample count, repeat count or dimensions")
    try:
        prefetch_depths = (
            [int(value) for value in args.prefetch_sequence.split(",")]
            if args.prefetch_sequence else [args.prefetch] * args.repeat
        )
    except ValueError:
        parser.error("Prefetch depths must be comma-separated integers")
    if not prefetch_depths or min(prefetch_depths) < 0:
        parser.error("Prefetch depths must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    # Keep napari's settings/cache and theme generation inside this workspace.
    import os
    os.environ["NAPARI_CONFIG"] = str(args.output / "settings.yaml")
    from unittest.mock import patch
    from contextlib import ExitStack
    import platformdirs
    with ExitStack() as stack:
        for kind in ("cache", "config", "data", "state", "log"):
            stack.enter_context(patch.object(platformdirs, f"user_{kind}_dir",
                                            return_value=str(args.output)))
        import napari
        import numpy as np
        from napari_animation import Animation
        import napari_animation.prefetch as prefetch_module
        if args.eager_prefetch_full_volume:
            prefetch_module._resident_full_view = lambda layer, array: False
        from qtpy.QtCore import QEventLoop, QTimer
        if args.eager_clip_planes:
            from napari_animation.scene.ortho_slice import OrthoSlice
            original_ensure = OrthoSlice.ensure_layer
            def eager_ensure(section, *pos, **kw):
                mode = section.render_mode
                try:
                    if mode == "clip":
                        section.render_mode = "plane"
                    return original_ensure(section, *pos, **kw)
                finally:
                    section.render_mode = mode
            OrthoSlice.ensure_layer = eager_ensure
        viewer = napari.Viewer(show=not args.hidden)
        animation = Animation(viewer)
        cache_stats = {"hits": 0, "insertions": 0, "inserted_keys": set()}
        cache_lock = None
        cache = None
        if args.cache_stats and prefetch_module._INTERACTIVE_CACHE is not None:
            from threading import Lock
            cache_lock = Lock()
            cache = prefetch_module._INTERACTIVE_CACHE.cache
            original_hit, original_put = cache.hit, cache.put
            def count_hit(key, value):
                with cache_lock:
                    cache_stats["hits"] += 1
                if original_hit is not None:
                    original_hit(key, value)
            def count_put(key, value, *pos, **kw):
                with cache_lock:
                    cache_stats["insertions"] += 1
                    cache_stats["inserted_keys"].add(key)
                return original_put(key, value, *pos, **kw)
            cache.hit, cache.put = count_hit, count_put
        report = {"input": str(args.animation.resolve()),
                  "sha256": hashlib.sha256(args.animation.read_bytes()).hexdigest(),
                  "python": sys.version, "platform": platform.platform(),
                  "napari": napari.__version__, "runs": [],
                  "legacy_viewport": args.legacy_viewport,
                  "eager_clip_planes": args.eager_clip_planes,
                  "eager_prefetch_full_volume": args.eager_prefetch_full_volume,
                  "prefetch_sequence": prefetch_depths}
        try:
            import cProfile
            import psutil
            process = psutil.Process()
            report["rss_before_load"] = process.memory_info().rss
            profiler = cProfile.Profile()
            if args.profile:
                profiler.enable()
            started = time.perf_counter()
            animation.load_keyframes(args.animation)
            report["load_seconds"] = time.perf_counter() - started
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
            report["ready_seconds"] = time.perf_counter() - started
            report["rss_after_ready"] = process.memory_info().rss
            if args.profile:
                profiler.disable()
                profiler.dump_stats(str(args.output / "initialization.prof"))
            native = viewer.window._qt_viewer.canvas.native
            report["native_canvas"] = [native.width(), native.height(), native.devicePixelRatioF()]
            report["sources"] = [dict(name=layer.name, shape=layer.data[0].shape if layer.multiscale else layer.data.shape,
                                       data_level=layer.data_level, contrast_limits=layer.contrast_limits,
                                       scale=list(layer.scale)) for layer in viewer.layers if layer._type_string == "image"]
            from vispy import gloo
            viewer.window._qt_viewer.canvas._scene_canvas.set_current()
            report["gl_renderer"] = str(gloo.gl.glGetParameter(gloo.gl.GL_RENDERER))
            import imageio.v3 as iio
            # Preview evidence is captured after the timed runs, avoiding warmup changes.
            original_count = len(animation._frames)
            indices = (np.linspace(0, original_count - 1, min(args.samples, original_count), dtype=int).tolist()
                       if args.samples else list(range(original_count)))
            report["original_frames"] = original_count
            report["original_frame_indices"] = indices
            from napari_animation.prefetch import read_keys, _read_signature
            signatures = set()
            for state in animation._frames:
                keys = read_keys(viewer, state.dims)
                signatures.add(tuple(_read_signature(arr, key) for arr, key in keys))
            report["prefetch_read_keys"] = [
                {"array_type": type(arr).__module__ + "." + type(arr).__name__,
                 "shape": list(arr.shape), "dtype": str(arr.dtype),
                 "dask_chunks": int(np.prod([len(axis) for axis in arr.chunks]))
                 if hasattr(arr, "chunks") else None,
                 "index": [str(part) for part in key]}
                for arr, key in read_keys(viewer, animation._frames[0].dims)
            ]
            report["unique_prefetch_signatures"] = len(signatures)
            report["unique_nonempty_prefetch_signatures"] = sum(
                bool(signature) for signature in signatures
            )
            if args.samples:
                states = [animation._frames[i] for i in indices]
                template = animation.key_frames[0]
                animation.key_frames.clear()
                animation.key_frames.extend(replace(template, viewer_state=state, steps=1)
                                            for state in states)
            import importlib
            capture_module = importlib.import_module(
                "napari_animation._qt.export_controller" if args.qt else "napari_animation.animation")
            capture_original = capture_module.capture_export_frame
            if args.legacy_viewport:
                class LegacyCaptureSession:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __enter__(self):
                        return self
                    def close(self):
                        pass
                    def __exit__(self, *args):
                        pass
                capture_module.ExportCaptureSession = LegacyCaptureSession
            for run, depth in enumerate(prefetch_depths):
                # Save outside export timing: retain only three frames in RAM.
                lossless = {}
                captured = [0]
                peak_rss = [process.memory_info().rss]
                def capture(*pos, **kw):
                    frame = capture_original(*pos, **kw)
                    index = captured[0]
                    peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
                    if args.lossless and index in {0, len(indices) // 2, len(indices) - 1}:
                        lossless[index] = frame.copy()
                    captured[0] += 1
                    return frame
                capture_module.capture_export_frame = capture
                if cache_lock is not None:
                    with cache_lock:
                        cache_stats.update(hits=0, insertions=0)
                        cache_stats["inserted_keys"].clear()
                io_before = process.io_counters()
                options = dict(path=str(args.output / f"run-{run + 1}.mp4"),
                               render_size=(args.height, args.width),
                               prefetch=depth, diagnose=not args.no_diagnose)
                if args.qt:
                    from napari_animation._qt.export_controller import QtExportController
                    controller = QtExportController(animation, options)
                    loop = QEventLoop()
                    result = []
                    controller.finished.connect(lambda path, error: (result.append(error), loop.quit()))
                    deadline = QTimer()
                    deadline.setSingleShot(True)
                    deadline.timeout.connect(controller.cancel)
                    deadline.start(args.timeout * 1000)
                    controller.start()
                    loop.exec_()
                    deadline.stop()
                    if result[0] is not None:
                        raise result[0]
                else:
                    animation.animate(**options)
                capture_module.capture_export_frame = capture_original
                report["rss_after_export"] = process.memory_info().rss
                io_after = process.io_counters()
                for index, frame in lossless.items():
                    iio.imwrite(args.output / f"run-{run + 1}-frame-{indices[index]}.png", frame)
                data = animation._last_render_perf.snapshot()
                data["metadata"]["original_frame_indices"] = indices
                animation._last_render_perf.metadata["original_frame_indices"] = indices
                animation._last_render_perf.save(options["path"])
                run_report = {k: v for k, v in data.items() if k != "events"}
                run_report["peak_rss_sampled"] = peak_rss[0]
                run_report["process_io_delta"] = {
                    name: getattr(io_after, name) - getattr(io_before, name)
                    for name in ("read_bytes", "write_bytes", "read_count", "write_count")
                }
                if cache_lock is not None:
                    with cache_lock:
                        run_report["dask_cache"] = {
                            "hits": cache_stats["hits"],
                            "insertions": cache_stats["insertions"],
                            "distinct_inserted_keys": len(cache_stats["inserted_keys"]),
                        }
                report["runs"].append(run_report)
                (args.output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"RUN {run + 1}: {data['wall_seconds']:.3f}s", flush=True)
            report["loaded_slices"] = []
            for layer in viewer.layers:
                if layer._type_string != "image":
                    continue
                view = layer._slice.image.view
                sample = np.asarray(view)[::16, ::16, ::16] if view.ndim == 3 else np.asarray(view)[::16, ::16]
                report["loaded_slices"].append(dict(name=layer.name, shape=view.shape, dtype=str(view.dtype),
                                                    sample_min=float(sample.min()), sample_max=float(sample.max())))
            (args.output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            iio.imwrite(args.output / "first-frame.png", iio.imread(args.output / "run-1.mp4", index=0))
        finally:
            if cache is not None:
                cache.hit, cache.put = original_hit, original_put
            animation.close()
            viewer.close()


if __name__ == "__main__":
    main()
