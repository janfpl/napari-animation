"""
Measure the cost of the scene features added in this branch.

Answers three questions:

1. What does each clipping plane cost per frame?
2. What does each ortho slice cost per frame, and in GPU memory?
3. What is the end-to-end render throughput?

Clipping planes and plane depiction are evaluated on the GPU against an
already-uploaded texture, so the *per-frame* cost of both should be close to
nothing -- the expense is a one-off upload and the VRAM it occupies. This
script is how you check that holds on your data and hardware rather than
taking it on faith.

Run headless (state-application cost only, no OpenGL needed)::

    python examples/benchmark_scene.py

Run against a real viewer, which additionally times screenshots and a full
render::

    python examples/benchmark_scene.py --real

Use ``--shape 256 512 512`` to benchmark at the size of your own data.
"""

import argparse
import time

import numpy as np

from napari_animation import Animation
from napari_animation.scene import ClipPlane, OrthoSlice


def make_viewer(real, shape, voxel):
    if real:
        import napari

        viewer = napari.Viewer(ndisplay=3)
    else:
        from napari.components import ViewerModel

        class HeadlessViewer(ViewerModel):
            """No canvas, so screenshots are stubbed and no GL is required."""

            def screenshot(self, *args, **kwargs):
                return np.zeros((512, 512, 4), dtype=np.uint8)

        viewer = HeadlessViewer()
        viewer.dims.ndisplay = 3

    rng = np.random.default_rng(0)
    data = rng.random(shape, dtype=np.float32)
    viewer.add_image(data, name="volume", scale=voxel, rendering="mip")
    return viewer, data


def time_apply(animation, repeats=20):
    """Median wall-clock time to push the whole scene onto the viewer."""
    viewer = animation.viewer
    ortho, _ = animation._capture_state()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        animation.scene.apply(viewer, ortho=ortho)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings)) * 1000.0


def bench_clip_planes(animation, centre, counts):
    print("\nClipping planes -- scene apply cost")
    print(f"  {'planes':>8}  {'apply (ms)':>12}  {'per plane':>12}")
    baseline = None
    for count in counts:
        for obj in list(animation.scene):
            animation.remove_scene_object(obj)
        for i in range(count):
            animation.add_scene_object(
                ClipPlane(name=f"cut {i}", position=centre)
            )
        elapsed = time_apply(animation)
        if baseline is None:
            baseline = elapsed
        marginal = (elapsed - baseline) / count if count else 0.0
        print(f"  {count:>8}  {elapsed:>12.3f}  {marginal:>12.3f}")


def bench_ortho_slices(animation, centre, counts, data):
    print("\nOrtho slices (plane mode) -- scene apply cost and VRAM")
    print(
        f"  {'slices':>8}  {'apply (ms)':>12}  {'per slice':>12}"
        f"  {'est. VRAM':>12}"
    )
    voxel_bytes = data.dtype.itemsize * data.size
    baseline = None
    for count in counts:
        for obj in list(animation.scene):
            animation.remove_scene_object(obj)
        for i in range(count):
            animation.add_scene_object(
                OrthoSlice(
                    name=f"Ortho {i}",
                    source="volume",
                    position=centre,
                    thickness=5.0,
                )
            )
        elapsed = time_apply(animation)
        if baseline is None:
            baseline = elapsed
        marginal = (elapsed - baseline) / count if count else 0.0
        # each plane-mode slice is a separate 3D texture of the same volume
        vram = (count + 1) * voxel_bytes / 1e9
        print(
            f"  {count:>8}  {elapsed:>12.3f}  {marginal:>12.3f}"
            f"  {vram:>10.2f} GB"
        )


def bench_render(animation, centre, path, frames):
    """End-to-end render throughput, including screenshots and encoding."""
    print(f"\nFull render -- {frames} frames -> {path}")
    viewer = animation.viewer
    animation.key_frames.clear()

    viewer.camera.angles = (0.0, 0.0, 90.0)
    animation.capture_keyframe(steps=1)
    viewer.camera.angles = (-20.0, 45.0, 70.0)
    animation.capture_keyframe(steps=frames - 1)

    start = time.perf_counter()
    animation.animate(path, fps=20, canvas_only=True)
    elapsed = time.perf_counter() - start
    print(f"  {frames / elapsed:.2f} frames/s over {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="use a real napari viewer (needs OpenGL); also times a render",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=(64, 256, 256),
        metavar=("Z", "Y", "X"),
        help="volume shape to benchmark",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        nargs=3,
        default=(2.0, 1.0, 1.0),
        metavar=("Z", "Y", "X"),
        help="voxel size in world units",
    )
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--output", default="benchmark.mp4")
    args = parser.parse_args()

    viewer, data = make_viewer(args.real, tuple(args.shape), tuple(args.voxel))
    animation = Animation(viewer)
    centre = tuple(s * v / 2 for s, v in zip(args.shape, args.voxel))

    print(
        f"volume {tuple(args.shape)} {data.dtype} "
        f"({data.nbytes / 1e9:.2f} GB), voxel {tuple(args.voxel)}"
    )
    print(f"viewer: {'real napari' if args.real else 'headless ViewerModel'}")

    bench_clip_planes(animation, centre, [0, 1, 2, 4, 8])
    bench_ortho_slices(animation, centre, [0, 1, 2, 3], data)

    for obj in list(animation.scene):
        animation.remove_scene_object(obj)

    if args.real:
        bench_render(animation, centre, args.output, args.frames)
    else:
        print(
            "\nSkipping the render benchmark: it needs a real viewer. "
            "Re-run with --real."
        )


if __name__ == "__main__":
    main()
