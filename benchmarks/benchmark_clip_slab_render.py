"""Time 1080p 3D captures with 0-3 orthogonal clip-mode slabs per channel.

usage: python benchmarks/benchmark_clip_slab_render.py animation.json out_dir

Opens the layers recorded in a saved animation, applies the appearance and
camera of its fourth keyframe, and times capture_export_frame with no slabs
and with one, two and three orthogonal 150-voxel clip slabs per channel.
Saves a screenshot per slab count to out_dir.
"""

import json, sys, time
import numpy as np, napari
from qtpy.QtWidgets import QApplication
from napari_animation import Animation
from napari_animation.render_capture import capture_export_frame
from napari_animation.scene import OrthoSlice

d = json.load(open(sys.argv[1]))
v = napari.Viewer()
v.window._qt_window.resize(1600, 1000)
for rec in d["layers"]:
    v.open(rec["path"])
for rec, layer in zip(d["layers"], v.layers):
    st = d["key_frames"][3]["viewer_state"]["layers"][rec["id"]]
    layer.colormap = st["colormap"]
    layer.contrast_limits = st["contrast_limits"]
    layer.gamma = st["gamma"]
    layer.blending = st["blending"]
    layer.rendering = "mip"
v.dims.ndisplay = 3
cam = d["key_frames"][3]["viewer_state"]["camera"]
v.camera.angles = cam["angles"]
v.camera.zoom = cam["zoom"]
v.camera.center = cam["center"]
a = Animation(v)
for _ in range(10):
    QApplication.processEvents()


def timed(n=5):
    capture_export_frame(v, render_size=(1080, 1920))
    s = []
    for _ in range(n):
        t = time.perf_counter()
        capture_export_frame(v, render_size=(1080, 1920))
        s.append(time.perf_counter() - t)
    return float(np.median(s))


res = {"no_slabs": timed()}
print("no slabs", res["no_slabs"], flush=True)
centre = (697.0, 1024.0, 1024.0)
specs = [("XY", (1.0, 0, 0)), ("XZ", (0, 1.0, 0)), ("YZ", (0, 0, 1.0))]
for k in range(1, 4):
    a.scene.clear()
    for name, normal in specs[:k]:
        for src in ("C1-no_bg", "C2-25px_bg"):
            a.add_scene_object(
                OrthoSlice(
                    name=f"{name} {src}",
                    source=src,
                    position=centre,
                    normal=normal,
                    thickness=150.0,
                    render_mode="clip",
                )
            )
    for _ in range(5):
        QApplication.processEvents()
    res[f"{k}_slabs"] = timed()
    print(
        k,
        "slab(s) per channel",
        res[f"{k}_slabs"],
        "planes/channel",
        len(v.layers["C1-no_bg"].experimental_clipping_planes),
        flush=True,
    )
    v.screenshot(
        path=sys.argv[2] + f"/slabs_{k}.png", canvas_only=True, flash=False
    )
print(json.dumps(res, indent=1))
v.close()
