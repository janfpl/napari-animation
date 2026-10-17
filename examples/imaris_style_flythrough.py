"""
Imaris-style fly-through: a volume cut away by clipping planes, with a thick
ortho slice sweeping through it while the camera orbits.

Demonstrates the scene-object API:

* several clipping planes cutting the same volume (they composite, so they
  accumulate rather than overwriting each other)
* an ortho slice -- a centre plane plus a thickness, max-projected -- backed by
  a real napari layer, so it appears in the layer list with its own visibility
  checkbox
* per-keyframe on/off switching of each object
* geometry that sweeps between keyframes while the camera moves
"""

import numpy as np
import napari

from napari_animation import Animation
from napari_animation.scene import VIEW_NORMALS, ClipPlane, OrthoSlice

# a blobby test volume with anisotropic voxels, like a light-sheet stack
z, y, x = np.mgrid[0:64, 0:128, 0:128]
volume = (
    np.sin(x / 9) * np.cos(y / 11) * np.sin(z / 5)
    + 0.6 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2 + (z - 32) ** 2) / 400)
).astype(np.float32)
VOXEL = (2.0, 1.0, 1.0)  # z is twice as coarse

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(
    volume, name="volume", scale=VOXEL, rendering="mip", colormap="magma"
)

animation = Animation(viewer)

# world extent along z, used to sweep things through the whole volume
z_extent = volume.shape[0] * VOXEL[0]
centre = tuple(s * v / 2 for s, v in zip(volume.shape, VOXEL))

# --- scene objects -----------------------------------------------------------
# Two cutaway planes at right angles, opening a corner of the volume. Both
# target the same layer; the compositor unions them.
cut_front = animation.add_scene_object(
    ClipPlane(name="cut front", position=centre, normal=(0.0, 1.0, 0.0))
)
cut_side = animation.add_scene_object(
    ClipPlane(name="cut side", position=centre, normal=(0.0, 0.0, 1.0))
)

# A 10 micron thick max-projected optical section facing XY (sliced along z).
ortho = animation.add_scene_object(
    OrthoSlice(
        name="Ortho XY",
        source="volume",
        position=(0.0, centre[1], centre[2]),
        normal=VIEW_NORMALS["XY"],
        thickness=10.0,
        projection="max",
    )
)

# --- keyframes ---------------------------------------------------------------
# 1. The whole volume, nothing cut, no slice.
animation.scene.apply(viewer)
viewer.camera.angles = (0.0, 0.0, 90.0)
animation.capture_keyframe(steps=1)
for obj in (cut_front, cut_side, ortho):
    animation.set_object_enabled(0, obj.id, False)

# 2. Camera orbits while the two clipping planes open the volume up.
viewer.camera.angles = (-15.0, 25.0, 75.0)
animation.capture_keyframe(steps=45)
animation.set_object_enabled(1, ortho.id, False)

# 3. The ortho slice switches on and sweeps from one end of z to the other
#    while the camera keeps orbiting.
ortho.position = (0.0, centre[1], centre[2])
viewer.camera.angles = (-25.0, 45.0, 60.0)
animation.capture_keyframe(steps=30)

ortho.position = (z_extent, centre[1], centre[2])
viewer.camera.angles = (-10.0, 70.0, 80.0)
animation.capture_keyframe(steps=60)

# 4. Snap back to a plain 2D view and scroll through the stack.
viewer.dims.ndisplay = 2
viewer.dims.set_point(0, 0.0)
animation.capture_keyframe(steps=30)
viewer.dims.set_point(0, z_extent)
animation.capture_keyframe(steps=45)

animation.animate("imaris_style_flythrough.mp4", canvas_only=True)

if __name__ == "__main__":
    napari.run()
