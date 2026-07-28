# Making Imaris-style animations

## The one idea that makes the rest make sense

**Scene objects are live things. Keyframes are snapshots of them.**

There is only ever *one* clipping plane object in the 3D scene list. It does
not belong to a keyframe. What belongs to a keyframe is a *recording* of where
that plane was, and whether it was switched on, at the moment you pressed
**Capture**.

So every workflow is the same three beats:

1. **Set the scene up** — move planes, change the camera, toggle things on/off.
   The canvas updates live.
2. **Capture.** Everything currently true is written into a new keyframe:
   camera, every scene object, every layer setting.
3. **Change something and capture again.** The animation is the interpolation
   between what you recorded.

If you add a clipping plane and nothing seems to "stick", it is almost always
because step 2 was skipped. The scene list is *space*; the keyframe list is
*time*.

## Where things are

The panel scrolls; these are the sections top to bottom.

| Section | What it does |
|---|---|
| **Capture / Delete** | Capture writes a keyframe. `Alt-F` does the same. |
| Keyframe list | Your timeline. Each row has an **Overwrite** button that replaces that keyframe with the current view. |
| **Steps / Ease** | How many frames it takes to reach the *selected* keyframe from the one before, and the easing curve. |
| **Voxel size** | Fix wrong physical spacing (e.g. a TIFF's Z step). Do this first. |
| **3D scene** | Clipping planes and ortho slices: add, remove, rename, switch on/off, edit. |
| **View and scroll-through** | Snap 2D/3D, and auto-generate sweeps. |
| **Save Animation** | Renders the movie. |

---

## Walkthrough 1 — a keyframe with the volume cut away

Goal: the whole volume, then a plane cutting into it.

**Set up.** Load your volume, switch to 3D (the **3D** button under *Snap
view*, or napari's own 2D/3D toggle). If the proportions look wrong, fix
**Voxel size** now — clipping planes are positioned in world units, so
everything downstream depends on it.

**1. Capture the uncut volume.**

- Set **Steps** to `1` (this is the *first* keyframe, so nothing interpolates
  into it).
- Press **Capture**. Keyframe 1 exists.

**2. Add the plane.**

- Press **+ Plane**. The volume is immediately cut in half — the plane is
  created at the centre of your data, facing +Z.
- The cut is live but *not yet in any keyframe*.

**3. Aim it.**

- **Facing** picks the axis: `+Z`, `-Z`, `+Y`… Data on the side the normal
  points *away* from is what gets removed.
- **Flip side** if it cut the half you wanted to keep.
- **Position Z/Y/X** moves the plane. Drag the Z box and watch the cut travel.
- **Cuts** restricts it to one layer, or leaves it cutting everything.

**4. Capture the cut version.**

- Set **Steps** to something like `45` — that is how many frames the transition
  takes.
- Press **Capture**. Keyframe 2.

Drag the frame slider at the bottom. The cut should sweep smoothly in over 45
frames, not pop.

### Making it appear rather than sweep

Right now the plane exists in both keyframes and slides. If instead you want
the volume whole and *then* cut, with no plane visible at first:

1. Select keyframe 1 in the list.
2. Untick the plane in the 3D scene list.
3. Press that keyframe's **Overwrite** button.

Keyframe 1 now records "plane off", keyframe 2 records "plane on". The switch
happens on arrival at keyframe 2 — cleanly, once, never half-applied.

That untick-and-overwrite move is the per-keyframe on/off from Imaris, and it
works the same for ortho slices and for ordinary napari layers.

---

## Walkthrough 2 — an ortho slice scrolling through in 3D

Goal: a thick optical section travelling through the volume while the camera
orbits.

**Be in 3D first.** Press **3D** under *Snap view*. A plane-mode slice hides
itself in 2D, because there it would just redraw its source layer on top of
itself — so if you add one in 2D, nothing appears and it looks broken.

**1. Add the slice.**

- Press **+ Slice**. A slab appears, and a new layer called *Ortho slice*
  shows up in **napari's own layer list**. It really is a layer: you can toggle
  and reorder it there like any other.

**2. Shape it.**

- **Source** — which layer's data it shows.
- **View** — `XY` faces down Z, `XZ` down Y, `YZ` down X. Or type a **Normal**
  by hand for an oblique section.
- **Thickness** — in world units, measured along the normal. At 2 µm Z
  spacing, a 10 µm XY slab is 5 planes thick.
- **Projection** — `max` / `mean` / `min` through that thickness.
- **Render as** — leave on `plane`. (`clip` carves the slab out of the source
  volume instead of drawing it alongside; no extra GPU memory, but you lose
  the rest of the volume.)

**3. Generate the sweep.** In *View and scroll-through*:

- Pick your slice in the **Slice** dropdown.
- Set **Steps** — the frame count for the sweep.
- Press **Add slice sweep**.

Two keyframes appear, and the slab travels from one face of the volume to the
other. The travel is computed along the slice's own normal, so an oblique slab
still crosses the whole volume.

**4. Add the camera orbit.** The sweep captured both keyframes with your
*current* camera, so the view is static. To make it orbit:

1. Select the **second** of the two new keyframes.
2. Rotate the camera in the canvas to where you want it to end up.
3. Press that keyframe's **Overwrite** button.

Now scrub: the slab travels while the view rotates. That is the Imaris effect.

### Doing it by hand instead

`Add slice sweep` is a shortcut for something you can do manually, and manual
is easier when the camera and slab should move together from the start:

1. Set the camera where you want to begin. Set **Centre Z** to one end of the
   volume. **Capture** (Steps `1`).
2. Rotate the camera. Set **Centre Z** to the other end. **Capture**
   (Steps `60`).

Same result, one fewer step.

---

## Two other sweeps

**Slider scroll-through** — the flat, slice-by-slice pass. Press **XY** under
*Snap view* to flatten to 2D, then **Add scroll-through**. The **Range** is
prefilled with the full extent of the data; narrow it to sweep part of the
stack. Works in 3D too, where it moves napari's own slice plane.

**Combining.** Nothing is exclusive. Cutaway planes, an ortho slice, and a
camera orbit can all be active in one keyframe — the clipping planes are
composited, so several cuts accumulate on the same volume rather than
replacing one another.

---

## Things that trip people up

**"I added a plane but the animation doesn't show it."** You did not capture
after adding it. Adding changes the live scene, not the timeline.

**"My ortho slice is invisible."** You are in 2D. Press **3D**.

**"The cut jumps instead of sweeping."** The two keyframes have the plane at
the same place, or **Steps** on the second keyframe is `1`.

**"Everything moved when I fixed the voxel size."** Expected — you changed the
physical size of the data. The correction is written into keyframes you already
captured, so nothing reverts mid-render.

**"I renamed my ortho slice and the layer name changed."** Also expected: the
scene object and its napari layer share a name.

**Ortho slices cost GPU memory.** Each `plane`-mode slice is a separate texture
of the same volume. Three slices on a 500³ float32 volume is about 1.5 GB on
top of the source. If you run out, switch one to `clip` mode or point it at a
downsampled copy.
