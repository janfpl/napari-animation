# Plan: Imaris-equivalent 3D animation capabilities

Target: the kind of movie produced by Imaris' Animation tab — a volume-rendered
dataset that is progressively cut away by several clipping planes, with
orthogonal slice planes of a defined thickness sweeping through the volume,
while the camera orbits, and with every scene object individually switchable
on/off at each keyframe. (Reference: Supplementary Movie 4 of *"Brain-wide
reconstruction of inhibitory circuits after traumatic brain injury"*, Nat.
Commun. 13, 3417 (2022) — a cleared whole-brain light-sheet volume.)

This document is the design/implementation plan.

**Status:** Phases 0, 1 and 2 (§5) are implemented and tested, including
their Qt widgets — the state-capture and interpolation foundations, N
animatable clipping planes with the compositor, and ortho slices backed by real
napari layers, all drivable from the GUI. Phases 3–5 are still design only. See
`examples/imaris_style_flythrough.py` for the API in use.

---

## 1. Requirements, restated

| # | Requirement | Imaris equivalent |
|---|---|---|
| R1 | A 3D volume cut away by **N** clipping planes, each with arbitrary position + orientation, animatable | Clipping Plane objects |
| R2 | **Ortho slicers**: a slab defined by a centre plane + thickness, projected (max/mean/min), that scrolls through the volume as the camera interpolates | Ortho Slicer objects |
| R3 | Toggle between **2D and 3D**; snap the window to a flat 2D image and animate scroll-throughs in either mode | 2D/3D view switch |
| R4 | Ortho slicers and clipping planes behave like **napari layers**: they appear in a list and can be switched **on/off per keyframe** | Object tree + keyframe visibility table |

---

## 2. Where the codebase stands today

`napari-animation` records a `ViewerState` (camera dict, dims dict, per-layer
state dict, plus an `ortho` dict) into each `KeyFrame`, and `FrameSequence`
walks that nested dict and interpolates every leaf between consecutive
keyframes. That architecture is a good fit for all four requirements — but
five concrete things block them. Each of the following was verified against
napari 0.8.0.

### B1 — `depiction` and `plane` are never captured

`ViewerState.from_viewer` captures `layer._get_base_state()`
(`viewer_state.py:75-79`). On napari 0.8 that yields:

```
affine, axis_labels, blending, experimental_clipping_planes, metadata,
name, opacity, projection_mode, rotate, scale, shear, translate, units, visible
```

`layer._get_state()` additionally has:

```
attenuation, colormap, contrast_limits, custom_interpolation_kernel_2d, data,
depiction, gamma, interpolation2d, interpolation3d, iso_threshold, multiscale,
plane, rendering, rgb
```

`depiction` and `plane` are napari's *native* ortho-slicer primitive
(`SlicingPlane` has exactly `position`, `normal`, `thickness`) and they are
invisible to the animation engine. So is every appearance property
(`colormap`, `contrast_limits`, `gamma`, `rendering`, `iso_threshold`), which
Imaris animations routinely ramp.

### B2 — Clipping planes snap instead of interpolating

`experimental_clipping_planes` *is* captured, as a `list[dict]`. But
`interpolation.default` (`interpolation.py:34-39`) has no dict branch, so a
list of plane dicts goes: `list` → `interpolate_seq` → each element is a
`dict` → falls through to `interpolate_bool` → returns the target value the
moment `fraction > 0`. Verified:

```python
default([{'position': (0,0,0), ...}], [{'position': (10,0,0), ...}], 0.5)
# -> [{'position': (10.0, 0.0, 0.0), ...}]   # jumped, did not sweep
```

A keyframed cutaway therefore **pops** rather than sweeping.

### B3 — Plane lists of unequal length silently truncate

`interpolate_seq` (`interpolation.py:58`) zips the two sequences. Going from
two clipping planes to one drops a plane instantly:

```python
default([planeA, planeB], [planeC], 0.5)  # -> length 1
```

Positional matching is wrong for objects that get added and removed across a
timeline. Planes need stable identity.

### B4 — The current ortho slicer *owns* every layer's clipping planes

`OrthoSlicer._apply_clip` assigns the whole list
(`ortho_slicer.py:265`) and `_clear_clipping_planes` resets it to `[]`
(`ortho_slicer.py:296-301`), for **every layer in the viewer**. So the
existing slab and any user-authored cutaway planes cannot coexist — one wipes
the other. It is also a single global slicer (`animation.py:92-94` captures
`ortho` only when `enabled`), axis-aligned only, and not per-layer.

### B5 — 2D↔3D switching clobbers the camera

`ViewerState.apply` applies **camera first, then dims**
(`viewer_state.py:96-97`). On napari 0.8, `ViewerModel._on_ndisplay_changed`
overwrites `camera.center/zoom/angles` from a per-mode cache (or
`fit_to_view()`) whenever `camera.synced` is `False`. Applying a state whose
`dims.ndisplay` differs from the live viewer therefore **discards the camera
that was just set**.

Separately, `ndisplay` is an `int`, so `interpolate_num` truncates:
`default(2, 3, 0.99) == 2`. The switch happens to land on the final frame of a
segment by accident, not by design, and camera angles/zoom interpolate across
the switch as if the two modes were commensurable.

---

## 3. Architecture

### 3.1 The scene-object model

Introduce a `Scene` — an evented, ordered list of `SceneObject`s that live
alongside the napari layer list and are captured into every keyframe.

```
napari_animation/scene/
    __init__.py
    base.py          # SceneObject
    ortho_slice.py   # OrthoSlice
    clip_plane.py    # ClipPlane
    scene.py         # Scene (list + compositor)
```

```python
@dataclass
class SceneObject:
    id: str                       # uuid4, stable across keyframes
    name: str                     # user-facing, editable
    enabled: bool = True          # the per-keyframe on/off switch
    targets: tuple[str, ...] = () # layer names; () == all image layers

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "SceneObject": ...
```

Two properties make this work:

* **Stable `id`.** Keyframe state stores `{id: params}`, so interpolation
  matches by identity, not list position — this is the fix for B3. An object
  missing from one of the two keyframes is treated as `enabled=False` holding
  its last known geometry, so it fades out of the scene instead of teleporting
  or vanishing.
* **Everything is a plain dict of floats/bools.** It drops straight into the
  existing `keys_to_list` walk in `FrameSequence._interpolate_state`, so it
  interpolates and serialises with the machinery that already exists.

### 3.2 Compositing, not assignment (fixes B4)

The single most important rule: **no scene object ever assigns
`layer.experimental_clipping_planes` directly.** `Scene.apply(viewer)`
gathers the contributions of every enabled object, groups them per target
layer, and performs exactly one assignment per layer:

```python
def apply(self, viewer, state=None):
    contributions = defaultdict(list)          # layer name -> [plane dicts]
    for obj in self._objects_for(state):
        if not obj.enabled:
            continue
        for layer_name, planes in obj.clipping_planes(viewer):
            contributions[layer_name].extend(planes)
    for layer in viewer.layers:
        if not hasattr(layer, "experimental_clipping_planes"):
            continue
        layer.experimental_clipping_planes = contributions.get(layer.name, [])
```

napari 0.8 accepts an arbitrary number of clipping planes per layer (verified
with three), so N cutaways plus a slab-bounded ortho slice compose naturally.
This directly satisfies R1.

Because the scene now owns clipping planes, `experimental_clipping_planes` is
**dropped from the captured per-layer state** so the two do not fight. On the
first capture of a viewer that already has planes set (by hand or by
napari-threedee), they are adopted into the scene as `ClipPlane` objects, so
nothing is lost.

### 3.3 `ClipPlane` (R1)

```python
@dataclass
class ClipPlane(SceneObject):
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)  # world coords
    normal:   tuple[float, float, float] = (1.0, 0.0, 0.0)
```

`clipping_planes()` converts world → each target layer's data coordinates
(reusing the scale/translate maths already in `ortho_slicer.py:250-256`, which
handles layers with fewer dims than the viewer). Interpolating `position`
sweeps the cut; interpolating `normal` (spherically, see §3.6) tilts it.

### 3.4 `OrthoSlice` (R2)

```python
@dataclass
class OrthoSlice(SceneObject):
    source: str | None = None       # layer whose data is sliced
    position: tuple = (0, 0, 0)     # centre of the slab, world coords
    normal:   tuple = (1, 0, 0)     # XY/XZ/YZ presets, or any oblique vector
    thickness: float = 1.0          # world units
    projection: str = "max"         # max | mean | min
    render_mode: str = "plane"      # plane | clip | dims
    follow_dims: bool = False       # lock centre to the dims slider
```

Three render modes, because they trade off differently:

**`"plane"` — the real Imaris ortho slicer, and the default.** Backed by a
*real napari Image layer* that shares the source layer's data array (no copy)
with `depiction='plane'`. The mapping is exact:

| OrthoSlice | napari |
|---|---|
| `position` | `layer.plane.position` (converted to data coords) |
| `normal` | `layer.plane.normal` |
| `thickness` | `layer.plane.thickness` (÷ voxel size along the normal) |
| `projection` | `layer.rendering`: `max`→`mip`, `mean`→`average`, `min`→`minip` |
| `enabled` | `layer.visible` |

This renders a thick oblique slab at an arbitrary orientation in 3D — exactly
"a center plane showing a projection of a certain thickness". Because it is a
genuine napari layer it appears in the layer list with a visibility checkbox,
which is half of R4 for free.

**`"clip"` — zero extra VRAM.** Two opposed clipping planes bounding the slab,
contributed to the *source* layer through the compositor. No second texture,
but the rest of the volume is cut away rather than shown alongside the slab.

**`"dims"` — the 2D thick slice.** The current `margin_left`/`margin_right` +
`projection_mode` path, which is the right mechanism when `ndisplay == 2`.
`Scene.apply` auto-falls back to this in 2D, since plane depiction is
meaningless there.

The backing layer is the **single source of truth** for the slab geometry.
`OrthoSlice` writes into `layer.plane` / `layer.rendering` and reads back out of
them at capture time, rather than holding an authoritative private copy. That
one decision means *any* external tool that moves the plane — a
napari-threedee manipulator, napari's own layer controls, a user script — is
picked up automatically when a keyframe is captured, with no glue code. See
§4.

**Scrolling as the camera interpolates** is then just interpolation: keyframe A
with `camera.angles=(0,0,90)` and `position` at one end of the volume,
keyframe B with `camera.angles=(0,45,90)` and `position` at the other. Camera
angles slerp, `position` interpolates linearly, and the slab sweeps while the
camera orbits. `follow_dims=True` reproduces today's behaviour of locking the
slab to the dims slider.

> **VRAM caveat (must be documented in the UI).** Each `"plane"`-mode slicer is
> a separate napari layer and therefore a separate GPU 3D texture of the same
> volume. Three slicers on a 500³ float32 whole-brain volume ≈ 3 × 500 MB.
> Mitigations to ship: a per-slicer **source level** selector (point the slicer
> at a downsampled pyramid level — visually near-identical for a projected
> slab), and `"clip"` mode which adds no texture at all. The widget should show
> estimated VRAM per slicer.

### 3.5 View mode: 2D ↔ 3D (R3)

Add an explicit `view_mode` to the captured state rather than relying on
`dims.ndisplay` numerics:

1. **Interpolate `ndisplay` as a discrete value.** Register
   `"dims.ndisplay": Interpolation.STEP_END` (new easing-like interpolators
   `STEP_START` / `STEP_END`, switching at the beginning or end of the
   transition) so the flip is a deliberate authored choice, not int truncation.
2. **Fix apply ordering (B5).** `ViewerState.apply` becomes: dims → scene →
   camera, with `camera.synced` applied *before* `ndisplay` changes. A tiny
   `_apply_ndisplay_first` helper sets `dims.ndisplay`, lets napari's
   `_on_ndisplay_changed` do whatever it wants, and only then writes the
   keyframe's camera on top. This must be feature-detected: `camera.synced`
   exists on napari 0.8 but not on older releases.
3. **"Snap to 2D" button + `animation.snap_2d(axis="XY")`.** Sets
   `ndisplay=2`, orients `dims.order` for the requested plane, calls
   `viewer.fit_to_view()` (napari ≥0.8; `reset_view()` on older), and captures
   a keyframe. `snap_3d()` is the inverse.
4. **Scroll-through helper.** `animation.add_scroll_through(axis, start, stop,
   steps, ease=...)` emits two keyframes differing only in `dims.point[axis]`.
   This already interpolates correctly today and works in both 2D and 3D — in
   3D combined with an `OrthoSlice` it *is* the Imaris slice fly-through.

### 3.6 Interpolation engine changes

In `interpolation.py`:

* `interpolate_dict(a, b, fraction)` — recurse into dicts key-wise; add a dict
  branch to `default()`. Fixes B2.
* `interpolate_seq` — pad/hold rather than `zip`-truncate when lengths differ.
  Fixes B3 for any remaining positional lists.
* `slerp_vector(a, b, fraction)` — normalise both, spherically interpolate,
  renormalise. Registered for `normal` fields so planes *rotate* smoothly
  instead of swinging through the origin. (The existing `slerp` operates on
  Euler angles and is not applicable.)
* `STEP_START` / `STEP_END` for discrete values (`ndisplay`, `enabled`,
  `projection`, `render_mode`).

In `frame_sequence.py`, `state_interpolation_map` (`frame_sequence.py:47-50`)
gains **glob support** so per-object rules can be expressed without knowing
ids:

```python
{
    "camera.angles":        Interpolation.SLERP,
    "camera.zoom":          Interpolation.LOG,
    "dims.ndisplay":        Interpolation.STEP_END,
    "scene.*.normal":       Interpolation.SLERP_VECTOR,
    "scene.*.enabled":      Interpolation.STEP_END,
    "scene.*.render_mode":  Interpolation.STEP_END,
    "layers.*.colormap":    Interpolation.STEP_END,
}
```

Matching is a simple `fnmatch` over the dotted path, longest literal prefix
wins.

### 3.7 Per-keyframe on/off (R4)

Two complementary pieces:

1. **`enabled` is already a bool in the captured state**, and step
   interpolation makes it a clean switch — so once the scene is captured per
   keyframe, toggling works with zero extra machinery. Same for napari layers,
   whose `visible` is *already* captured today.
2. **An Imaris-style toggle matrix widget**: rows = scene objects + napari
   layers, columns = keyframes, cells = checkboxes. This is the piece that
   makes it feel like Imaris, where you scan across a row to see when an object
   is on.

The API behind the checkbox:

```python
animation.set_object_enabled(keyframe_index, object_id, enabled)
animation.set_layer_visible(keyframe_index, layer_name, visible)
```

Implementation note: `ViewerState` is a frozen dataclass, so these must
`dataclasses.replace()` the state, write it back onto the `KeyFrame`, and then
invalidate the interpolation cache — `FrameSequence._rebuild_frame_index()`,
the same path `Animation.overwrite_keyframe` already uses
(`animation.py:136-139`).

---

## 4. Relationship to napari-threedee

The question was raised whether
[napari-threedee](https://github.com/napari-threedee/napari-threedee) (n3d) is
a better starting point than building plane handling ourselves. I read its
source at `src/napari_threedee/`. The answer is: **use it, but as an optional
authoring layer on top of this design — not as the foundation.** Four findings
drive that.

**1. Its render-plane manipulator drives the exact same napari state this plan
already targets.** `RenderPlaneManipulator` is a vispy gizmo whose entire
model-facing surface is three lines:

```python
def _while_dragging_translator(self):
    self.layer.plane.position = self.layer.world_to_data(self.origin)

def _while_dragging_rotator(self):
    self.layer.plane.normal = world_to_data_normal(self.z_vector, layer=self.layer)
```

It owns no state. napari core owns `layer.plane`; n3d just gives you a
draggable handle for it. So it is not an alternative to the `OrthoSlice` model
— it is a nicer input device for it. Because §3.4 makes the backing layer the
source of truth, the two compose with **zero glue**: drag the gizmo, hit
"capture keyframe", and the new position is recorded.

**2. It does not do clipping planes at all.** `grep -rn clipping src/` over the
entire n3d tree returns nothing. The clipping-plane manipulator referenced in
blog posts and search results is
[napari-threedee#17](https://github.com/napari-threedee/napari-threedee/issues/17),
still open and unimplemented. *(An earlier revision of this plan cited it as
shipped — that was wrong.)* R1 is entirely ours to build regardless.

**3. It never touches `plane.thickness`.** n3d manipulates position and normal
only. The "projection of a certain thickness" half of R2 — `plane.thickness`
plus the `rendering` mode mapping in §3.4 — has no n3d equivalent.

**4. Its dependency footprint is unacceptable for a hard dependency.**

```
einops, imageio, libigl, magicgui, morphosamplers, mrcfile, napari>=0.5.0,
numpy, pandas, pooch, psygnal, pydantic, qtpy, scipy, superqt, vispy, zarr<3
```

That pulls cryo-EM-specific packages (`morphosamplers`, `mrcfile`, `libigl`)
into a general-purpose animation plugin. Worse, the hard `zarr<3` pin drags in
`asciitree`, which does not build on Python ≥3.11 — **napari-threedee could not
be installed in this environment at all**, on either attempt. Making it
required would immediately break `napari-animation` installs on current Python.

Also note the manipulator needs a live Qt canvas, vispy visuals, and mouse
callbacks, so anything built on it is untestable in the headless `ViewerModel`
harness the rest of this plan (and the existing test suite) relies on.

### Decision

| Concern | Owner |
|---|---|
| Plane geometry state, keyframing, interpolation | napari-animation (this plan) |
| Thickness + projection rendering | napari-animation, via napari core `plane.thickness` / `rendering` |
| Clipping planes (all of R1) | napari-animation — n3d has none |
| **Interactive in-canvas plane dragging** | **napari-threedee, optional** |

Integration lands in Phase 2 as a soft dependency:

```python
# napari_animation/scene/_manipulators.py
def attach_plane_manipulator(viewer, layer):
    try:
        from napari_threedee.manipulators import RenderPlaneManipulator
    except ImportError:
        return None          # UI shows a "pip install napari-threedee" hint
    return RenderPlaneManipulator(viewer, layer=layer, enabled=True)
```

Declared as `pip install napari-animation[threedee]`, surfaced as an "Edit in
canvas" button on each ortho-slice panel, and gracefully absent otherwise.
n3d's `RenderPlaneManipulator._on_depiction_change` already auto-enables itself
when `layer.depiction == 'plane'`, which is exactly the state our backing
layers are in.

Two adjacent things worth tracking rather than duplicating: napari's own
in-progress clipping-planes control widget
([napari#7993](https://github.com/napari/napari/pull/7993)), and n3d's
`CameraSpline`, which drives the camera along an annotated spline path. The
latter is a genuinely different authoring paradigm from keyframes and could
later feed a keyframe generator, but it is out of scope here.

---

## 5. Phased implementation

Each phase is independently shippable and independently testable. All tests can
run headless against `napari.components.ViewerModel` — no GL context, which is
how `_tests/test_ortho_slicer.py` already works.

### Phase 0 — Foundations ✅ *implemented*

No user-visible feature; everything else depends on it.

* **Curated per-layer capture** (`viewer_state.py`): `_get_base_state()` plus
  `ANIMATABLE_LAYER_PROPERTIES` — `depiction`, `plane`, `rendering`,
  `contrast_limits`, `gamma`, `iso_threshold`, `attenuation`,
  `interpolation2d/3d` — and `colormap` captured *by name* (the expanded form
  is colour arrays that neither interpolate nor survive JSON). Explicitly
  excludes `data`: `FrameSequence` deep-copies state per frame, so capturing
  the array would copy the whole volume once per frame. Degrades per layer
  type (Points has none of these; Labels has some).
* **`interpolate_dict`** (`interpolation.py`) — fixes B2. `plane` and
  clipping-plane dicts now interpolate field by field instead of snapping.
* **Non-truncating `interpolate_seq`** — fixes B3. Unequal-length sequences
  interpolate over the common prefix and *hold* the remainder.
* **`slerp_vector`** — plane normals rotate along the shortest arc instead of
  collapsing toward the origin mid-turn. Handles opposed and zero vectors.
* **`STEP_START` / `STEP_END`** — discrete transitions. `STEP_END` holds the
  starting value for the whole transition and switches on arrival.
* **Glob matching** in `state_interpolation_map` via `resolve_interpolation`
  (`frame_sequence.py`): exact keys win, then the most specific pattern.
  Registers `dims.ndisplay → STEP_END` (fixes the int-truncation half of B5)
  and `layers.*.plane.normal → SLERP_VECTOR`.
* **Apply ordering fixed** (`viewer_state.py`) — dims → layers → camera, so a
  2D/3D switch can no longer discard the keyframe's camera (B5). Per-property
  `setattr` is now individually guarded so one unsettable property cannot
  abandon the rest of the frame.

*Tests:* `_tests/test_animatable_state.py`, 22 tests, all headless against
`ViewerModel`. Note the camera regression test must use an **interpolated**
state whose camera differs from anything napari has cached — capturing and
re-applying the same state passes under both orderings, because napari's
per-mode cache happens to hold the right answer.

*Still outstanding from this phase:* bump `setup.cfg` from `napari>=0.4.8`.
The code already requires `dims.margin_left` and `projection_mode`, so the
floor is really `napari>=0.5`, with 0.8-only features feature-detected.

### Phase 1 — Clipping planes (R1) ✅ *model implemented; widget outstanding*

* **`napari_animation/scene/`** — `SceneObject` (base, stable uuid, `enabled`,
  `targets`, kind registry), `ClipPlane`, and `Scene` (a
  `SelectableEventedList`).
* **The compositor** (`scene/scene.py`) — `apply_scene_state` is the only code
  that assigns `experimental_clipping_planes`. Every source *contributes*
  planes and the compositor unions them per layer, which is what makes N
  cutaways coexist (fixes B4). The ortho slicer's clip-mode slab now
  contributes through `OrthoSlicer.clip_contributions` instead of assigning,
  so an optical section and cutaway planes are simultaneously active.
* **World↔data transforms** — positions via `layer.world_to_data`; normals via
  the transpose of the layer-to-world linear matrix, so anisotropic voxels
  don't skew plane orientation.
* **`ViewerState.scene`** keyed by object id, captured by `Animation`. When a
  scene exists, `experimental_clipping_planes` is dropped from the per-layer
  capture so the two cannot fight; without one, the old per-layer behaviour is
  untouched.
* **`Scene.adopt_existing_clipping_planes`** so planes set by hand or by
  another plugin become real scene objects instead of being wiped.
* **`Animation.add_scene_object(..., backfill=True)`** records a new object in
  the keyframes captured so far, so adding a plane doesn't make it vanish when
  scrubbing to an older keyframe.
* **`Animation.set_object_enabled` / `set_layer_visible`** — the per-keyframe
  toggle (most of R4's model layer, ahead of schedule). Both rebuild the
  interpolation cache, since `ViewerState` is frozen and gets replaced.

*Tests:* `_tests/test_scene.py`, 20 tests. Two notes for whoever picks this up:
`@dataclass` resets `__hash__` to `None` on every subclass, which makes objects
unusable in napari's selectable evented list — `register_kind` restores it.
And `ViewerModel` has no `screenshot`, so testing the capture API headlessly
needs the small `HeadlessViewer` subclass in that file.

*Widget:* `_qt/scene_widget.py` — an Imaris-style object list with
add/remove/rename, a per-object on/off tick, and a `ClipPlaneEditor` with
position and normal spin boxes, orientation presets, a flip button and a
target-layer selector.

### Phase 2 — Ortho slices (R2) ✅ *model implemented; widget outstanding*

* **`OrthoSlice`** (`scene/ortho_slice.py`) in `"plane"` and `"clip"` modes.
  `"plane"` creates a real napari Image layer sharing the source array (no
  copy) with `depiction='plane'`; `"clip"` contributes a pair of opposed
  planes to the compositor and creates no layer.
* **Projection → rendering**: `max`→`mip`, `mean`→`average`, `min`→`minip`.
  Projecting through the slab *is* the napari rendering mode.
* **Thickness in world units**, converted per layer by projecting the world
  offset onto the data-space normal — correct for oblique planes through
  anisotropic voxels, which dividing by a single axis' scale is not.
* **The backing layer is the source of truth**, per §4. `sync_from_viewer`
  reads `layer.plane` back before each capture, so a plane moved by a
  napari-threedee manipulator or napari's own controls is what gets recorded.
  It compares against the geometry last written in `apply()` to tell an
  external edit from a programmatic one — reading back unconditionally would
  silently discard changes made through the object.
* **Optional napari-threedee integration** (`scene/_manipulators.py`),
  guarded so its absence is never fatal.
* Disabling hides the backing layer rather than deleting it; deleting layers
  mid-animation is disruptive. `Animation.remove_scene_object` deletes it.

*Tests:* `_tests/test_ortho_slice.py`, 18 tests.

*Widget:* `OrthoSliceEditor` in `_qt/scene_widget.py` — source layer, view
preset (XY/XZ/YZ/custom), centre and normal, thickness in world units,
projection type, render mode, and an "Edit in canvas" button wired to
`_manipulators.attach_plane_manipulator` (disabled with an install hint when
napari-threedee is absent).

Also added alongside these: `_qt/voxel_size_widget.py` and
`Animation.set_voxel_size`, for correcting the physical voxel spacing of
imported files. This is a prerequisite rather than a nicety — slab thickness is
specified in world units and clipping planes are positioned in world
coordinates, so neither is physically meaningful until the voxel size is right.
A corrected spacing is written into every already-captured keyframe, because
`scale` is part of the captured layer state and replaying an old keyframe would
otherwise restore the wrong spacing.

*Outstanding:* pyramid-level selection and a VRAM estimate on the slice panel;
and porting the legacy `OrthoSlicer` onto `OrthoSlice` as a third `"dims"`
render mode, so there is one ortho-slice concept rather than two.

### Phase 3 — 2D/3D (R3)

* `snap_2d()` / `snap_3d()` and toolbar buttons.
* `add_scroll_through()` helper.
* Auto-fallback of `"plane"` slicers to `"dims"` mode when `ndisplay == 2`.

*Tests:* a 3D→2D→3D animation preserves each mode's camera; scroll-through
generates the expected `dims.point` sequence; plane-mode slicers do not leave
orphaned visible layers in 2D.

### Phase 4 — Keyframe × object toggle matrix (R4)

* `set_object_enabled` / `set_layer_visible` + cache invalidation.
* The matrix widget, with a "propagate to all following keyframes" action
  (Imaris has this and it saves a lot of clicking).

*Tests:* toggling rebuilds the frame index; a toggled-off object contributes
no clipping planes and no visible layer; toggles are step functions, never
half-applied.

### Phase 5 — Polish

* `FILE_FORMAT_VERSION` → `"2"` (`io.py:35`) with a v1→v2 upgrade path that
  converts the old single `ortho` dict into one `OrthoSlice` scene object.
  Serialisation itself needs no new code — `_to_builtin` already handles the
  nested float/bool dicts (verified).
* An `examples/imaris_style_flythrough.py` reproducing the reference movie:
  volume + 2 cutaway planes + 1 sweeping ortho slice + orbiting camera.
* README section and screenshots.

---

## 6. Performance notes

The existing prefetch/dask-cache work (`prefetch.py`, `animation.py:337-353`)
optimises *dims-based slicing*, i.e. reads from disk per frame. The features
here mostly sidestep that path:

* Clipping planes and `depiction='plane'` are evaluated **on the GPU** against
  an already-uploaded texture. Sweeping a plane across 300 frames triggers zero
  additional reads — these are the cheapest possible things to animate.
* The cost moves to a **one-off upload** and to **VRAM**, per §3.4.
* 2D scroll-throughs (`ndisplay=2`, `dims.point` sweeping) *do* hit the read
  path, and benefit from the existing prefetcher unchanged.

So the recommendation for whole-brain-scale data is: do fly-throughs in 3D with
plane/clip objects at a downsampled level, and reserve full-resolution 2D
scroll-throughs for short segments.

---

## 7. Open questions

1. **Scene objects vs. napari layers in the layer list.** `"plane"`-mode ortho
   slices *are* real layers, so they show up in napari's own list. Clipping
   planes cannot be layers — nothing in napari renders them. The plan gives
   them a first-class row in our own object list instead. Worth confirming
   that this split is acceptable, or whether clipping planes should also get a
   (non-rendering, gizmo-only) companion layer purely so that everything lives
   in one list.
2. **Whether ortho-slice backing layers should be user-visible/editable** in
   the napari layer list, or hidden and managed exclusively by the widget.
   User-visible is more napari-native and gives free per-keyframe visibility;
   hidden is harder to break.
3. **Physical-units UX.** `plane.thickness` is in data coordinates;
   `position`/`normal` here are world. Anisotropic voxels make "10 µm thick"
   ambiguous for oblique planes (thickness along the normal vs. along an axis).
   Proposal: define thickness strictly along the plane normal in world units
   and convert per-layer.
4. **Minimum napari version.** 0.5 vs 0.8 — 0.8 brings `camera.synced` and
   `fit_to_view`, which make R3 markedly cleaner.
