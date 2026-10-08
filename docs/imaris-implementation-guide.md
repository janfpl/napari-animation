# Implementation guide: reliable Imaris-style section animation

Date: 2026-09-21. This is the implementation specification and current status
for the Imaris-style rendering work. It supersedes conflicting
implementation/status claims in `imaris-parity-plan.md`. Keep the original
plan as historical context.

## 1. Read this before changing code

The required workflow is:

1. Load a volume and establish correct physical voxel spacing.
2. Create XY, XZ, and YZ sections with a physical thickness and max/mean/min
   projection; scroll them through the data.
3. Show multiple sections with the volume and clipping planes in 3D.
4. Select a section for a flat 2D view, then return to the same scene in 3D.
5. Capture, select, edit, replace, reorder, and delete keyframes.
6. Save, close, reopen with the source data, edit again, and export a movie.

Preserve napari as the rendering/viewer platform. Do not build a replacement
renderer, adopt napari-threedee as a mandatory dependency, or rewrite the entire
plugin in one change. Fix the state ownership and lifecycle first.

Implement one milestone at a time in section 5. Run its acceptance tests before
continuing. A passing test of a private helper is insufficient when the bug is an
interaction among selection, widgets, playback, and saving.

### Current implementation status and evidence

The first implementation pass now covers the highest-risk state and rendering
paths: Python 3.13-safe easing/interpolation enums, controller-owned keyframe
timing edits, stable scene topology interpolation, in-place scene restoration
when scrubbing, compositor-safe legacy ortho controls, 2D thick-section
projection, 3D plane sections, anisotropic and multiscale thickness/scale
conversion, native napari plane/visibility/rendering edit capture, source
rebinding, version-2 atomic persistence with version-1 migration, and slider
cursor clamping. The full suite currently passes **304 tests** with the real
napari 0.8.0/PyQt5 environment described below. The remaining architectural
work is called out by the milestone table; the current remaining work is
primarily visual/GPU acceptance, relinking UX, and a future resampling backend
for oblique true-2D sections.

### Evidence and limits of the original review

The pre-fix full suite, run with Python 3.13.14, napari 0.8.0, PyQt5 5.15.11,
pytest 9.1.1, pytest-qt 4.5.0, and `QT_QPA_PLATFORM=windows`, produced **272
passed, 11 failed, 1 skipped**. See `../.test-tmp/pytest-real-napari.log`. After
the implementation passes the same environment produces **304 passed**.

Eight failures have one cause: the easing enum is empty in this environment.
Seven IO tests fail during **saving**, before they reach their load assertions;
these failures do not independently prove seven loading defects. One GUI test
fails because its easing combo is empty. A separate replacement test demonstrates
a stale frame-index/selection bug. The other two failures are obsolete tests.

The 58 existing scene/slice/scroll tests pass. Many use real napari `ViewerModel`
with a **stub screenshot returning zeros**. They establish model behavior, not
rendered image correctness or scientific equivalence of 2D and 3D projections.
Do not describe that result as complete GPU/visual validation.

Bug IDs B01-B11 below match the order in the last review message. Extra problems
found while designing the fix are explicitly called integration requirements,
not retroactively claimed as failing existing tests.

## 2. Architecture decision: one scene, one rendering path

### 2.1 Separate authored state from rendered state

Use these responsibilities. Keep existing classes where possible.

| Component | Responsibility | Must not do |
|---|---|---|
| `Animation` | Authoring controller; commands, selection, working state, transactions | Let widgets mutate keyframes and caches independently |
| `Scene` / scene objects | Editable canonical parameters, stable object IDs | Infer identity from a layer's display name |
| `ViewerState` | Detached snapshot of camera, view, source appearance, and scene | Hold volume arrays, callbacks, viewer objects, or mutable references to live state |
| `FrameSequence` | Pure evaluation of two snapshots and timing | Change live objects, widgets, or selected keyframe during evaluation |
| `SceneRenderer` (new) | Reconcile managed layers, apply transforms/appearance, composite clips, apply 2D section view | Recapture its own temporary display overrides as authored edits |
| `LayerRegistry` (new) | Source identity and binding; managed-layer ownership by scene-object ID | Reuse or delete an unrelated layer because a name matches |
| Qt widgets | Display state and call controller commands | Assign clips, invalidate caches, or perform independent load migrations |
| `io.py` | Versioned encode/decode, validation, migration, atomic saving | Clear the live timeline before validating a file |

Authoring flow:

```text
widget / script command -> Animation working state -> SceneRenderer -> napari
native napari edit -> guarded event adapter -> Animation working state
capture / overwrite -> detached ViewerState snapshot -> timeline
timeline + time -> FrameSequence -> ViewerState -> same SceneRenderer -> napari
select keyframe -> replace working state + render it + refresh controls
```

The working state is editable and separate from every captured keyframe.
Changing a control does not change a saved keyframe until capture/overwrite.
Selecting a keyframe intentionally replaces the working state. Export temporarily
renders evaluated snapshots and restores the working state/view in `finally`.

### 2.2 State and identity contract

The current implementation introduces these concepts incrementally; keep the
compatibility bridges while older projects migrate:

- `object_id`: existing stable scene-object ID, immutable for its lifetime.
- `source_id`: stable source-layer ID. Source names remain display labels. The
  registry stores the live binding; version 2 files store the source record.
- A managed backing layer is owned by exactly one `object_id`, using registry
  references and namespaced metadata such as `napari_animation.object_id`.
- Version 2 `ViewerState.layers` is keyed by source IDs. A compatibility lookup
  by captured display name remains available to old scripts. Managed slice
  appearance is still captured for backwards-compatible playback, while the
  scene object remains authoritative for geometry.
- Keep the existing scene snapshot shape `{object_id: parameters}`. Add source
  ID, selected pyramid level, and authored appearance as needed. Snapshot
  `enabled` controls the object. For a section, displayed visibility is
  `enabled AND allowed_by_current_view`, never a second authored checkbox.
- `ViewerState.view` carries the optional `active_section_id`. Keep
  `dims.ndisplay` as the sole numeric display-mode field. Do not add a competing
  `view_mode` value that can disagree with it.
- Record explicit source/scene drawing order by ID (for example, lists in
  `view`). Treat order changes as discrete. Normalizing dictionary keys for
  interpolation must not silently reorder the visible layer stack.
- Keep `ortho` as a version-1 compatibility input until migration is complete;
  new authoring uses the scene. Never apply both representations of the same
  slab to one frame.
- Camera state is authored. Layer extents, dims ranges, and other derived napari
  caches are not authoritative project data. Capture an explicit supported dims
  subset (order, ndisplay, point, and compatibility margins), not every cache in
  `dims.dict()` as a command to restore later.

Source records should identify path/URI, reader/plugin if needed, layer type,
channel/returned-layer selector, name, transform, units, and whether relinking
is required. A path is not sufficient to distinguish multiple channels returned
by one reader. Do not reload one multi-layer file once per channel record.

### 2.3 Invariants that every fix must preserve

1. A saved snapshot never changes when a live object or another keyframe changes.
2. Applying a frame is idempotent and independent of the previously displayed
   frame, including first/last frames and backward scrubbing.
3. Applying an empty scene clears managed scene effects and hides managed layers.
   It does not remove unrelated user layers or their unmanaged state.
4. A single compositor writes managed clipping-plane lists. Two clip slabs on
   one source intersect; they do not create a union. Explain that in the UI.
5. Loading/selecting a frame reconstructs the controls as well as the pixels.
6. Continuous values use eased progress. Discrete identity, modes, topology,
   visibility, and axis order use **raw timeline progress**.
7. Render-triggered layer/dims events never become user edits.
8. Data remains lazy where possible. Snapshots and serialization contain no
   image arrays. Reuse backing layers during a sweep; do not create one per frame.
9. Any source/level/transform change updates its managed layers and their plane
   conversion before rendering. World-space geometry remains the authored truth.
10. Missing source data produces a retained, relinkable object plus an actionable
    message. Export fails explicitly if a required visible object cannot render.
11. Required state failures are visible errors. Do not catch every exception and
    silently produce a scientifically different frame.

### 2.4 Controller entry points to implement

Add controller methods with these responsibilities (names may follow existing
style, but keep one implementation per responsibility):

```python
class ControllerInterface:
    # Signatures only; implement these responsibilities on Animation.
    def select_keyframe(self, index): ...
    def preview_frame(self, frame_index): ...
    def capture_keyframe(self, **capture_options): ...
    def overwrite_keyframe(self, index): ...
    def set_keyframe_timing(self, index, *, steps=None, ease=None): ...
    def update_scene_object(self, object_id, **changes): ...
    def set_active_section(self, object_id): ...
    def apply_working_state(self): ...
    def replace_project(self, decoded_project): ...
```

`set_movie_frame_index` and `set_to_keyframe` should delegate to these paths.
Interactive scrub previews must update the working state/controls to what is
visible so capturing a scrubbed frame captures that view. Export must use a
separate non-authoring render call, not repeatedly invoke user selection.

A transaction suppresses recursive **controller callbacks**, not napari's own
rendering events. Use a nesting counter with `try/finally`. On exit: rebuild
dirty frame index/cache once; clamp the cursor; apply the selected/preview state
if requested; emit one authoring-state notification; refresh controls with Qt
signals blocked. Do not suppress every event in the viewer.

## 3. Rendering and section semantics

### 3.1 Specify supported views rather than guessing

For the initial completion target, support axis-aligned XY/XZ/YZ sections in
true 2D, and multiple axis-aligned or oblique sections in 3D. In one true-2D
napari canvas, one `active_section_id` owns the displayed spatial plane and
section margins. Multiple independent orthogonal 2D views require separate
canvases; do not simulate them by repeatedly changing global dims in a loop.

In 2D, show the active section through a managed Image backing layer with the
proper projection mode. Hide other managed sections and suppress source volume
layers as a **view override**, without changing their authored visibility. Keep
non-image annotations according to their own authored visibility. Returning to
3D restores authored visibility and shows all enabled sections.

Native dims slicing is only valid when the selected world section maps to a
single source-data axis. Check both the requested view and the transformed
normal. For the initial true-2D implementation, additionally require that the
complete spatial linear transform is a signed axis permutation with independent
scales (one nonzero entry per row and column, within a documented numeric
tolerance), plus translation. This supports anisotropy, flips, and axis swaps
without assuming that shear preserves projection sampling along the normal.
General rotation/shear remains supported by the 3D geometry path. For oblique
planes or arbitrary transforms that fail the true-2D checks, reject
the true-2D command before mutation with a useful message and retain the current
3D view. A later resampling backend can remove that restriction. Do not silently
show a different axis-aligned slice.

Use the selected section's source and pyramid level in both modes; silently
changing resolution on the mode switch defeats geometry/pixel comparisons.

### 3.2 Thickness and projection contract

Canonical thickness is finite, positive, and measured along a unit normal in
world coordinates. Keep the existing correct inverse-transpose normal handling.
For `world = A @ data + b`:

```text
normal_data = normalize(A.T @ normal_world)
thickness_data = thickness_world / norm(A.T @ normal_world)
thickness_world = thickness_data * norm(A.T @ normal_world)
```

The inverse formula is essential when adopting native napari thickness edits.
Do not use one arbitrary voxel spacing for an oblique normal. For leading time
axes, use the spatial transform at the selected time and reject unsupported
mixing of spatial and time coordinates. Napari plane properties are spatial 3D
tuples: never send an arbitrary-length N-D normal to them.

Distinguish continuous physical slab thickness from a legacy **count of sampled
planes**. A half-width `(N-1)*step/2` used for sample centers is not a universal
physical-thickness formula. Build a coordinate-ramp fixture and establish the
exact voxel inclusion/rounding behavior of the supported napari slicing API.
Test odd/even thickness, subvoxel positions, and volume boundaries explicitly.
Document that convention and use it in migration. Never hide a one-voxel error
by loosening the assertion.

For true-2D projection, compare actual projected scalar data with NumPy
max/mean/min over the expected index interval. At boundaries use only valid
samples; no zero padding for mean/min. A single-voxel/subvoxel section must still
show the agreed nearest/interpolated slice rather than an empty reduction.

Use this explicit default for new physical sections: voxel centers in the
half-open depth interval `[center - thickness/2, center + thickness/2)`;
for an axis-aligned source with positive depth step, data bounds are
`start = ceil(center_data - thickness_data/2)` and
`stop = ceil(center_data + thickness_data/2)`, clamped to `[0, axis_length]`.
If the interval overlaps the source but contains no voxel center, show the
nearest valid voxel (ties choose the lower index; clamp to the valid index range).
If the physical interval does not
overlap the source's voxel support `[-0.5, axis_length - 0.5)` in data coordinates
at all, show an empty section. For negative scale, convert
the two world bounds to data coordinates and sort before taking bounds. Centralize
this calculation and use an independent small hand-computed table in its tests.
Do not impose this new convention on old files without migration/version metadata.

Napari 0.8's installed `_ScalarFieldSliceRequest._data_slice_to_slices` uses
rounding and a conditional inclusive-high adjustment, so simply assigning
`thickness/2` to both margins does **not** implement this convention. The renderer
adapter must translate the chosen sample bounds to effective sampling point and
margins without changing canonical world position. Store those as derived view
state under the event guard. If the supported API cannot reproduce the bounds,
use a direct axis-aligned array slice/reduction backing image for true 2D; do not
modify napari internals or silently change the promised voxel set. Keep that
fallback bounded to the active slab and selected time, preserving lazy reads.

For a nonempty desired integer interval `[start, stop)` in the supported aligned
case, an explicit native-API mapping is:

```python
effective_center_data = (start + stop - 1) / 2.0
effective_margin_data = (stop - start - 1) / 2.0
```

Convert the effective center to a world point and margins by the positive world
step for the selected depth axis; set both margins to that world half-width.
The two data endpoints are then exactly `start` and `stop - 1`, matching napari
0.8's inclusive-high adjustment. Test this against the actual API. Preserve the
original physical center/thickness on the section object: these effective dims
values are sampling settings, not an instruction to move or shrink the section.
Snapshots in section-2D mode must retain the canonical object and recompute its
effective dims on apply; capture must not overwrite its geometry with effective
dims. For an empty interval, hide its output and do not fabricate margins.

For 3D plane rendering, verify the scalar slab behavior at representative camera
angles. Do not assume selecting `average` or `mip` automatically projects along
the slab normal identically to a CPU reduction: the underlying volume renderer
may sample along viewing rays. If the scientific section appearance changes
incorrectly with camera angle, add a projected-section backend that computes a
2D reduction/resampled slab and displays that result on a positioned plane.
Put that behind the same scene/renderer interface; do not change snapshot
semantics. Pixel validation is a release gate for claiming projection parity.

### 3.3 Discrete switch versus smooth transition

First deliver a correct 2D/3D **cut**: source/section geometry and physical center
are preserved, camera is applied last, and the mode changes at an explicit frame.
Hold camera state across a cross-mode segment and switch to the destination
camera at the cut; do not interpolate incomparable 2D/3D camera states blindly.

Then implement smooth reveal as an explicit generated sequence:

1. End the 2D sweep at a known section state.
2. At a one-frame mode boundary enter a 3D orthographic, plane-aligned view with
   matching section center, screen scale, and visible contents.
3. Stay in 3D while orbiting the camera and optionally revealing the volume.

Provide a reverse sequence too. Use an asymmetric labeled synthetic volume to
verify orientation for XY/XZ/YZ and compare images around the boundary. Do not
promise a seamless transition until alignment/crop/projection tests pass. A
toggle of `ndisplay` alone is not an animated transition.

## 4. File-level work map

| File | Intended changes |
|---|---|
| `easing.py`, `interpolation.py` | Stable callable enums; explicit continuous/discrete policies |
| `key_frame.py` | Stable keyframe identity; safe update notification contract |
| `animation.py` | Transactional commands, working-state restoration, source-aware sweeps |
| `frame_sequence.py` | Separate raw/eased progress; normalized state interpolation; cache invalidation |
| `viewer_state.py` | Detached canonical snapshot; renderer delegation; explicit supported fields |
| `scene/scene.py` | Reconcile objects by ID; empty-state cleanup; compositor delegation |
| `scene/ortho_slice.py` | Canonical section geometry/appearance; source binding via registry |
| `scene/base.py` | Validation, IDs, normal/position conversions; identity equality contract |
| `scene/renderer.py` (new) | One application transaction for scene, generated layers, clips, and 2D |
| `layer_registry.py` (new) | Sources and managed-layer references, transform/data revision tracking |
| `ortho_slicer.py` | Legacy compatibility adapter; no competing authoring model |
| `_qt/*` | Commands into Animation, guarded UI refresh, one section selection model |
| `io.py` | Version 2 persistence, version 1 migration, validation, atomic save/load |
| `prefetch.py` | Update any source-ID/name assumptions; skip generated layers; preserve lazy reads |
| `docs/*`, `examples/*` | Honest capability/status documentation and end-to-end example |

Do not introduce a large general-purpose framework. The renderer and registry
can be small classes attached to one `Animation`. Where a public
`ViewerState.apply(viewer)` call has no Animation, obtain a viewer-associated
render context through an explicit helper. The context must remember managed
layers/cleanup and be disposed with the viewer; it must not create a second
independent ownership system.

## 5. Required implementation order

| Milestone | Deliverable | Gate before proceeding | Status |
|---|---|---|---|
| M0 | B01 enum fix and the two stale-test repairs | IO and GUI tests reach their intended assertions; enum tests cannot pass vacuously | **Done** |
| M1 | B02/B10/B11 transactional timeline and Qt fixes | Replace/overwrite/steps/ease/scrub/reorder/delete tests pass | **Done** |
| M2 | B03/B08/B09 controller, registry, renderer ownership | Working state equals selected frame; clips persist during edits; native edits captured once | **Done** |
| M3 | B04 lifecycle and interpolation normalization | Empty/add/remove/rename/backward-scrub tests; deterministic out-of-order evaluation | **Done** |
| M4 | B06/B07 source/level/transform correctness | Multiscale and source-switch tests with anisotropy, affine, relinking | **Done for local bindings**: stable source IDs, multiscale, anisotropy, and rebinding are covered; relinking UX remains |
| M5 | B05 unified section views and legacy adapter | Numeric 2D projection oracle and 2D/3D round-trip tests | **Done for native 2D**: active sections, exact napari-0.8 voxel-interval mapping, real `_data_view` max/mean/min oracle checks, and pre-mutation oblique rejection are implemented; resampling remains |
| M6 | Complete B03 version 2 persistence and v1 migration | Fresh-viewer save/load/edit/export; invalid-file load leaves project intact | **Done for current schema**: v2, migration, atomic writes, and pre-validation are implemented |
| M7 | Smooth reveal, visual/GPU/export validation, docs | Whole workflow acceptance scene, nonblank moving export, no managed-layer leaks | **Pending** |

M2 may restore current v1 snapshots in memory, but do not finalize the v2 writer
until M3-M5 settle the schema. The temporary v1 path must still work after M0.

## 6. Bug-by-bug implementation instructions

### B01 — Empty callable enums break save/load and GUI capture [P1]

**Observed:** Both `list(Easing)` and `list(Interpolation)` are empty in the tested
Python 3.13.14 environment. Their attributes are raw partial objects. IO fails
at `kf.ease.name`; the GUI fails on `Easing['']`.

**Files:** `easing.py`, `interpolation.py`, tests of both enums and IO/Qt.

Use a tuple wrapper for callable enum values; this supports the declared Python
3.9 floor without depending on the newer `enum.member`. This is a concrete
pattern, not an instruction to leave only the two members shown:

```python
from enum import Enum
from functools import partial

class Easing(Enum):
    LINEAR = (partial(linear_interpolation),)
    QUADRATIC = (partial(quadratic_ease_in_out),)
    # Convert EVERY existing member in this class in the same way.

    def __call__(self, *args):
        return self.value[0](*args)
```

Apply the same wrapper and call implementation to **all six** `Interpolation`
members. Keep all ten existing easing names unchanged because JSON uses those
names. Audit `.value` consumers: this representation changes `.value`, not the
public enum call or `.name`. Serialize these enums by name explicitly; never
serialize their tuple/function payload via `_to_builtin`.

**Required tests:** explicit `len(Easing) == 10`, `len(Interpolation) == 6`, all
existing names present, `Easing['LINEAR'] is Easing.LINEAR`, callable numeric
outputs, every easing name round-trips through IO, and the GUI combo has ten
entries and can capture. Do not rely solely on a parametrization over an empty
enum: it silently skips and conceals this bug. Test a supported Python 3.9
environment with compatible napari dependencies and Python 3.13 with napari
0.8, or accurately narrow advertised support; do not force napari 0.8 onto a
Python version its own package metadata rejects.

### B02 — Keyframe replacement selects stale objects [P1]

**Observed:** `capture_keyframe(insert=False)` emits list `changed`; Animation's
callback runs before FrameSequence's rebuild, navigates the old index, and tries
to select the removed keyframe.

**Files:** `animation.py`, `key_frame.py`, `frame_sequence.py`, frame/list widgets.

1. Construct `FrameSequence` and connect its structural-list listeners before
   connecting Animation's selection/change listeners. This is an immediate
   ordering fix, not the entire architecture.
2. Introduce one controller method for updating an existing keyframe **in
   place**. Preserve object identity and its name. `overwrite_keyframe(index)`
   preserves steps/ease; `capture_keyframe(insert=False, steps=..., ease=...)`
   applies explicitly supplied timing. Route both GUI actions through it.
3. Within a controller transaction: construct/validate the new snapshot first;
   assign snapshot/thumbnail/timing to the target; mark cache dirty; rebuild;
   then update selection/view and notify the list model. Do not synthesize an
   early `changed` event that recursively navigates before rebuilding.
4. Avoid duplicate rebuilds by using a guard on FrameSequence's event adapter
   while emitting a post-commit notification. Keep one clear owner of rebuild.
5. Guard selection navigation during controller preview so it cannot recursively
   call itself. A direct external list replacement must also rebuild before
   navigation; cover that path, not only controller helpers.

**Tests:** replace first/middle/last frame repeatedly; overwrite a non-selected
frame; preserve frame identity/name/timing as specified; capture insertion after
the selected frame still works; direct list replacement does not select an old
object; reorder/delete while previewing clamps a valid cursor; no duplicate
selection/renderer events from one command.

**Interaction:** B10 fixes the slider event blocker; B11 must use this same
mutation/invalidation mechanism rather than creating another cache policy.

### B03 — Selection/loading leaves authoring models stale [P1]

**Evidence:** Code inspection proves selection applies snapshots to temporary
render objects while later capture reads the old live Scene. The current IO
suite is blocked earlier by B01; add an explicit load-edit-recapture test.

**Files:** `animation.py`, `scene/scene.py`, renderer/registry, `viewer_state.py`,
`io.py`, all relevant Qt controls.

1. Implement `Scene.restore_snapshot(scene_dict)` that reconciles by ID. Update
   matching live objects, create missing ones, remove objects absent from the
   working snapshot, and reset adapter bookkeeping. Validate kinds before
   touching live objects. Use detached copies of nested appearance parameters.
2. `select_keyframe` loads scene, compatibility ortho state, source appearance,
   dims/view, and camera into the working state; renders once under the event
   guard; then emits `working_state_changed`. All scene/section/scroll/voxel
   controls refresh from this notification, retaining selected object by ID if
   present. Refresh must not issue edit commands.
3. `preview_frame` follows the same working-state rule for an evaluated snapshot.
   The export loop invokes renderer directly and restores authoring/view state
   after success, cancellation, or error. Export must not change keyframes.
4. Capture flushes pending external edits, validates working state, and deep
   copies small state dictionaries. Exclude volume arrays and registry/cache
   objects from those copies. `frozen=True` does not freeze nested dictionaries.
5. The project loader parses, migrates, and validates **all** frames and registry
   references before clearing anything. Stage data reloads and track newly
   opened layers; on failure roll back only staged additions and retain the old
   timeline/working state. Existing user layers are never removed by rollback.
6. Install the validated project in one transaction, reset counters to avoid
   confusing new frame names, select frame zero once, and refresh controls. For
   an empty loaded timeline, clear managed effects and reset authoring controls.

**Version 2 persistence:** Store source records with stable IDs, per-frame source
appearance, scene objects, view/active section, camera, dims subset, steps,
easing names, and any authored interpolation overrides/transition settings.
Encode transforms as numeric matrices and colormaps as reconstructable data
when not a built-in name. Replace the catch-all `str(obj)` with explicit codecs
or a descriptive unsupported-value error. Never write callable representations.

**Migration requirements:**

- Continue reading version 1. Build one source-name-to-ID map for the entire
  project, not a new random ID for each keyframe. Source names may be ambiguous;
  report/relink rather than silently bind the wrong data.
- Retain scene object IDs. Move scene-backed layer properties into scene
  appearance. Historical scene-owned `enabled` wins over duplicate backing-layer
  visibility because 2D used to hide backing layers automatically.
- Preserve legacy global `ortho` effects with a dedicated compatibility adapter
  while converting them. One legacy slab affects multiple source layers and
  may use `sum`; blindly making one single-source max/mean/min OrthoSlice loses
  behavior. For unsupported conversion, keep an explicit legacy effect record
  processed by the renderer until a faithful migration exists.
- Resolve conflicts deterministically: the legacy scene dict owned geometry
  when present; genuine unmanaged plane layers keep their own layer state.
- Existing source data that matches unambiguously may be reused; unresolved
  sources keep their records and object parameters. Missing data is not grounds
  to delete a keyframe or declare a successful export.
- Save with a temporary sibling file, flush/close, then `os.replace`; reject NaN
  and infinity (`allow_nan=False`) and preserve the old file on serialization
  errors. Do not overwrite input files during migration merely to open them.

**Tests:** save two differently configured scene frames; load into a fresh
viewer; check model, widgets, and image state; edit only thickness at frame A;
overwrite; verify every other property remains A's, frame B is unchanged, then
save/reload again. Also test same-session backward selection, load into a viewer
with a different existing scene, malformed second frame, missing source, renamed
source, renamed section, multi-channel reload, old `ortho`, empty project, and
export exception restoring the original authoring view.

### B04 — Object addition/removal and interpolation topology [P1]

**Confirmed:** Removing an object between endpoints supplies `None` to normal
interpolation and raises a TypeError. Walking only source keys omits new objects.

**Files:** `frame_sequence.py`, `utils.py` if necessary, scene lifecycle/renderer.

Normalize scene endpoints before generic numeric interpolation. This helper is
safe as a starting implementation for scene dictionaries only:

```python
from copy import deepcopy

def normalized_scene_pair(left, right):
    left, right = left or {}, right or {}
    a, b = {}, {}
    for object_id in sorted(set(left) | set(right)):
        if object_id in left:
            a[object_id] = deepcopy(left[object_id])
        else:
            a[object_id] = deepcopy(right[object_id])
            a[object_id]['enabled'] = False
        if object_id in right:
            b[object_id] = deepcopy(right[object_id])
        else:
            b[object_id] = deepcopy(left[object_id])
            b[object_id]['enabled'] = False
        if a[object_id]['kind'] != b[object_id]['kind']:
            raise ValueError('An object ID cannot change kind')
    return a, b
```

Fill optional schema fields from kind-specific defaults **before** walking the
normalized dictionaries. Identity/source/topology changes cannot be interpolated
numerically. Do not apply this `enabled` padding rule indiscriminately to camera,
colormap, or arbitrary layer dictionaries.

For layer additions/removals, use the union of source IDs and explicit
`visible=False` on the missing side, retaining other source properties. Never
invent image data to interpolate a missing source. Validate shape, RGB status,
dimensionality, and transform compatibility; discrete-switch incompatible state.

Store raw and eased progress separately in the frame index:

```text
raw = step / destination.steps
eased = destination.ease(raw)
continuous field -> numeric interpolation using eased
discrete field   -> selected endpoint using raw
```

Register complete discrete policies: `dims.ndisplay`, `dims.order`, scene
enabled/name/id/kind/targets/source_id/source_level/projection/render_mode,
view.active_section_id, layer visibility, projection mode, depiction, rendering,
colormap selection, and nonnumeric units. Treat numeric categorical fields as
discrete too. `dims.order` must always be a permutation; linearly blending two
orders creates duplicates. Dotted paths containing source display names should
disappear after stable IDs; match field paths structurally where practical.

Back/elastic easing can overshoot 0..1. Do not let an overshoot turn visibility
on and then off mid-segment. Clamp or reject out-of-domain continuous values
(positive thickness/zoom, opacity range) according to a documented field policy.
Keep exact endpoint snapshots; the final frame is exactly the final keyframe.
The frame count remains `1 + sum(k.steps for k in key_frames[1:])`.

**Lifecycle decision:** Removing an object from the working scene removes it
from future captures, not historical keyframes. Replaying an old frame may
recreate/show its managed layer. To delete from the entire animation, provide a
separate command that edits all snapshots and invalidates once. Fix the current
docstring claiming historical objects are ignored; they are reconstructed by
playback. Renderer must hide stale managed layers even for raw endpoint frames
that bypass interpolation. Rename uses ID and changes a label, never ownership.

**Tests:** empty-to-object, object-to-empty, two-to-one, unrelated IDs, disabled
endpoints, `backfill=False`, renamed objects, kind mismatch, opposite normals,
zero-normal validation, 0/interior/1 fractions, overshooting easing with discrete
modes, XY-to-XZ order changes, and random/out-of-order frame application. Verify
no orphan visible slab and no extra layers after repeated scrubbing.

### B05 — No shared thick-section behavior across 2D and 3D [P1]

**Confirmed:** The present 2D path only hides the slab; it leaves zero projection
margins. The architecture's automatic fallback is missing.

**Files:** renderer, `OrthoSlice`, `ortho_slicer.py`, `animation.py`, section and
scroll widgets, view state schema.

Implement section 3, in this exact order:

1. Add explicit active section selection and a shared world-space section model.
   A 2D snap command uses that section; if no section exists, retain the existing
   plain thin-slice snap behavior. Never choose an arbitrary scene object just
   because it appears first in a list.
2. Resolve and validate source, level, normal/axis mapping, thickness, and view
   order **before** changing dims. A selected plane orthogonal to a different
   named view needs an explicit edit to its normal, not a hidden reinterpretation.
3. In one render transaction, set the requested `ndisplay/order`, set the point
   from the section's effective sample centre, install calibrated section
   margins, and set the source Image's `projection_mode` (`mean`, not the 3D
   `average` name). Hide the plane-depiction backing layer in 2D so the source
   is drawn exactly once. Set thickness only on the intended sliced spatial
   axis; time margins must stay zero. Do not add margins to all axes.
4. Hide other managed sections and apply source-visibility view overrides.
   Rendering overrides never update scene `enabled` or source `visible`.
5. When leaving section-2D mode, restore the ordinary captured dims margins and
   layer projection settings, not blindly zeros everywhere. Apply 3D plane/clip
   representation using the same canonical section geometry.
6. Route native dims scrolling to the active section position only when it is a
   user edit and affects its depth axis. Render-triggered dims changes are
   guarded. Route the section slider/sweep to that same command in both modes.
   Leave `add_scroll_through` as plain dims scrolling when no section is active;
   a 3D dims point alone does not move a scene plane.
7. Apply camera after source transforms, managed-layer creation, dims, and scene
   effects. Apply `camera.synced` first when supported; do not call fit-to-view
   during playback. Explicit user snap/fit commands may fit before capturing.
8. Provide the cut transition first, then the M7 smooth sequence described in
   section 3.3. Do not silently upgrade every legacy 2D/3D segment to an orbit.

**Tests:** every XY/XZ/YZ view, each projection, anisotropic scale, translated
sources, supported negative scale with orientation handled explicitly, thickness
rounding/boundaries, dims scroll followed by capture, 2D -> 3D -> 2D, two enabled
sections with only the selected one in 2D, and unsupported oblique true-2D leaving
the pre-command view unchanged. Check image values, not only `ndisplay` and
`plane.thickness` fields.

**Interaction:** renderer guards (B09), source transforms (B06/B07), and raw
discrete mode interpolation (B04) must exist before this can be correct.

### B06 — Multiscale backing layer shrinks in world coordinates [P2]

**Confirmed:** Coarse data is assigned unchanged source scale. This reduces its
physical extent and corrupts world-to-data plane positioning.

**Files:** source registry/renderer, `OrthoSlice.ensure_layer` compatibility path,
voxel-size propagation, source-record codec.

For napari's scale-based pyramid convention, use the selected level's factors:

```python
factors = np.asarray(source.downsample_factors[level], dtype=float)
data = source.data[level]
level_scale = np.asarray(source.scale, dtype=float) * factors
# Backing transform components:
# scale=level_scale, translate=source.translate, rotate=source.rotate,
# shear=source.shear, affine=source.affine.affine_matrix.copy()
```

Verify API availability in the compatibility adapter. The installed napari 0.8
`downsample_factors` is `level_shapes[0] / level_shapes`; use that convention, not
an invented `(shape-1)` ratio. Preserve the additional affine matrix as well as
scale/rotate/shear/translate. Alternatively use the complete homogeneous
transform `T_level = T_source @ D_level` on an otherwise identity backing layer.
Choose **one** representation; applying both double-scales the result.

If a source format provides per-level translation or arbitrary transforms,
preserve its actual level-to-world mapping via an adapter. Do not infer those
offsets from array shape. For the initial supported scope, explicitly reject
unknown non-scale pyramid mappings rather than claim physically correct output.

Keep a selected level per object. A coarsest-level default may limit VRAM, but
the UI must show the choice and its spatial resolution. Estimate texture size
without converting lazy arrays to NumPy. Changing level changes the data and
transform together under the render guard; then recompute plane position,
normal, and thickness from canonical world values.

`set_voxel_size` must invalidate all dependent managed-layer bindings and update
source transform state in every keyframe according to its existing global-edit
contract. Do not leave backing layers at their old scale. Preserve authored
world positions/thickness and document that voxel-size correction does not
silently recenter the scene.

**Tests:** two-level anisotropic pyramid, nonsquare dimensions, noninteger level
factor, translate + rotate + shear + affine, level switch at a keyframe,
voxel-size correction after layer creation, and saved/reloaded geometry.
Assert corresponding data coordinates map to equal world points and plane
thickness has the same physical meaning. Coarse and fine sampled center extents
can differ at edges; test the mapping, not a false equality of array endpoints.

### B07 — Source changes do not update the backing layer [P2]

**Confirmed:** Editing `source` returns the already-existing layer, still bound
to the old array and transforms.

**Files:** registry/renderer, section editor, source/level snapshot schema.

1. Replace name-based `ensure_layer` lookup with object-ID ownership lookup.
   Never adopt an arbitrary user Image with the same name.
2. Track a binding fingerprint `(source_id, level, data_revision,
   transform_revision, ndim, rgb)` in the renderer, outside captured state.
   Source data/transform events increment revisions. Do not hash or compare
   full arrays to detect edits.
3. Resolve/validate the target before rebinding. If compatible, update data and
   the complete level transform under the guard. If dimensionality/RGB change
   requires recreation, disconnect handlers, replace only the owned layer, and
   reconnect them exactly once. Preserve canonical appearance and world geometry.
4. New sections inherit initial appearance from their source once. A source
   change preserves the section's explicit appearance unless the user invokes
   a separate reset-to-source command. This avoids hidden per-frame color changes.
5. Missing/deleted sources hide the managed output and retain relinkable state;
   they must not keep showing stale data. Use the strict export rule in 2.3.
6. A source switch is discrete. Hold complete section/source binding state until
   the boundary; do not interpolate geometry across incompatible source frames.

**Tests:** A contains zeros, B ones and different scale/translation; switch and
assert array identity, world transform, plane geometry, and unchanged appearance;
switch back; animate A->B; rename source and section; collide with a user layer
name; delete/relink source; repeated switches do not leak layers or callbacks.

### B08 — Legacy controls erase composited clipping planes [P2]

**Confirmed:** Applying the legacy projection slicer changed the real source's
clip-plane count from one to zero.

**Files:** `ortho_slicer.py`, its widget, `scene/scene.py`, renderer/controller.

Immediate safe fix before full migration:

```python
# Inside the controller render guard, not directly in the Qt widget:
params = animation.ortho_slicer.to_dict()  # include enabled=False explicitly
OrthoSlicer.apply_state(viewer, params, assign_clipping_planes=False)
animation.scene.apply(viewer, ortho=params)
```

This retains compatibility margins while the compositor combines clips. Replace
**both** widget call sites (manual edit and dims callback). This snippet is a
transitional fix only: after M5 the legacy widget edits the canonical section or
the explicit migrated legacy effect, and delegates to SceneRenderer.

Adopt preexisting unmanaged clipping planes exactly once, **before** the first
scene render takes ownership. The existing adoption helper is not enough unless
called. Record which layers are managed; do not clear every unrelated layer in
the viewer. Subsequent external clip edits need an explicit import/adopt command
or a documented guarded adapter; do not repeatedly adopt compositor output as
new planes. Switching legacy enabled -> disabled must actively remove only its
contribution even when the captured `ortho` field was formerly `None`.

**Tests:** two cuts + slab, projection and clip modes, moving dims, changing
thickness, disabling/removing a section, no-scene endpoint after a scene endpoint,
external planes before adding a section, and backward keyframe selection. Assert
the complete plane set and geometry, not only a nonzero count.

### B09 — Native layer thickness/visibility/appearance edits are lost [P2]

**Confirmed:** Changing native thickness/visibility leaves scene parameters
unchanged; the next scene apply restores old values.

**Files:** renderer event adapter, `OrthoSlice.sync_from_viewer`, section editor,
`ViewerState` capture/application.

Replace `_applied_plane` comparisons with a guarded event bridge. Keep a snapshot
comparison only as a fallback for napari versions that lack a required event.

```python
from contextlib import contextmanager

class RenderGuard:
    def __init__(self):
        self.depth = 0

    @contextmanager
    def applying(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    @property
    def active(self):
        return self.depth != 0
```

The renderer owns one guard shared with controller and adapters. In every layer
edit callback, return immediately while guarded. Otherwise read the changed
property, validate/convert it, update the matching object ID, and notify controls.
If rendering is needed, perform it guarded. Connect to plane position/normal/
thickness, visible, rendering/projection, opacity, contrast, gamma, and colormap
as supported. When replacing a plane model or layer, rebind nested events; do
not leave listeners attached to the old model.

- Map 3D rendering `mip/average/minip` back to `max/mean/min`.
- Convert position/normal and **thickness** using the complete transform and
  formulas in section 3.2.
- Native visibility edits set canonical `enabled` only when the layer is a
  visible/editable representation in the current view. Renderer suppression in
  2D must never disable the object. Prevent misleading editing of suppressed
  representations or route it through a clear object-level command.
- Route `set_layer_visible` on a managed layer to its owning scene object's
  enabled flag. Do not write a second contradictory visibility field.
- Native scale/translate/affine edits on managed layers must not silently break
  source registration. Make these derived transform controls read-only where
  possible, or reject/revert the edit with a clear message and direct the user
  to source voxel/transform controls. Position the slab via its plane parameters.
- Controllers/scripts should use `update_scene_object`. For compatibility with
  direct object mutation, `scene.apply` flushes object parameters under the guard;
  it must not first recapture a stale layer and overwrite those parameters.

**Tests:** native layer edits followed by capture/select/save/load; simultaneous
normal + thickness change; user checkbox off; 2D suppression followed by 3D;
script object edit; source rebind; replacement plane model; count callbacks to
detect duplicate subscriptions; failed render unwinds the guard.

### B10 — Timeline slider remains blocked after rebuild [P2]

**Evidence:** `_nframes_changed` blocks signals when the timeline is nonempty and
does not restore the previous signal state. Some navigation callbacks happen to
unblock it later; that does not make a timing edit safe.

**Files:** `_qt/animation_widget.py`, frame widget refresh logic, controller.

Concrete replacement pattern:

```python
def _nframes_changed(self, event):
    slider = self.animationSlider
    previous = slider.blockSignals(True)
    try:
        count = int(event.value)
        maximum = max(0, count - 1)
        slider.setEnabled(count > 0)
        slider.setRange(0, maximum)
        slider.setValue(min(max(slider.value(), 0), maximum))
    finally:
        slider.blockSignals(previous)
```

Use the same save/restore pattern for steps/ease controls during programmatic
selection refresh. Merely setting signals back to `False` breaks nested blockers.
After a duration edit, the controller clamps/reapplies its current frame as
needed; the widget reflects it without issuing another command.

Also fix the adjacent current-index setter in `frame_sequence.py`: compare the
new integer with `self.__current_index`, **not** the `_frame_index` dictionary.
Do not use the current-index-changed event to signal changed frame contents at
the same index. A separate state/timeline refresh notification handles that case.

**Tests:** change steps, drag slider, verify evaluated viewer/section position;
empty/nonempty timelines; shrinking duration; nested preexisting signal blocker;
same-index content update; repeated rebuild does not disable user interaction.

### B11 — Easing edits leave old interpolation fractions cached [P2]

**Evidence:** The callback assigns `active_keyframe.ease` but never rebuilds
`_frame_index`, whose fractions were computed using the previous easing. The
normal GUI path is currently masked by B01.

**Files:** controller mutation API, `FrameWidget`, FrameSequence invalidation.

1. Implement `set_keyframe_timing(index, steps=None, ease=None)` in the controller.
   Validate steps as an integer >= 1 (reject bool/noninteger), ease as a supported
   Easing member, and index before mutation. Use the M1 transaction.
2. Route both timing callbacks through it. If there is no active keyframe, return
   without dereferencing `None`. Selection refresh must block control signals.
3. Rebuild raw/eased segment fractions and clear frame cache in the same commit;
   do not merely clear `_cache`, which would reuse old fractions from the index.
4. Preserve selected keyframe, working state, and frame names. If the user is
   previewing a timeline frame, reevaluate it once after the timing change.
5. Document that direct dataclass field assignment requires this command or
   replace fields with validated events routed into the same invalidation path.
   Do not advertise fully evented KeyFrame fields while silently permitting stale
   direct assignments.

**Tests:** cache a quarter-segment frame with LINEAR, change to QUADRATIC, and
assert its numeric position changes; the midpoint is a poor fixture because
several symmetric easings all give 0.5 there. Verify endpoints and frame count
unchanged, new save/load persists the easing, new export uses it, and the slider
still works (B10). Test no-active-selection callback and invalid timing rejection.

## 7. Two failed tests to repair without changing correct production behavior

### T01 — Camera `interactive` assertion is obsolete

`test_set_viewer_state` reads `camera.dict()['interactive']`, absent in napari
0.8. Create state A with explicit camera values, mutate to different state B,
apply A, and assert center/angles/zoom/perspective and supported pan/zoom flags
are restored. Feature-detect version-dependent fields. Do not just remove the
assertion and leave a test that compares identical default states. Include a
2D/3D switch so camera-last ordering is exercised.

### T02 — PNG filename test mocks a path no longer used

`test_animate_filenames` mocks `FrameSequence.iter_frames` for 30 frames, but
`Animation.animate` now evaluates `_frames` directly. The actual timeline has
16 frames. Set a real small timeline with explicit steps, mock only the costly
screenshot/encoder boundary, and expect exactly
`1 + sum(k.steps for k in frames[1:])` output files with correct names.
Alternatively construct a 30-frame timeline if that count matters; never alter
the renderer to output 30 just to satisfy the old mock. Keep separate real PNG/
video encode-decode tests; count actual outputs and sample nonblank frames.

## 8. Integration traps the implementer must check

These are not optional if touched by the architecture changes:

- `source=None` currently means first eligible layer and can select another
  generated section. Resolve it once to a genuine source ID when creating the
  object; skip all owned backing layers.
- Renaming must not create a second backing layer or redirect `remove_layer` to
  an unrelated source. IDs decide ownership; names are labels only.
- Recreating generated layers after applying per-layer state loses their captured
  appearance. Apply their scene appearance after ensuring/rebinding them.
- External image layers missing from a keyframe need an explicit visibility
  policy; do not let arbitrary live layers leak into deterministic export.
  Limit project-managed visibility overrides to registered sources/owned layers.
- `snap_2d`/`snap_3d` currently change dims without reconciling scene visibility;
  both must finish through the controller render path, even `capture=False`.
- `add_scroll_through`/`add_slice_sweep` cannot assume newly inserted frames are
  `key_frames[-1]` when insertion is after a selected middle frame. Return the
  actual created keyframes/IDs and select/insert the pair in known order.
- An invalid normal (zero/NaN/infinity), thickness, source type, or transform must
  fail before pushing it to napari. Do not rely on warnings and NaN normalization.
- A dataclass with field-based equality and identity-based hash violates set
  invariants. For selectable live scene objects, use identity equality/hash
  consistently (e.g. `eq=False` on base AND subclasses). Test value equality on
  serialized snapshots, not on live controller objects.
- World geometry depends on units as well as scale. Store units consistently,
  and reject/convert incompatible source units before compositing a shared cut.
- Source clipping targets and section backing layers are distinct. A cut aimed
  at one source should affect its derived sections only under an explicit target
  policy. Preserve the current intended policy in tests rather than accidentally
  expanding `all layers` to overlays or unrelated images.
- A clip-mode slab carves a source volume; its pixels use that source's rendering
  method. The current slab `projection` parameter does not independently set a
  projection in clip mode. Label clip mode as inheriting source rendering and
  disable the misleading per-slab projection selector there. Multiple clip slabs
  cannot assign different rendering methods to one source. Use plane mode for
  independently projected sections; retain their projection parameter for that
  representation rather than overwriting source appearance behind the user's back.
- Asynchronous slicing can produce stale screenshots. Export must wait for the
  selected frame's slice/render completion through a version-tested adapter,
  not arbitrary sleeps. User interaction must not mutate the timeline during
  export; restore controls in `finally` after errors too.
- Custom colormaps, affine transforms, layer order, and units need serialization
  tests. Rendering the same scene after reopening is more than camera + position.

## 9. Validation plan and definition of done

### 9.1 Regression tests by layer

Add meaningful tests, preferably in focused new files rather than inflating one
monolithic test. Suggested grouping:

- `test_enum_contract.py`: B01 counts/names/calls; runs without GUI fixtures.
- `test_timeline_editing.py`: B02/B04/B11, list event ordering, raw/eased progress.
- `test_scene_lifecycle.py`: B03/B04/B08/B09, restore/edit/remove/rename/cleanup.
- `test_section_geometry.py`: B05/B06/B07, physical mapping and projection oracle.
- `test_project_roundtrip.py`: B03 format versions and fresh-viewer edit/export.
- `_qt/_tests/test_authoring_workflow.py`: real buttons, steps/ease changes,
  scrubbing, native layer edits, source changes, and control refresh.
- `test_rendered_sections.py`: marked real Qt/OpenGL image tests; no fake screenshots.

Use deterministic small scalar arrays, e.g. `100*z + 10*y + x`, unequal axis
lengths, anisotropic scale, and translated origins. Set `rgb=False` explicitly
for synthetic volumes whose final dimension is 3 or 4. Include time + ZYX data
separately so spatial normal and active time are not conflated.

Before fixes, each regression should fail for its intended reason. If B01 blocks
a deeper test, implement B01 first, then demonstrate the deeper failure. Do not
mock away the interaction being tested. Keep pytest-qt fixtures for event-driven
widget behavior, and napari ViewerModel for numerical model tests.

### 9.2 End-to-end acceptance scenario

1. Load a reproducible synthetic volume from a temporary file (so reopening is
   meaningful); set anisotropic spacing and a nonzero translation.
2. Add three orthogonal sections, two cut planes, and distinct section colors.
3. Select XY in 2D, set a thick mean projection, scroll to start/end, capture.
4. Enter 3D with matching section geometry, orbit, hide one section at the next
   keyframe, then return to XZ in 2D.
5. Select an earlier frame, edit thickness through native controls, overwrite;
   change duration and easing; scrub; reorder; replace a middle frame; remove an
   object from the working scene and capture another frame.
6. Save, close/dispose the viewer, open a fresh viewer, load sources/project,
   select each keyframe, and compare semantic state and representative images.
7. Edit a loaded clipping plane and section, overwrite/save/reload again. Confirm
   untouched frames do not change.
8. Export PNGs and a short MP4; verify frame count, first/last state, nonblank and
   changing representative frames, expected section position, camera motion,
   and stable visibility. Decode the MP4 rather than checking file existence.
9. Confirm no extra managed layers or accumulated listeners after repeated
   backward/forward scrubbing. Export completion/failure restores the working view.
10. Repeat source/level selection with a small multiscale pyramid and a missing
    source/relink case. These model checks do not require a huge real dataset.

The existing 272 passes remain the baseline. Repair the two obsolete tests with
stronger assertions, not blanket skips. Run with Windows native Qt for GPU tests;
the earlier offscreen backend produced OpenGL context failures. Skips must state
a concrete missing capability; a black stub screenshot is not a visual pass.

### 9.3 Commands for this workspace

From the repository directory (the inner `napari-animation-...` directory):

```powershell
$env:QT_QPA_PLATFORM = 'windows'
$env:NAPARI_CONFIG = ''
..\.venv\Scripts\python.exe -m pytest napari_animation/_tests napari_animation/_qt/_tests -q -p no:pretty --tb=short
```

This virtual environment is deliberately in the parent workspace to avoid
Windows path-length failures. Tests require permission to write napari's normal
user cache and may require desktop graphics access. Do not change the user's
global Python installation to fix this project. Record dependency versions and
test logs for the final implementation report.

### 9.4 Completion report the implementing model must provide

List B01-B11 individually as fixed/remaining, with a regression test for each.
Report full-suite counts, real rendering checks, tested Python/napari versions,
v1/v2 migration coverage, and known unsupported 2D/oblique cases. Explain any
remaining scientific rendering mismatch. Do not mark Imaris-style parity complete
solely because the current unit suite turns green.

Update the old plan's status, user guide, and example only after the corresponding
behavior exists. The implementation is complete when the acceptance workflow
works, saved projects remain editable, and the supported rendering semantics are
verified—not when isolated sliders and plane setters work individually.

## 10. Copyable handoff prompt

Use this prompt with the model that will implement the changes:

> Implement `docs/imaris-implementation-guide.md` in this repository. Read the
> entire guide and applicable repository instructions first. The guide is the
> corrective specification; conflicting completion claims in the original
> parity plan are historical. Preserve unrelated user changes.
>
> Follow milestones M0 through M7 in order. Before each milestone, identify the
> files and acceptance tests it affects. Make bounded changes, run those tests,
> and inspect failures before proceeding. Do not skip deeper regressions because
> the existing 58 scene tests pass. Do not silently broaden supported rendering
> semantics or discard existing saved state to make a test pass.
>
> Keep one canonical editable scene, detached keyframe snapshots, one guarded
> rendering/compositing path, and stable IDs for ownership. Route editing,
> selection, preview, source changes, and loading through the shared controller.
> Prove save/reopen/edit/overwrite behavior with real napari. Use real rendered
> images for projection and transition checks; stub screenshots are only for
> model/GUI event tests.
>
> Treat sample enum code as a pattern that must be applied to every existing
> member. Treat interface signatures as a contract to implement, not empty
> methods to paste into production. Preserve callable enum names used in old
> files. Keep v1 reading until v2 migration is verified.
>
> If you encounter an API mismatch or a failed numerical/rendering oracle,
> investigate the local supported napari implementation and update the
> compatibility adapter/test. Do not suppress the exception or weaken the
> assertion without explaining and documenting a deliberate contract change.
>
> Maintain an implementation checklist against B01-B11, T01-T02, and M0-M7.
> Record completed changes, exact tests/results, schema decisions, and remaining
> work as you go so another session can resume without repeating or overlooking
> work. Finish with the completion report specified in section 9.4.
