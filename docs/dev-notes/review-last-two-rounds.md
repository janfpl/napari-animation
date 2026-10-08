# Review of the last two implementation rounds

Date: 2026-09-21

## Follow-up review

The next review found regressions and incomplete fixes in the resolution pass
below. See [the follow-up findings and fixes](review-followup-2026-09-21.md)
for the current status and real Windows Qt/OpenGL validation. The earlier
claim that all findings were resolved should be read as historical, not as a
current completion assessment.

## Previous resolution update

All 18 findings below were addressed in the implementation pass on the same
date.  The descriptions are retained as the original review record; their
locations and observations describe the pre-fix code.

| Finding | Resolution |
|---|---|
| R01 | Managed layers are resolved by object ID and role, use deterministic IDs, and receive a collision-free display name. User layers are never claimed by name. |
| R02 | Loading always binds existing layers to saved IDs, independent of file reopening. Every loaded frame and v1 name-keyed snapshot is normalized to the project ID map before timeline replacement. |
| R03 | Source records and every scene object, geometry, active section, and target 2D mode are validated before commit. Timeline replacement rolls back if first application fails. |
| R04 | A resolved source ID is authoritative. The Qt editor and controller source-change command update source name and ID together. |
| R05 | The registry repairs duplicate live IDs before capture and rejects duplicate saved IDs. |
| R06 | Active-section selection sets `dims.order` from the section's world normal, and scene-list selection invokes the controller command. XY/XZ/YZ pixel regressions cover all axes. |
| R07 | A guarded dims event adapter updates the active section position while scrolling, with capture-time synchronization as a fallback. |
| R08 | Snap, selection, and snapshot application preflight the proposed 2D section. Public controller commands roll back order, mode, and active ID after a failure. |
| R09 | True 2D uses a managed projected layer built from the explicitly selected pyramid level. |
| R10 | A fully outside interval hides managed output, and the oracle returns deterministic zeros. |
| R11 | Signed axis scales and permutations are projected into world-axis order and displayed through a diagonal output transform. Visible napari pixels are compared with the oracle. |
| R12 | Every renderer application reconciles all managed IDs and hides layers absent from an empty or out-of-order snapshot. |
| R13 | Backing layers use the complete composed source-to-world affine, including pyramid factors and fixed leading-axis coordinates. |
| R14 | Snapshot apply pre-creates deterministic managed layers before restoring captured appearance. First and repeated applications now agree. |
| R15 | Sweep commands adopt native edits before computing endpoints, render each requested endpoint before capture, and advance the plane comparison baseline after adoption. |
| R16 | All identity, topology, order, source, level, projection, render-mode, and visibility fields use explicit discrete policies driven by raw progress. Layer topology is normalized by source ID. |
| R17 | Leading time/channel coordinates are selected explicitly; backing and projection layers are always spatial 3D and the oracle always returns 2D pixels. |
| R18 | Active 2D sections have their own managed output. Source visibility is a reversible renderer override and authored visibility is preserved in snapshots. |

Additional consistency work from the review was also completed: the sampling
convention is stored on each section and validated, and movie export restores
the working viewer, scene, active section, and legacy slicer in its `finally`
path.

Validation after the fixes:

- 130 focused real-napari renderer, scrolling, persistence, export, and state
  tests pass, plus 56 core/diagnostic/prefetch tests (186 total passing checks).
- 16 new regression tests cover the reviewed ownership, transaction, exact
  projection, transform, multiscale, 4D, empty-output, and interpolation cases.
- The original diagnostic script now reports the expected result for identity,
  rollback, active-axis selection, empty-snapshot reconciliation, affine/time
  handling, first-apply appearance, hidden sources, and bidirectional legacy
  interpolation. Its `actual` fields for several old probes still inspect the
  suppressed source layer; authoritative 2D pixels now live on the managed
  `section_2d` layer and are covered by the regression suite.

The host's Qt5 offscreen backend cannot create an OpenGL context for tests that
instantiate a full `napari.Viewer`; those fail during napari setup at
`glGetIntegerv(GL_MAX_TEXTURE_SIZE)`, before project code runs. `ViewerModel`
tests use real napari layer, transform, slicing, and `_data_view` behavior.

## Original review

The original assessment was: **not ready to mark M2–M6 complete**. The new
public classes and source-ID snapshot format were useful foundations, but
ownership, state application, source identity, and section sampling violated
the implementation guide in reproducible workflows.

Scope: the renderer/registry and source-ID/active-section changes, their controller/persistence integration, and the subsequent exact voxel-interval changes. This directory has no Git metadata, so this review compares the current implementation with the preceding change descriptions and the implementation guide, rather than assigning findings to particular commits. Some findings are regressions introduced by the new paths; others are incomplete fixes for bugs the plan already required these rounds to address.

## Validation

- Installed real napari 0.8.0, Python 3.13, PyQt5; offscreen model execution.
- Existing focused suites: **84 passed** across `test_ortho_slice.py`, `test_animatable_state.py`, `test_io.py`, and `test_scene.py`.
- Additional diagnostic probes reproduce the failures below. These probes use real napari layers, transforms, slicing, and `_data_view`. Their screenshot stub is used only for keyframe thumbnails: this is **not** full GPU screenshot or oblique pixel validation.
- The ordinary disable/re-enable probe succeeded; no defect is claimed for that case.
- Production code was not changed during this review.

Reproduction script: [review_increments.py](../.test-tmp/review_increments.py).
Observed output: [review-increments-output.txt](../.test-tmp/review-increments-output.txt).

From the project directory:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
..\.venv\Scripts\python.exe .test-tmp\review_increments.py
```

One probe can be selected by appending its name. The script reports observations and intentionally continues after exceptions; exit code zero does not mean the contract checks passed.

## Findings, ordered by repair priority

### R01 — P1: a section can overwrite an unrelated user's layer

Location: [ortho_slice.py:331](../napari_animation/scene/ortho_slice.py#L331), especially the ownership check and `layer.data = data` at lines 334–347.

An existing layer with matching display name and **no** ownership metadata is accepted as the backing layer. The code replaces its data/transforms and assigns the section's ownership ID. The `layer_collision` probe created a user image containing 999; adding a section with that name replaced its data with the source volume, whose maximum was 503. Removing the section can subsequently remove that formerly user-owned layer.

Resolve managed layers by object ID, never by an unowned matching name. Allocate a noncolliding display name for a new backing layer. Apply the same ownership rule to hiding, removal, rename, and missing-source paths. This is a direct violation of B07 and the registry's stated contract.

### R02 — P1: loaded and recaptured keyframes can use incompatible source IDs

Locations: [io.py:277](../napari_animation/io.py#L277), [io.py:196](../napari_animation/io.py#L196), [viewer_state.py:232](../napari_animation/viewer_state.py#L232).

With `reload_layers=False`, existing layers are never bound to the saved IDs. Name fallback makes the first loaded frame appear to work, but a subsequent capture assigns a different ID. Interpolating the old and new frames raises `TypeError('only 0-dimensional arrays can be converted to Python scalars')` in the `reload_recapture` probe.

The v1 migration also leaves genuine name-keyed snapshots unchanged. The `legacy_recapture` probe showed that old-to-new interpolation works through the name alias, while new-to-old interpolation fails after reordering or returning to an old frame. Existing v1 tests mostly relabel newly generated v2 data and miss this asymmetry.

Separate binding existing layers from opening missing files; binding must happen regardless of `reload_layers`. Normalize every loaded frame and source reference to a single project-wide ID mapping before replacing the timeline. Test fresh load, capture, overwrite, reorder, and bidirectional interpolation.

### R03 — P1: invalid scene data replaces the timeline before failing

Locations: [io.py:214](../napari_animation/io.py#L214), [io.py:280](../napari_animation/io.py#L280).

Pre-validation constructs a `ViewerState` but does not validate its scene objects. An unknown scene kind therefore passes decoding, replaces the existing timeline, and raises only during selection/restoration. The `invalid_load` probe confirmed that the old keyframe object was no longer present after the error.

Validate all object kinds, IDs, geometry, active-section references, supported view state, and source records before committing. Retain the previous working state and timeline if application fails. The current claim that invalid files leave the project intact is only true for the narrow invalid-file test.

### R04 — P1: reusing a source's old name overrides a valid stable binding

Location: [ortho_slice.py:119](../napari_animation/scene/ortho_slice.py#L119).

If volume A is renamed and volume B takes A's old display name, `source_layer` treats the stale saved name as an intentional source edit, even though A still resolves by the saved ID. In `rename_reuse`, the section silently switched to B and changed its source ID.

A resolved source ID must remain authoritative. An explicit source-selection command should update the ID and label together; a read/resolve operation must not infer user intent from a name collision. Legacy name fallback needs an explicit migration/binding boundary.

### R05 — P1: duplicated metadata collapses two layers into one saved entry

Locations: [layer_registry.py:59](../napari_animation/layer_registry.py#L59), [viewer_state.py:194](../napari_animation/viewer_state.py#L194).

`ensure_id` trusts any existing metadata ID without checking uniqueness among live layers. Copying a layer's metadata produces two layers with one ID; the later snapshot entry overwrites the earlier one. The `duplicate_ids` probe had two live layers and only one captured layer.

Track ID-to-live-layer ownership, detect duplicates, and allocate a new ID for a newly duplicated layer without changing the original binding. Validate duplicates on load as well. Persisting a UUID field alone does not establish unique identity.

### R06 — P1: active-section selection does not select the displayed plane

Locations: [animation.py:229](../napari_animation/animation.py#L229), [ortho_slice.py:534](../napari_animation/scene/ortho_slice.py#L534), `SceneWidget._selection_changed`.

The section path changes a point and margins but never sets the corresponding `dims.order`. Selecting XZ while the viewer is in XY leaves Z hidden, places margins on displayed Y, and displays XY pixels. The `axis_selection` probe produced shape `(8, 9)` instead of the expected XZ `(7, 9)`.

There is also no Qt call to `set_active_section`: selecting a scene row only changes the editor, while the scroll panel's section combo chooses a sweep target. The feature is currently a partial script API, not the planned authoring selection.

Make one controller command select the section, validate its plane, set display order, and render. Wire a clear UI selection to that command. Assert pixels and orientation for XY/XZ/YZ, not only stored active IDs.

### R07 — P1: scrolling a 2D section is lost when capturing

Locations: [animation.py:119](../napari_animation/animation.py#L119), [ortho_slice.py:610](../napari_animation/scene/ortho_slice.py#L610).

Capture synchronizes only the hidden 3D plane, not user edits to the active section's dims depth. In `scroll_capture`, the user moved the depth from 2 to 5, but the keyframe recorded section position 2 and applying the newly captured keyframe returned the display to 2.

Add a guarded dims-to-canonical-section adapter and use it for user scrolling and scroll-through generation. Preserve canonical physical position separately from the renderer's effective sample-center offset. Otherwise the application's central scroll/capture workflow is incorrect.

### R08 — P1: oblique 2D rejection is not a pre-mutation guarantee

Locations: [animation.py:311](../napari_animation/animation.py#L311), [animation.py:238](../napari_animation/animation.py#L238), [viewer_state.py:226](../napari_animation/viewer_state.py#L226).

The Qt snap command calls `snap_2d(capture=False)`, which changes order/mode without validating or rendering the scene. `oblique_snap` changed a valid oblique 3D view to 2D without raising. Separately, `set_active_section` commits the invalid active ID before rendering; `invalid_selection` raised but retained that invalid selection. Snapshot apply also changes dims and layer properties before scene validation.

Preflight the complete proposed view before mutations, and use a transaction with rollback for selection/mode changes. Validation only inside `Scene.apply` does not protect the public command paths. The preceding completion report's claim is too broad.

### R09 — P1: 2D samples can come from the wrong pyramid level

Location: [ortho_slice.py:545](../napari_animation/scene/ortho_slice.py#L545).

The interval calculation uses `source_level`, but the visible source image continues to use napari's independently selected level. The selected-level backing image is hidden and its argument is unused. Thus geometry, actual pixels, and the oracle can refer to different data.

In `pyramid`, the section selected coarse level 1 while the source selected level 0: the first displayed pixel was 360 versus oracle 1045. A level change or camera-driven level selection can change the mismatch.

Render the selected level through a managed section output or another explicitly controlled adapter; do not derive coarse-level margins while drawing the uncontrolled source pyramid. Test deliberately different level values, not only shared array identities and scale.

### R10 — P1: sections outside the volume display edge data

Locations: [ortho_slice.py:568](../napari_animation/scene/ortho_slice.py#L568), [ortho_slice.py:234](../napari_animation/scene/ortho_slice.py#L234).

When `_sample_interval` returns `None`, the renderer sets the point outside the source, zeroes margins, and leaves the source visible. Napari clamps the point to its range. The `outside` probe requested z=100 on a seven-plane volume and displayed z=6, with maximum pixel value 503.

The oracle's empty case additionally returns `np.empty` with a full image shape: these are uninitialized values, not an empty section.

Treat no overlap as an explicit empty render result and suppress only managed section output. Define a deterministic oracle representation for empty output, and test fully outside and partially overlapping slabs at both boundaries.

### R11 — P1: the support check accepts transforms the native path renders incorrectly

Locations: [ortho_slice.py:173](../napari_animation/scene/ortho_slice.py#L173), [ortho_slice.py:599](../napari_animation/scene/ortho_slice.py#L599).

Signed scales and axis permutations pass `supports_2d`, but napari's native slicing does not support them through the current point/margin mapping. Under negative depth scale, positive world margins become reversed data margins. Axis swaps additionally require data-axis/display-axis remapping; selecting a different world axis alone is insufficient.

`negative_scale` returned 216 instead of expected 180 for the first mean-projection pixel. `permutation` returned shape `(8, 9)` versus `(7, 9)`, with napari warning that the requested slicing was unsupported.

Implement a verified transform-aware section output or narrow the advertised support with preflight rejection until it exists. The plan requires flips and swaps, so rejection is a temporary restriction, not completion.

### R12 — P1: an empty snapshot leaves previous sections visible

Locations: [scene/renderer.py:41](../napari_animation/scene/renderer.py#L41), [scene/scene.py:124](../napari_animation/scene/scene.py#L124).

Snapshot rendering only visits objects in the requested snapshot; it never reconciles the set of already managed layers against absent IDs. `snapshot_empty` left the previous backing plane visible after `renderer.apply_snapshot({})`.

The authoring scrub path partly masks this by restoring absent live objects as disabled, but export and direct snapshot rendering do not use that reconciliation. Endpoints can therefore render differently from the preview or depend on previously applied frames.

Make the renderer reconcile managed object IDs on every application, including empty snapshots and endpoints. Hide absent outputs without deleting unrelated layers. Test out-of-order snapshot application and export, not just working-scene selection.

### R13 — P2: backing layers omit the source's additional affine

Locations: [ortho_slice.py:342](../napari_animation/scene/ortho_slice.py#L342), [ortho_slice.py:353](../napari_animation/scene/ortho_slice.py#L353).

Creation and rebinding copy scale/translate/rotate/shear but omit `source.affine`. In `affine_copy`, the source origin was `(25, 0, 0)` and the backing origin `(0, 0, 0)`; a section at world z=29 was placed at data z=29 instead of 4.

Copy the complete source-to-world transform, compose the level transform exactly once, and preserve spatial units. Test creation, source switch, level switch, and animated affine changes. M4 cannot claim general transform correctness with this omission.

### R14 — P2: a freshly loaded scene loses section appearance on first application

Location: [viewer_state.py:229](../napari_animation/viewer_state.py#L229).

Layer appearance is applied before missing scene backing layers are created. Their saved display settings are skipped, and creation copies the source's defaults. Applying the same frame a second time finds the layer and produces a different result.

`fresh_scene_appearance` saved a red section, loaded it as gray, then restored red on the second application. This violates idempotence and can corrupt the first exported frame.

Reconcile/create managed layers and their stable identities before applying their saved appearance, then apply geometry and view overrides under one guard. Verify a fresh load and first export frame without a preparatory scrub.

### R15 — P2: starting a sweep after a native plane edit ignores the requested start

Locations: [animation.py:432](../napari_animation/animation.py#L432), [ortho_slice.py:634](../napari_animation/scene/ortho_slice.py#L634).

The native edit callback adopts geometry but leaves the comparison baseline at the last programmatic plane. `add_slice_sweep` writes the intended start, then calls `_capture_state` before rendering; synchronization re-adopts the prior native edit and overwrites the new start.

After dragging to z=2, `native_edit_then_sweep` requested start=4/stop=6 and captured start=2/stop=6. Advance the adopted baseline under the shared event adapter, and synchronize before applying a new authoring command rather than between mutation and render.

### R16 — P1: discrete interpolation policies remain incomplete

Location: [frame_sequence.py:118](../napari_animation/frame_sequence.py#L118).

The new active-section policy does not address `dims.order`, section source/source_id, selected level, projection, or render mode. Interpolating XY to XZ produces the invalid order `(0, 0, 2)`; changing source switches to the destination as early as fraction 0.1. Levels can interpolate numerically. In addition, all policies still receive eased progress, so BACK/ELASTIC overshoot can switch nominally end-only discrete values early.

The order and source cases were reproduced by `interpolation_order` and `interpolation_source`; the overshoot issue follows directly from the frame index retaining only eased progress at lines 163–166.

Use raw progress for discrete fields and explicit policies for every identity/mode/order field. Hold the full source binding and compatible geometry at the boundary. Validate all generated orders and test overshooting easing plus source/level/view changes.

### R17 — P2: time volumes pass N-D tuples to a 3D plane API

Location: [ortho_slice.py:477](../napari_animation/scene/ortho_slice.py#L477).

The world conversion helpers return one coordinate per source dimension, but `SlicingPlane` requires three spatial coordinates. Adding a section to `(T, Z, Y, X)` data raised a tuple-too-long validation error in `time_volume`.

Select the leading time/channel indices explicitly and pass only spatial geometry to the plane, with transforms restricted to the appropriate subspace. The oracle also currently reduces only the depth axis and leaves leading dimensions, so it is not a 2D pixel oracle for time volumes. Add time-index and mixed-dimensional-layer tests.

### R18 — P1: a visible 3D section becomes blank in 2D if its source volume is hidden

Locations: [ortho_slice.py:466](../napari_animation/scene/ortho_slice.py#L466), [ortho_slice.py:510](../napari_animation/scene/ortho_slice.py#L510).

Hiding the volume while showing its section is a normal workflow. In 2D the renderer hides the section backing layer and relies on the source being visible, without a view override. `hidden_source` retained an enabled active section but had **zero visible images**.

Use a managed 2D section output with the section's appearance, and restore source visibility/projection settings as authored when leaving section view. Do not require users to expose the source volume to see an enabled section. Section 3.1 of the guide still requires this, while the modified B05 step 3 describes the conflicting source-layer shortcut.

## Consistency with the plan

| Milestone | Current claim | Review assessment |
|---|---|---|
| M2 | Done | Partial: renderer is a delegation facade; registry does not enforce managed ownership/ID uniqueness; there is no common application transaction. |
| M3 | Done | Partial: working-scene restoration helps, but empty snapshot cleanup and discrete interpolation fail. |
| M4 | Done for local bindings | Partial even locally: source-name reuse, duplicate IDs, additional affine, time volumes, and 2D level selection fail. |
| M5 | Done for native 2D | Partial: simple positive-scale XY examples pass; other view axes, user scrolling, empty slabs, flips/swaps, hidden sources, and public rejection paths fail. |
| M6 | Done for current schema | Partial: atomic writing exists, but binding/migration, full validation, and first-application appearance do not meet save/reopen/edit requirements. |
| M7 | Pending | Correctly pending. Scalar slice comparisons and a nonblank screenshot do not establish scientific 3D projection or smooth-transition parity. |

Full oblique true-2D pixel validation was **not implemented**. The guide permits rejecting oblique true-2D in its initial scope, which is a reasonable staged restriction, but the user-requested full capability remains outstanding and even that rejection is not enforced by all public entry points (R08).

The numerical tests cover one positive-scale XY slab with max/mean/min. The production oracle shares `_sample_interval` with the renderer; that is not an independent oracle for interval selection. Add hand-computed bounds and expected scalar arrays for odd/even/subvoxel/boundary/transform/time/level cases, then compare actual output. The earlier 304-pass total and this review's 84-pass focused run do not demonstrate those untested behaviors.

Two additional plan gaps visible in the code should remain on the implementation checklist: the new half-open sampling convention is not tagged/migrated in saved scene state despite the guide's compatibility requirement, and `Animation.animate` still lacks the planned restoration of working/view state in its `finally` block. These are static consistency observations, not additional newly reproduced regressions in this review.

## Recommended repair order

1. Establish unique source IDs, managed-layer ownership by object ID, and explicit project binding/migration. Protect existing layers and timelines first (R01–R05).
2. Implement one renderer transaction: validate, reconcile/create/hide managed layers, apply transforms and appearance, set view overrides, apply camera last; restore on failed commands. Unify snap/selection/preview/export paths (R08, R12–R15).
3. Complete active-section authoring, GUI selection, and dims scrolling, then render the selected source level through managed output (R06–R07, R09, R18).
4. Complete numerical contracts for empty slabs, transforms, and leading time dimensions using independent expected values (R10–R11, R13, R17).
5. Complete raw-progress discrete policies and genuine v1/v2 reopen/edit/reorder tests (R02, R16), then update milestone claims and proceed to oblique resampling/GPU acceptance.

Do not expand the resampling feature on top of the current ownership and state-application defects; it would add another output path whose state could diverge.
