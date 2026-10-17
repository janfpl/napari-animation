# Follow-up review and fixes — 2026-09-21

Scope: review the renderer/registry repair pass documented in
[the preceding review](review-last-two-rounds.md), reproduce remaining
failures, and fix them without changing the canonical-scene architecture.
This workspace has no Git metadata; the review used the current source,
the preceding review record, and regression tests rather than a commit diff.

## Findings addressed

| Area | Failure | Fix |
|---|---|---|
| Destination display mode | Preparing an oblique 3D snapshot while the viewer was in 2D rejected the snapshot against the current mode. | Both validation and preparation use the destination mode. |
| Destination transforms | Valid snapshots were rejected against the current source transform; invalid destination transforms failed after changing the source. | Validate the composed destination transform from captured scale, rotation, shear and affine before applying state. JSON matrices are converted to arrays. |
| Render callbacks | The render guard covered only scene application, leaving dims, layer setup, appearance and view-mode changes able to enter native-edit callbacks. | A nesting-safe guard covers snapshot application and controller view changes. Native plane callbacks honor it. |
| Disabled active section | Disabling the selected section raised an exception or selected a different enabled section implicitly. | Preserve the selected ID while disabled; hide its outputs and restore source visibility. |
| Absent scenes | A snapshot with `scene=None` left outputs from the previous scene visible. | Reconcile absent scenes as well as empty dictionaries, preserving legacy clipping behavior. |
| Clip mode in 2D | The renderer suppressed the source while the section hid both outputs, producing a blank view. | Clip-mode sections use a managed thick projection in 2D and clipping contributions in 3D. |
| Section appearance | Switching to 2D replaced the section's colormap, gamma, contrast and opacity with source settings. | Copy appearance from the managed section backing layer. |
| Managed names | A user-layer name collision made generated names alternate suffixes every render. | Exclude the owned layer itself when allocating its display name. |
| Source identity | Deleting a source and reusing its name silently rebound the section and snapshot appearance to unrelated data. | Stable IDs are authoritative; name lookup is reserved for unbound legacy state or explicit project binding. Missing sources hide both outputs. Auto-discovery excludes managed layers. |
| Backfilled identity | Adding a section to existing keyframes copied it before its source ID was resolved. | Resolve the source binding before copying section state into prior keyframes. |
| Canonical positions | Capture replaced authored out-of-volume positions with the dims slider's clamped point. | Adopt actual user scroll events, not every capture; time-only changes do not overwrite section position. |
| Boundary-spanning slabs | The projection array was correct, but its singleton layer was positioned at the clipped sample midpoint. Napari sliced it out at the authored plane coordinate. | Place the projection at the authored plane; the sample interval determines pixels only. |
| Time scrolling | A 3D section retained the previous timepoint after the time slider moved. | Refresh managed volumes on dims events in 3D as well as 2D. |
| Invalid editing commands | Rejected thickness/geometry edits and additions left invalid objects in the live scene. | Validate detached proposed scene state before mutation or keyframe backfill. |
| Interpolation | Orthogonal 2D orientation changes could create unsupported oblique intermediate sections; source transforms could switch or blend before the corresponding view change. | Hold section normals and source orientation transforms while in 2D. Switch source-dependent geometry together; preserve continuous compatible position/thickness interpolation. |
| Load rollback | A failed first frame application restored the timeline but retained rebound metadata, generated layers and a changed working scene. | Restore original layer bindings/names, remove newly added layers, restore scene objects, timeline selection, frame position, slicer and working camera. Revalidate every frame after binding sources. |
| Managed load records | A generated-layer record could claim a user layer with the same display name. | Recreate deterministic managed IDs through the renderer; skip them during source-file/name binding. |
| 4D clipping | Clip sections passed four-coordinate positions/normals to napari's three-coordinate clipping-plane model. | Pass trailing spatial coordinates for section slabs and cutaway planes. |
| Failed export | A screenshot failure restored the viewer but left the encoder open. | Drain the worker, close the encoder on success and failure, and restore the working scene/camera. Preserve the original failure when finalization also fails. |

## Verification

Final result: **357 passed** using real napari 0.8.0, Python 3.13 and native
Windows Qt/OpenGL. This includes 36 added regression cases, nine of which
compare real rendered pixels against independent reference images. The run
reported 75 warnings from Qt5 theme detection and intentionally transformed
test layers; no tests failed or were skipped.

The new cases are in
[test_review_followup.py](../napari_animation/_tests/test_review_followup.py),
with export-failure coverage in
[test_animate.py](../napari_animation/_tests/test_animate.py).
An older signed/permuted-axis fixture now requests an explicitly valid world
position; it had depended on capture silently replacing the requested one.

The OpenGL pixel tests compare rendered sections against a separate napari
viewer displaying a NumPy-computed projection. They cover all three spatial
axes, ordinary slabs, slabs larger than the volume, and partially outside
slabs. Both canvases have a fixed equal size and the test rejects blank
reference screenshots. Pixel tolerance is one 8-bit intensity unit.

Use the native Windows Qt backend on this host. The earlier offscreen failure
does not imply that real napari/OpenGL testing is unavailable:

```powershell
$env:QT_QPA_PLATFORM='windows'
..\.venv\Scripts\python.exe -m pytest napari_animation -q --tb=short
```

Final test results are recorded in
[review-full-suite.txt](../.test-tmp/review-full-suite.txt).
Python compilation also succeeds.

## Scope limits

True oblique 2D resampling remains unsupported and is explicitly rejected.
These fixes preserve oblique sections in 3D and make transitions to supported
orthogonal 2D views consistent. The new screenshot oracles cover orthogonal
projections; they are not a claim of full oblique 3D GPU pixel-oracle coverage
or completion of every milestone in the broader Imaris-like roadmap.
