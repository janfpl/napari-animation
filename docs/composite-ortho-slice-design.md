# Composite multi-channel ortho slices

Status: proposed implementation design; runtime behavior is not changed.

## User-facing behavior

Extend the existing `OrthoSlice` into one scene object containing any number of channel bindings. All enabled channels share one world-space position, normal, thickness, and projection method. Each channel retains its own color and intensity settings. Moving or animating the slice moves every channel together.

Replace the Source dropdown with a scrollable, searchable Channels checklist populated from existing scalar Image layers with at least three spatial axes. Exclude every renderer-owned layer by metadata, not by name. Channel axes already split into separate napari layers work directly; interpreting an unsplit channel dimension or RGB input is outside this change.

```text
Ortho slice                         [✓ Enabled]

Channels                         [Search…    ]
[✓] ■ DAPI          [Linked]          Ready
[✓] ■ GFP           [Linked]          Ready
[ ] ■ mCherry       [Linked]          Ready
    2 channels enabled          [All] [None]

View        [XY ▾]         [Edit in canvas]
Position    [────────●────────────────────]
Thickness   [10.0]         Projection [Max ▾]
Resolution  [Native ▾]     [Per-channel…]
```

Checking a channel renders its slice immediately; unchecking removes its contribution without deleting its settings. All/None applies to eligible rows matching the current search. An empty selection is valid and renders nothing. New source layers appear unchecked. The scene Enabled checkbox switches the entire composite on or off while preserving channel choices.

Channel selection is independent of the source volume's visibility in 3D: users can hide the full volumes and keep their colored slices visible. Source opacity is an appearance setting and still applies in Linked mode. Show the effective color swatch and expose per-channel opacity, colormap, contrast limits, and gamma through a row editor. Default to Linked appearance; editing a slice-specific setting switches that channel to Custom, with a reset-to-source action.

Use the union of participating sources' transformed world bounds for slice navigation, projected onto the current normal. A channel outside the slab contributes nothing. Show missing sources as unresolved rows with a relink action; never bind a replacement just because it has the same name. Incompatible sources get a reason and cannot be newly enabled. Validation happens before changing the live scene.

## Rendering decision

Use one managed scalar napari Image plane per enabled channel, owned by the same scene object, with `depiction='plane'` and `blending='additive'`. Project each channel independently through the shared slab, apply that channel's transfer function, then add its color contribution. Do not merge scalar volumes before projection: that would lose channel-specific contrast and can change the result of max/min projection.

Conceptually, the composite color on a black background is:

`RGB(x) = clamp(sum(enabled_i * opacity_i * LUT_i(normalize_and_gamma(project_i(x)))), 0, 1)`

Colormap alpha also modulates a channel's contribution. This describes the intended color behavior; final framebuffer alpha and interaction with other scene objects follow napari's additive blending. Bright overlaps can saturate. There is no fixed RGB-channel count or channel-count limit in the model; practical limits are available GPU memory and rendering time.

Napari's official [3D plane example](https://napari.org/0.9.0/gallery/3D_image_plane_rendering.html) uses additive blending for a plane alongside a volume. This supports reusing the existing plane renderer, but does not prove correct depth behavior for several coincident planes on this application's installed renderer. A visual acceptance test with overlapping colored planes, volumes, surfaces, and different camera angles is a release gate. Do not introduce geometric offsets to hide depth artifacts: all channels must sample the same physical slab.

CPU data references remain shared with each source's selected volume/pyramid level. Each active plane can still require its own GPU texture. Initially create planes lazily, retain hidden ones only within a bounded resource policy, and allow eviction/recreation by deterministic identity. Do not upload every unchecked channel. Report allocation failures per channel and offer coarser resolution; never silently drop a channel or silently lower export quality.

A future common-grid RGBA plane renderer could reduce GPU volume memory, but would require oblique resampling, resolution policy, asynchronous composition, and separate picking/manipulator integration. It is not required for this first implementation.

## State and identity

Keep `kind='ortho_slice'` and the existing scene ID so selection, active section, and animation tracks survive. Add a `channels` mapping keyed by stable source ID. Illustrative serialized state:

```json
{
  "kind": "ortho_slice",
  "id": "section-1",
  "name": "Composite section",
  "enabled": true,
  "position": [25, 0, 0],
  "normal": [1, 0, 0],
  "thickness": 10,
  "projection": "max",
  "render_mode": "plane",
  "sampling_convention": "voxel-centers-half-open-v1",
  "channels": {
    "source-a": {
      "source_id": "source-a",
      "source_name": "DAPI",
      "enabled": true,
      "source_level": 0,
      "appearance_mode": "source",
      "appearance": null
    },
    "source-b": {
      "source_id": "source-b",
      "source_name": "GFP",
      "enabled": true,
      "source_level": 0,
      "appearance_mode": "custom",
      "appearance": {
        "colormap": "green",
        "contrast_limits": [0, 4095],
        "gamma": 1,
        "opacity": 0.8
      }
    }
  }
}
```

Require each mapping key to equal its binding's source ID. Resolve using `LayerRegistry`; names are display labels and legacy migration hints only. A source cannot appear twice in one composite. Explicitly distinguish absent legacy `channels` from an empty modern mapping: empty must never trigger the old first-image fallback.

Store per-channel resolution because sources may have different pyramid depths. The common Resolution control offers Native and Coarsest; explicit level choices live in the row editor and are validated against that source. Preserve selected levels through toggles and keyframes. Different voxel sizes and shapes are supported when sources are registered in a common world coordinate system; this feature does not perform image registration or infer unit conversions.

Managed layer identity becomes `(section_id, role, channel_source_id)`, with roles remaining exactly `section_3d` and `section_2d`. Add a separate metadata key for the originating channel ID; keep `napari_animation.source_id` as the managed layer's own unique identity. Extend `managed_id`, `mark_managed`, and `find_managed` with a channel argument and add iteration over all layers owned by a section. Role names must remain stable because transient-layer filtering currently compares the exact role string.

Labels can be `Composite section · GFP`; labels never determine ownership. All deletion, renaming, resource release, and reconciliation operate on the complete owned set. Two composites can use the same source without sharing editable plane objects.

## Geometry, appearance, and external edits

Reuse `_level_volume`, `_spatial_affine`, world/data position and normal conversion, and physical thickness conversion per binding. Compute each channel's plane from the shared world geometry and that channel's transform and level factors. Copying the first channel's data-space plane to all channels would misalign anisotropic or translated data.

Extract source-specific helpers so rendering accepts a binding explicitly; do not temporarily mutate `self.source_id` while looping. Preflight all requested sources and saved transforms before allocating or modifying layers. Publish the geometry update under the existing renderer application guard.

Keep one manipulator attached to a deterministic enabled, resolved channel. Convert a manipulation back into world geometry, then immediately propagate to every sibling plane under a reentrancy guard. Track the last applied plane separately for each backing layer. Reattach the manipulator if its channel is disabled, removed, or evicted. External changes to any managed plane's geometry or projection update the shared object; hiding one managed plane toggles that channel, while the scene checkbox controls the whole object. Temporary renderer suppression must not become an authored toggle.

Linked appearance is read from the source after source layer state is restored for the frame. Subscribe to source appearance/data/transform events and update all affected bindings without recalculating scalar projections for a color-only edit. Custom appearance is owned by the channel state; edits made through napari controls on a backing layer switch it to Custom. Reuse existing colormap serialization, including custom LUTs, rather than saving only names.

Managed planes are derived render outputs in the new format. Capture their external edits into channel state first, then omit them from ordinary layer snapshots to avoid conflicting appearance/visibility authorities. Source layers remain captured normally. Disconnect listeners when bindings are removed or renderer ownership changes during playback.

## 2D projections, scheduling, and visibility

The requested feature is in 3D, but the same scene object already participates in 2D viewing and export. Preserve that behavior for every channel.

For an active 2D composite, create a managed additive projection layer per enabled channel using its own affine. Validate native 2D support for every resolved enabled source, including captured transforms; reject the entire view switch before mutation if any requires unsupported oblique resampling. Missing sources remain relinkable while valid channels render.

Generalize `_suppressed_source_id` to a set. While the composite owns the 2D canvas, suppress all resolved bound source layers, including unchecked bindings, so an unchecked channel cannot leak through its ordinary image layer. Preserve each source's authored visibility and restore it when leaving 2D, removing the binding, or changing active section. Merely displaying an unchecked candidate in the channel picker does not create a binding or take control of that source. Unrelated source layers retain their existing behavior.

`ProjectionService` currently has one active read, one pending read, and generations keyed by section ID. A loop over its existing `request(section_id, ...)` would invalidate sibling requests and overwrite pending work. Replace single requests with composite batches:

1. Freeze one section revision containing enabled source IDs, source revisions, selected levels/timepoints, transforms, geometry, and projection method.
2. Build one scalar projection plan per enabled, resolved channel. Reuse source-based cache entries; color changes do not invalidate scalar data.
3. Keep a bounded queue for the current batch, with the existing bounded worker count. A newer section revision supersedes the previous batch; sibling channel jobs within a revision never cancel each other.
4. Collect results, including explicit empty/out-of-bounds results, and publish the completed batch together on the GUI thread. Discard callbacks with stale section revision or source binding.
5. During scrubbing, retain the previous coherent composite until the new batch is ready. Hide disabled/removed channels immediately. Clear previews when changing source, orientation, or timepoint so old content cannot masquerade as a new composite.
6. Coarse preview and exact refinement are whole-composite stages. Export waits for all exact results before capture and reports missing/failed required channels instead of producing an incomplete movie silently.

Cancellation, source invalidation, `when_idle`, and shutdown must account for queued jobs and staged results. Keep the scalar cache and staged-result memory bounded across channels. Share/reuse plans with existing prefetch and HDF5/process backends; do not create one worker pool per channel.

In clip mode, contribute the same world slab converted independently to each enabled source, using the existing clipping compositor. Clip mode retains source-layer appearance and visibility semantics; expose custom slice appearance only in plane mode. Two clip objects targeting one source still intersect their constraints, consistent with the current compositor. Switching modes hides all obsolete managed outputs.

## Keyframes and compatibility

Channel membership, enabled flags, source identity, level, appearance mode, and colormap are discrete and switch at the destination keyframe using `STEP_END`. Shared geometry keeps existing position/thickness interpolation and normal SLERP. Numeric custom appearance can interpolate only when the same binding has Custom appearance at both endpoints; otherwise hold the complete appearance until the destination.

Do not rely on generic dictionary interpolation for channel membership: it currently carries one-sided keys through the entire interval. Implement explicit channel-map interpolation so a destination-only channel cannot appear early and a removed channel cannot linger after arrival. Test fractions 0, intermediate values, and 1, including backwards scrubbing. A disabled channel retains its settings.

Bump the saved animation version from 2 to 3 and accept versions 1, 2, and 3 on read. Migrate each legacy single source into one enabled binding, preserving source ID, level, scene ID, geometry, and projection. Resolve name-only legacy state through existing source normalization; retain an unresolved placeholder identity/name when the source cannot be found. Never retarget a missing established ID.

Migrate legacy `section_3d` layer snapshots into that channel's Custom appearance so existing movies retain independently edited slice colors, gamma, opacity, and contrast. Use the source appearance only when no legacy backing appearance exists. Convert legacy managed IDs in saved references and adopt/relabel existing managed layers without duplication. Remove their redundant layer snapshots only after migration has captured their authored state. Define precedence as scene geometry/enabled state plus legacy backing appearance. Existing Python `source`, `source_id`, and `source_level` arguments remain a supported single-channel shorthand; reject ambiguous use alongside a non-legacy `channels` mapping.

Older applications must reject version 3 explicitly instead of silently displaying only one channel. Migration and parsing finish before modifying the live viewer; failed loading preserves the current scene.

## Implementation map

| Location | Required change |
| --- | --- |
| `scene/ortho_slice.py`, `scene/base.py` | Channel schema, normalization, per-binding helpers, shared geometry and event ownership, serialization |
| `layer_registry.py` | Channel-qualified managed IDs, lookup and iteration, full-owner cleanup |
| `_qt/scene_widget.py` | Channel checklist, swatches, appearance/level editor, missing-source state, navigation bounds, manipulator selection |
| `animation.py`, `scene/scene.py` | Validated channel mutations; reconcile, rename, remove and restore every owned layer |
| `scene/renderer.py`, `viewer_state.py` | Multi-source preflight and visibility overrides; derived-output capture policy |
| `scene/processing.py` | Batch scheduler, coherent publication, cancellation and exact export barrier |
| `frame_sequence.py`, `io.py` | Identity-based channel interpolation, version 3 migration, backward compatibility |
| `resolution.py`, `prefetch.py`, `export_estimates.py` | Enumerate every enabled binding for resolution changes, work planning and estimates; audit singular source assumptions |

Implement in this order: (1) schema/identity/migration and single-channel parity; (2) multi-plane 3D rendering plus visual depth validation; (3) checklist and shared manipulator; (4) 2D batching, visibility and export integration; (5) keyframe and performance acceptance. Ship the user-visible feature only after all paths agree.

## Acceptance criteria

- One composite displays red and green fixtures together; red-only, green-only and overlap regions have the expected colors. Reordering source layers does not change channel membership or additive color results. Test more than three channels, zero channels, and all channels.
- XY/XZ/YZ, oblique movement, thickness changes, and max/mean/min apply to every enabled channel. Numerically verify world alignment with anisotropic scales, translations, axis permutations and different pyramid levels. Do not require unlike voxel grids to produce identical samples.
- Full-volume visibility remains independently controllable in 3D. Entering/leaving 2D and switching active sections restores every source's authored visibility, including initially hidden sources and unchecked bound channels.
- Editing geometry from any backing layer or the manipulator updates the entire composite without feedback loops. Toggling a channel preserves its appearance and level; disabling the composite preserves all toggles.
- Source rename, deletion, same-name replacement, relink, new source insertion and duplicate section names never redirect bindings or delete user layers. Removing one composite releases only its layers, listeners and jobs.
- Linked/custom appearance, custom LUTs, channel membership and geometry survive capture, interpolation, backwards playback, save/reload and video export. Legacy version 1/2 fixtures retain their rendered appearance.
- A deliberately slow three-channel projection batch completes every sibling job. Rapid A-to-B-to-C scrubbing publishes only the latest complete revision; no old result revives an unchecked channel. Test cancellation, source refresh, per-channel failure and export waiting for the slowest channel.
- GPU visual tests verify coincident colored planes and their interaction with surrounding objects. Benchmark 1/4/8/16 channels on a recorded representative dataset; record GPU/CPU memory, median/p95 interaction latency, reads and exact export time. Establish supported hardware expectations from measurements rather than promising unlimited throughput.

This design is based on inspection of the current model, Qt editor, registry, projection service, renderer, interpolation and persistence code. Implementation tests and GPU acceptance checks remain to be run once the feature is built.
