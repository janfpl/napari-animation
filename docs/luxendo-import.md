# Processed Luxendo folder import

The animation dock now has an **Import Luxendo folder** panel. Paste a processed
task folder or browse to one, then select **Import channels**. Keep **Include
subfolders** checked to start from an enclosing `processed` folder.

The reader follows the channel discovery and embedded metadata conventions in
`reference_repos/shifter-main/shifter/h5_utils.py`. It skips `main*` and
`*_bdv.h5` companion headers and imports each valid image file as a separate
layer. It supports `channel` and `channel_id`, including channel zero.

Spacing comes from `processingInformation.voxel_size_um`: depth maps to Z,
height to Y, width to X. Every layer keeps its own spacing and micrometer units.
The Voxel size controls select the first imported channel and display its
values. Full metadata precision stays on the layer; the controls show six
decimal places. No manual application of voxel sizes is required.

## Loading and lifetime

- Folder preparation, metadata parsing, and bounded contrast sampling run in
  a worker. Each contrast sample is one plane, at most 128 by 128 pixels,
  from the coarsest available level.
- Full image levels use lazy Dask arrays over read-only HDF5 datasets.
  Existing direct HDF5 section optimizations still recognize these arrays.
- Qt creates layers individually on the GUI thread. Initial napari slicing
  and texture creation can still pause that thread, especially with cold
  files. Each new layer applies napari's Auto contrast to its displayed slice.
  Background preparation does not make that display step asynchronous.
- Duplicate paths already imported by this reader are skipped. Invalid
  metadata is reported instead of silently assuming unit voxel sizes.
- Cancellation discards pending work and keeps successfully added channels.
  Closing the dock cancels pending imports, while loaded and derived image
  arrays retain their file handles until their last reference is released.
- Saved keyframes record the Luxendo reader and source path, allowing the
  same image files to reopen with their physical spacing.

Timepoints and tiles remain separate layers. This importer does not assemble
a time series, register tiles, or apply acquisition transforms.

## Validation on Embryo8b

Tested through the native Windows Qt interface using
`D:\081926\Embryo8b\processed`:

- One channel imported as **ch0 — 561**; companion headers were excluded.
- Native shape: **487 × 2048 × 2049** (Z/Y/X), uint16.
- Metadata spacing: **0.7 × 0.29250002 × 0.29250002 µm**.
- All four supplied pyramid levels remained available.
- Native data size: **4,087,246,848 bytes**, retained as a lazy array.
- Repeating the import added no duplicate layer.
- Image slices remained readable after closing the animation dock.

The first measured GUI import took **4.11 seconds**, including a **4.08-second**
gap in the Qt heartbeat during initial layer display. Subsequent runs after
filesystem/cache warming took approximately **0.39–0.84 seconds**. These are
observations from one machine, not a cold/warm benchmark comparison or a
guaranteed loading time. Metadata preparation took 0.005–0.042 seconds.
The latest measurement and exact source path are saved in
[`luxendo-import-validation.json`](dev-notes/luxendo-import-validation.json).

Regression coverage includes multiple channels with different X/Y/Z spacing,
recursive discovery, missing/corrupt metadata, pyramid validation, lazy reads,
file lifetime, duplicate prevention, keyframe save/reload, GUI-thread ownership,
responsive metadata preparation, cancellation, and dock closure.

The native Windows/Qt regression suite passed: **474 tests**, including 12
new Luxendo import cases. Existing Qt theme and non-orthogonal slicing warnings
remain.
