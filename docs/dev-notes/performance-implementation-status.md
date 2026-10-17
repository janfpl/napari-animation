# Performance implementation status — 2026-09-23

The answer to “is the entire plan implemented?” is **no**. This work delivers
the main responsiveness changes in `napari_animation`, but shared native-reader
caching, a measured CUDA path, and live-file GUI profiling remain
open. The application tree is an unpacked source directory rather than a Git
checkout. The extracted Shifter and Imaris-loader repositories were treated as
references; their source was not installed or modified by this pass.

The subsequent [code-review fixes](performance-review-fixes-2026-09-23.md)
address all seven reported correctness issues. The final native Windows suite
passes **462 tests**, including an exact comparison of GUI-export captures
with synchronous OpenGL screenshots. The supplied IMS/Luxendo worker benchmark
also passes with the accelerated reader paths retained.

## Parallel HDF5 reader implementation

Large compressed 3D HDF5 slabs now use up to four independent read-only
processes from the first request. The section worker starts the pool without
blocking Qt, then divides output into native-chunk-aligned stripes. This
avoids h5py's global thread lock during
gzip decompression. An unchanged direct h5py graph qualifies. The exact
singleton-time, singleton-channel graph emitted by `napari-imaris-loader` also
qualifies when its `ims_reader` is read-only, unsqueezed, resolution-locked,
and the selected HDF5 dataset's shape and dtype match. Other transformed
Dask graphs keep their prior path. A process startup/read failure
falls back to the thread worker; source switches and viewer close release the
pool. Auto mode requires a compressed source and at least 64 MiB of estimated
native chunk work. `NAPARI_ANIMATION_H5_PROCESS_LANES=0` disables it; an
integer from 2 to 8 forces that lane count. Auto mode reduces to two lanes
below 4 GiB available RAM and disables the pool below 2 GiB.
For uncompressed sources whose output crosses at least 32 Dask chunks, the
worker instead reads the verified HDF5 dataset directly with one independent
read-only handle. This avoids Dask scheduling overhead without process startup.
The graph match follows the [upstream `ims_reader` implementation](https://github.com/CBI-PITT/imaris_ims_file_reader/blob/master/imaris_ims_file_reader/ims.py),
which reads its selected resolution's HDF5 `Data` dataset without a voxel transform.

`benchmarks/benchmark_h5_production_backend.py` exercised the actual Qt
projection service on the supplied files and compared every output pixel with
a direct HDF5 reduction. On `single tile.ims`, resolution 1 took 1.01 s for
the first four-plane slab including process startup and 0.15 s thereafter.
The matching synthetic `napari-imaris-loader` graph took 1.01 s first and
0.15 s thereafter. Full resolution took 1.51 s first and 0.68 s
thereafter; the comparable
single-reader warm run was about 2.43 s. The four lightweight workers used
about 530 MiB combined RSS and all exited on service close. The measured
maximum Qt heartbeat gap for those IMS runs was 11 ms. Embryo8b's uncompressed
Luxendo `Data` stayed on one worker and used the direct HDF5 path: 0.10 s per
four-plane slab from the first request versus 0.34–0.38 s through native-chunk
Dask in the earlier production run. The synthetic Imaris-header graph that
points to the same Luxendo data also took about 0.10 s. Results are in
`docs/performance-h5-production-processes.json`. The separate
`benchmarks/benchmark_h5_process_lanes.py` records the 1/2/4/8-lane comparison
at IMS resolutions 0 and 1 and on Luxendo. All figures can benefit from the
OS file cache and exclude a live OpenGL canvas.
The synthetic graph validates dispatch and pixels, but the upstream reader
package is absent from this environment, so it is not an installed-plugin
benchmark. At resolution 1 with a warm OS cache, the pool's 1.01 s first request is slower
than the approximately 0.58 s single-reader result; repeated reads are faster.
This first-request crossover and the roughly 530 MiB worker footprint should
be checked in the user's actual GUI before changing the default lane policy.

The complete native-Windows suite passed **443 tests**. It covers numerical
parity for all three axes and max/min/mean, external-link headers, process
fallback, source-switch cleanup, low-memory policy, GUI export/prefetch,
and OpenGL pixel-oracle checks. A hidden napari window produced a real canvas
screenshot. These checks use small test images, not the supplied files in a
user-visible napari session.

## Follow-up implementation and validation

Export lookahead now groups frames by the Dask chunks they require. Adjacent
scroll frames that select different planes in the same Z chunk share one
background cache-warming task; cancelled windows stop queued work and running
tasks stop between layer reads. The task remains bounded by the lookahead
depth. Native non-Dask arrays are deduplicated only for identical indices,
because reading one thin slice does not guarantee its full chunk stays cached.

`benchmarks/benchmark_h5_prefetch_dedup.py` compares the former concurrent
frame-based pattern against this grouping on the supplied files. With three
lookahead frames, `single tile.ims` resolution 3 made **48 versus 16** HDF5
dataset reads; Embryo8b `Data_4_4_4` made **192 versus 64**. Both grouped
cases scheduled one task. These are backend read counts, not a measured live
GUI frame-rate gain. Results are in `docs/performance-prefetch-dedup.json`.

The Qt export writer now performs environment inspection and finished-video
verification, including full decode and FFmpeg probing, off the GUI thread.
A test with a deliberately slow verifier confirmed that Qt timers kept firing
while verification ran on the writer thread.

A direct-HDF5 bypass for oversized Dask blocks was benchmarked and
**not enabled for that layout**. On the OS-cache-warm full-resolution Embryo8b `Data` slab,
the existing Dask path took 0.101 s and added 59 MiB peak RSS; direct h5py
took 0.093 s and added 56 MiB. Pixels and sums matched. The improvement was
too small to justify bypassing Dask's shared cache and reader semantics.
In contrast, the native 64³ Dask graph scheduled more than a thousand chunk
tasks for the same slab. Its 0.33 s versus 0.09 s direct-HDF5 result motivated
the guarded uncompressed bypass above. The production run confirmed exact
pixels and approximately 0.10 s service latency; a forced four-process Luxendo
run reached 0.06 s after a 0.91 s first request and used about 557 MiB of
child-process memory, so it is not the default.
`benchmarks/benchmark_h5_block_overread.py` and
`docs/performance-lux-forced-processes.json` preserve these comparisons.

An earlier affected headless suite passed **145 tests**, including the new
chunk-deduplication and writer-thread checks. A broader attempted run added
five OpenGL setup/teardown errors from the Windows offscreen context. The
later native-Windows suite passed those OpenGL checks.
The headless IMS/Luxendo interaction benchmark was repeated after this pass:
Snap XY stayed at 0.10–0.12 s and the two sweep buttons and keyframe selection
stayed at 0.011–0.015 s across its three tested pyramid sources. The export
prefetch improvement does not change those controller paths.

## Implemented and tested

| Plan area | Current implementation |
|---|---|
| Reported sweep buttons | `add_scroll_through` and `add_slice_sweep` build two detached endpoint states from one capture, insert them together, render the final endpoint once, and take no synchronous endpoint screenshots. A generated thumbnail is captured after exact data is ready when that keyframe is selected. |
| Keyframe selection and scene application | One guarded navigation render; snapshot preflight no longer runs twice. Unchanged source binding, plane geometry, appearance and 2D projection data avoid repeat assignments. A 2D section no longer toggles its source visible on every apply. |
| Async 2D sections | One coordinator worker performs bounded basic ROI reads and reductions, using independent processes for qualifying large compressed HDF5 slabs and a direct read for unchanged uncompressed HDF5 graphs with costly Dask scheduling. At most one job is active and one is pending; obsolete jobs are cancelled between reads and cannot publish. Identical in-flight requests share the same producer. Qt/napari layer writes stay on the GUI thread. Headless calls stay synchronous for deterministic scripts/tests. |
| Cache and large slabs | A 128 MiB projection LRU keys by source ID/revision, array identity, level, leading indices, axis, interval and operation. Large slabs use output tiles aligned to available Dask/HDF5 chunk dimensions within a 64 MiB logical read target. A shared, byte-bounded interactive Dask cache defaults to 512 MiB and is released with the dock or viewer window; `NAPARI_ANIMATION_INTERACTIVE_CACHE_MB=0` disables it. Export logging reports the configured cache budget rather than current occupancy. |
| Source edits | Source data events and explicit refresh events invalidate projections. `Animation.invalidate_source(name_or_id)` covers in-place edits to a source hidden by the 2D section, for which napari does not emit `set_data` on ordinary `refresh()`. |
| Progressive quality | For a finer selected level, a multiscale 2D section first requests the coarsest level, then the selected level after 300 ms idle. The scene editor exposes resolution levels. Export bypasses preview and requests exact quality. |
| Prefetch | Napari `MultiScaleData` is accepted; managed and invisible layers are skipped; layer transforms and downsample factors map frame world points to array indices. Requests are frozen on the GUI thread. Adjacent frames sharing Dask chunks use one producer; old lookahead futures are cancelled and the backlog is bounded. Read failures are logged. |
| GUI export | The Save button now uses `QtExportController`: a timer advances frames, waits for current section and native layer readiness, captures in order, and puts frames into a bounded queue without blocking Qt. Resize, diagnostics, image writes, encoding, muxing and finished-video verification run in the writer thread. The pre-existing synchronous `Animation.animate()` remains available to scripts. |
| Other scripts | The Save dialog uses widget geometry for size estimates before attempting a screenshot. The interpolated-frame cache is capped at 128 states, so long exports do not retain every frame. Legacy ortho-slicer control changes coalesce over 40 ms. |
| Instrumentation | Setting `NAPARI_ANIMATION_PERF_TRACE` to a JSON path records controller phase timing, worker projection timing, thread IDs and Qt heartbeat gaps over 100 ms; the dock saves the trace on close. |

The section-processing tests compare NumPy max/min/mean across all three axes
and integer/floating dtypes. They also exercise a 100-request scrub, in-flight
deduplication, section cancellation, cached revisits and coarse-to-fine
refinement. Export tests verify ordered PNG output, scene restoration, and
continued Qt heartbeats with an intentionally slow writer and a full queue.
An earlier selected headless regression set passed **144 tests**; 10 cases requiring
an OpenGL canvas or a restricted temporary-directory fixture were excluded.
An earlier broader attempt encountered the Windows offscreen OpenGL limitation
and sandbox temporary-directory permissions; the later native-Windows run
with a fresh workspace temp directory passed the complete suite.

## Real-file evidence

`benchmarks/benchmark_h5_interactions.py` exercises the production controller
with a headless `ViewerModel` and small Imaris/Luxendo pyramid levels. Its
latest JSON is `docs/performance-after-second-optimizations.json`.

| Source / selected dataset | Snap XY | Add scroll-through | Add slice-sweep | Select keyframe |
|---|---:|---:|---:|---:|
| `single tile.ims`, resolution 3 | 0.119 s | 0.012 s | 0.012 s | 0.015 s |
| `Embryo8b/main_st-1.ims`, resolution 2 | 0.105 s | 0.012 s | 0.012 s | 0.014 s |
| Embryo8b direct `.lux.h5`, `Data_4_4_4` | 0.107 s | 0.012 s | 0.012 s | 0.014 s |

The same script measured 0.23–0.32 s for the two sweep buttons after the
first optimization pass. The new call count is one section projection per
sweep and zero endpoint screenshots in the headless probe. These timings can
benefit from the OS disk cache and do not include Qt painting or GPU upload.

Before the direct HDF5 bypass, `benchmarks/benchmark_async_h5_projection.py`
directly exercised the Qt worker
with real bounded slabs, including the full-resolution Embryo8b `Data`
dataset. On one run the four-plane full-resolution Luxendo projection took
10.25 s while the Qt heartbeat had a 10.7 ms p95 gap and 76 ms maximum gap.
A later OS-cache-warm run took 0.41 s with a 10.3 ms p95 gap and 81 ms maximum.
See `docs/performance-async-h5-benchmark.json`. This proves the worker can
keep a minimal Qt event loop responsive during that read; it does not measure
the live napari/VisPy canvas, paint, input handling or true cold-storage
latency. No whole 27.5 GB IMS volume was materialized.

## Open items from the plan

1. **Native reader cache and lifecycle.** The interactive Dask cache and
   tiled requests are useful, but there is not yet a single native-HDF5-chunk
   LRU shared with the separate Imaris/Luxendo reader plugins, nor a stable
   source/dataset descriptor protocol, in-flight chunk deduplication or
   coordinated file-handle lifetime. The extracted reference readers are not
   the installed GUI reader. Implementing this in the active reader package
   requires identifying and testing that installed package/environment.
2. **Broader reader parallelism and optional preview sidecars.** The Windows
   spawned-reader pool handles large compressed direct HDF5 sources and the
   reference Imaris loader's unchanged singleton time/channel graph. Other
   reader layouts and multi-timepoint/channel graphs need a source descriptor
   protocol; there is no shared-memory transfer path or background sidecar pyramid writer.
   The supplied files already have useful pyramids, so a sidecar would add
   cost without a demonstrated need for these examples.
3. **CUDA and Numba enablement.** The first pass added an optional pure Numba
   reduction prototype and benchmarked it. It did not clear the plan's 20%
   end-to-end gate on these HDF5 workloads; reads dominate. No compatible
   CuPy/CUDA runtime is present in this project environment, so a CUDA path
   was not enabled or claimed as tested. The RTX 3070's existing OpenGL
   rendering is already GPU work and is a separate bottleneck to profile.
4. **Full GUI acceptance gates.** Native Windows OpenGL unit tests now pass,
   but a user-visible napari session with the actual installed reader must
   still be profiled for callback p95, heartbeat
   p95/max, screenshot/texture upload, cache bytes, RSS and VRAM. The full
   suite passed using native Windows Qt and a fresh workspace temp directory.
5. **Thumbnail completion and export polish.** Generated endpoint thumbnails
   are refined on selection, so an endpoint never selected still has its
   placeholder thumbnail if keyframes are saved immediately. The GUI export
   pump is tested for PNG order/slow writers, cancellation, concurrent edits,
   and OpenGL capture equality; live MP4 plus async
   napari slicing, cancellation during an actual FFmpeg failure, and pixel
   comparisons of exported video frames against a real OpenGL render remain
   to be verified.

The next decisive measurement is a live napari session using the actual
installed Imaris/Luxendo reader with `NAPARI_ANIMATION_PERF_TRACE` enabled.
That trace and a simultaneous GPU memory/paint profile will show whether
further effort belongs in native HDF5 chunk caching, 3D texture upload,
OpenGL paint, or encoding rather than CUDA reductions.
