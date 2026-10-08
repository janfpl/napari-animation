# Ortho animation rendering: measurements and improvement plan

**Update (2026-09-26):** The first three points have been implemented and
measured on the full 181-frame animation at 1280 × 720. The corrected visible
viewer render is no longer dark; the original diagnostic harness used a hidden
viewer. The definitive results, output checks, and remaining work are in
[ortho-render-performance-720p.md](ortho-render-performance-720p.md). The
640 × 480 observations below are retained as the initial investigation, not
as the reference for the implemented 720p changes.

**Follow-up:** Points 4–6 are complete. The state and prefetch changes, nine
alternating prefetch trials, cache/read audit, and encoder decision are in
[ortho-render-performance-followup-720p.md](ortho-render-performance-followup-720p.md).

The logging expansion is implemented. The measurements point to frame capture and first-use initialization, followed by state restoration. They do **not** support spending time on encoder parallelism. No rendering optimization or quality reduction has been enabled by this change.

## Workload and evidence

Input: the workspace's `../animation.json` (unchanged). Four keyframes produce 181 frames. All are 3D, with an axis-aligned, 25 µm thick clipping slab moving through three channels, using MIP and source level 0. This is the clipping-volume path, not the CPU 2D projection path.

The channels exist on disk and each has shape `(491, 2074, 2073)`, uint16, uncompressed HDF5, with `(128, 128, 128)` chunks. The three logical full-resolution arrays total about 12.66 GB (11.80 GiB); this is an array-size calculation, **not measured resident memory**. The loaded viewer also contains three managed section layers backed by full-size data, even though this animation uses clipping. Whether their storage/textures are shared must be measured.

Environment: Windows, Python 3.13.14, napari 0.8.0, NVIDIA GeForce RTX 3070. Benchmarks used actual HDF5 loading, Qt/OpenGL capture, H.264 encoding, and decode verification at 640 × 480, 20 fps. Seven original frames were sampled: 0, 30, 60, 90, 120, 150, 180. The harness materializes these sampled states before timing; its interpolation timings therefore do **not** measure full-timeline interpolation cost.

Each case ran twice in one viewer. “First” means first export in that process; it does not mean cold OS disk caches. Loading is outside export timing and took roughly 14 seconds in the later cases. These are small diagnostic samples, not repeated controlled trials or full-animation speed estimates.

| Export path | Prefetch depth | First export, 7 frames | Repeated export, 7 frames | First capture | Repeated capture mean, frames 1–6 |
|---|---:|---:|---:|---:|---:|
| Script | 4 | 11.878 s | 2.189 s | 9.013 s | 161.6 ms |
| Qt Save controller | 4 | 11.723 s | 2.275 s | 9.196 s | 164.7 ms |
| Script | 0 | 10.590 s | 1.933 s | 8.271 s | 150.7 ms |

The first script export spent 88.3% of wall time in screenshot capture. The repeated script export spent 54.2% in capture, 11.6% applying frames, and 28.0% restoring the editor. Encoding took 33 ms of worker time over all seven frames, overlapping rendering; queue submission waits were negligible. Qt's worker spent 1.418 s waiting for frames during its repeated run. That is encoder starvation, not an encoding bottleneck. Native slice waits were near zero during capture; the data had already been loaded.

Disabling prefetch was associated with a 12% shorter repeated export in this single pair. This is only an experiment worth repeating: process order, caching, instrumentation revisions and normal timing variation prevent treating it as a demonstrated gain. For a static 3D volume, prefetch can request whole displayed volumes rather than a small changing slice; inspect actual bytes/cache reuse before choosing a default.

### Output validity limits the conclusions

Both script and Qt samples were nearly black: sampled RGB values were 0–1, and original frames 120 and 150 were reported blank. Decoded RGB values also stayed near black, so this is not just a text-report issue. The source arrays contain substantial nonzero data and saved contrasts are approximately 0–4,832 / 5,245 / 5,424. The hidden-viewer initialization, texture normalization, source-level binding, clipping geometry and saved camera state need investigation. The precise cause has not been established.

The benchmark's first-frame preview is in `.performance-audit/ortho-profile-no-prefetch-20260925/first-frame.png`. These results identify expensive API boundaries in this environment; they cannot certify performance or visual quality for a correct production render. Do not interpret the much faster repeated export as an optimization delivered here.

## Logging now available

Both `Animation.animate()` and Qt Save export write a readable render log and a machine-readable performance sidecar. For `movie.mp4`, these are `movie.render_log.txt` and `movie.render_perf.json`; for PNG output, they are inside the output folder. `perf_log=False` disables collection and performance files. Qt defaults to enabled, matching the script API.

Measurements include:

- Interpolation, state application, scene preparation/application, ortho layer setup, dims, layer-property and camera updates.
- Screenshot slice synchronization, synchronous materialization when needed, draw preparation, framebuffer capture and viewport restoration.
- Prefetch scheduling, script queue submission waits, Qt queue-full polling/backpressure and data readiness waits, encoder queue idle time, encoding, finalization and output verification.
- Editor restoration; partial reports on script capture/write failure and Qt cancellation/failure after startup.
- Counts of captured/written frames; frame index and thread per event; mean, median, p95, maximum and wall-time percentage per phase; throughput and render settings.
- Output dimensions, sampled pixel range and blank-frame indices in new structured reports. `status=complete` means the export pipeline completed, not that the image passed a visual correctness check.

Timings are inclusive. Child operations, repeated recursive ortho calls and background encoding overlap. Percentages use elapsed wall time and must not be summed. Frame-less events include setup/cleanup; use frame-tagged events to distinguish frame application from restoration. The logger is thread-safe, stops its wall clock before reporting, and retains at most 100,000 events and the latest 10,000 samples per phase; lifetime totals/counts/maxima remain exact. JSON records dropped-event counts. Quantiles describe retained samples if a limit is exceeded.

`capture.framebuffer` measures the complete napari screenshot call, including any draw, upload, synchronization or readback it triggers. It is not a pure GPU execution timer. `capture.draw` measures CPU-side draw preparation. Qt `data.await_ready` also includes event-loop scheduling and signature checks. Background prefetch I/O and asynchronous projection internals are not individually attributed by these export timers.

## Prioritized work

1. **Establish a visually correct reference.** Reproduce first/last/middle frames in a fully initialized native viewer and compare with the saved thumbnails and an independent axis-aligned slab projection. Check source level, transforms, camera framing, shader contrast units and RGBA before encoding. Save lossless frame references and confirm nonblank expected content in each channel. Only then run the entire 181-frame sequence at the user's intended export size. This is the gate for evaluating every optimization below.

2. **Explain and reduce first-capture initialization.** Profile `capture.framebuffer` with CPU call profiling and GPU upload/draw/readback measurements. Compare a fresh viewer, repeated export and an explicit untimed warmup, keeping total load-to-finished-file time visible. Inspect `SceneRenderer.prepare_snapshot()` calling `ensure_layer()` for clip-only objects: it currently prepares full-volume managed plane layers although 3D clip rendering hides them. Prototype lazy placeholders for these unused planes, materializing only on a plane/2D mode transition. Measure HDF5 reads, host memory and GPU allocations; verify that plane editing, mode changes and scene restoration remain correct. Moving initialization earlier without reducing total work is not a throughput improvement.

3. **Reduce steady-state capture work.** First test keeping the export viewport at the target size for the duration of the export, restoring it once at completion/cancel/failure. Current warm CPU draw preparation plus viewport restoration is only about 7 ms/frame, so direct savings from those measured calls are modest; measure any additional GPU benefit before claiming it. Preserve camera scale, fractional device-pixel ratios and exact dimensions. If framebuffer work still dominates, prototype an exact axis-aligned slab ROI path or GPU ray-entry/ray-exit clipping to avoid traversing irrelevant volume. Preserve world position, MIP semantics, sample boundaries and three-channel additive composition. Oblique slabs and MIP/mean/min must retain their tested fallback. Lowering source level is a separate user-selected quality tradeoff, not an equivalent optimization.

4. **Reduce redundant state work and restoration.** Warm frame application is approximately 28–36 ms/frame; scene application alone is around 8–9 ms/frame. Inspect unchanged layer transforms, duplicated scene validation/reconciliation and the restoration path. The measured 0.5–0.6 s restoration is a one-off cost, so its apparent 26–28% share in a seven-frame sample will shrink in a long movie. Cache immutable frame/scene plans or skip verified unchanged work using revision-aware invalidation. Keep transactional state restoration on success, cancellation and writer failure. Avoid caching across source replacement or appearance edits without invalidation.

5. **Tune prefetch only after an A/B data-read audit.** Run depth 0/1/4 with at least three repetitions in alternating order. Count bytes, unique Dask chunks, cache hits, peak RSS and disk activity. Static 3D clip animations may warrant suppressing redundant whole-volume read-ahead while time-varying or 2D animations still benefit. Keep current defaults until the measured end-to-end advantage is repeatable and does not increase memory pressure.

6. **Leave encoder changes lower priority.** The queue currently has ample capacity and the worker is waiting for frames. Revisit hardware encoding, codec presets or larger queues only if validated high-resolution runs show persistent producer backpressure. Keep decode verification timed separately; do not silently remove it to advertise faster rendering.

## Acceptance and repeatability

For each proposed change: compare unchanged source JSON/settings, warmed and fresh-process runs separately, and full-timeline wall time, frame-capture p50/p95/max, load time, memory and quality. Require a reproducible improvement larger than run-to-run noise (a practical initial target is ≥10% median end-to-end improvement across three runs), no new blank frames, exact frame count/dimensions, and unchanged pixels for transformations expected to be exact. Use lossless captures for pixel comparison; H.264 is lossy. Also exercise 2D/3D transitions, multiscale source locks, three-channel appearance, native DPI scaling, slow/failing encoders and cancellation.

Run from the inner repository directory using the parent virtual environment. Use a new output directory for every invocation:

```powershell
../.venv/Scripts/python.exe examples/profile_saved_animation.py ../animation.json .performance-audit/ortho-next --samples 7 --repeat 2
../.venv/Scripts/python.exe examples/profile_saved_animation.py ../animation.json .performance-audit/ortho-next-qt --samples 7 --repeat 2 --qt
# After output correctness is established: full timeline at the desired size.
../.venv/Scripts/python.exe examples/profile_saved_animation.py ../animation.json .performance-audit/ortho-full --samples 0 --repeat 3 --width 1920 --height 1080
```

The harness refuses an existing output directory, never edits the input JSON, records its SHA-256, saves movies/sidecars/metadata and a first-frame preview, and supports `--prefetch 0`, `--no-diagnose` and a Qt cancellation deadline. It now creates a visible native viewer by default; `--hidden` retains the old diagnostic behavior.

Validation of the logging changes: 59 targeted tests passed, covering logger concurrency/quantiles/bounds/disable behavior, partial failure reports, script encoding/restoration, Qt queue responsiveness/cancellation, capture sizing and scene renderer regressions. Four existing non-orthogonal-slicing warnings occurred. These tests include stubbed captures and do not establish the large dataset's visual correctness. Test runtime used workspace-local platformdirs and a test-only mkdir adapter to avoid Windows sandbox ACL failures with pytest's mode-0700 temporary directories.

Machine-readable summary: [ortho-render-performance-results-2026-09-25.json](ortho-render-performance-results-2026-09-25.json). Raw runs: `.performance-audit/ortho-profile-20260925`, `.performance-audit/ortho-profile-qt-20260925`, `.performance-audit/ortho-profile-no-prefetch-20260925`.
