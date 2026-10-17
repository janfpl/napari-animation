# 720p ortho render: state, prefetch, and encoder follow-up

This continues [the first three 1280 × 720 improvements](ortho-render-performance-720p.md) on the unchanged `../animation.json` (SHA-256 `6bf34903553dbbf76639609be0ef4f8d16475e970d6610611cb435bf6b81a961`). The workload is 181 frames from three 491 × 2074 × 2073 uint16 Luxendo/HDF5 channels, with an axis-aligned moving 3D clipping slab. All timings are complete, encoded, verified exports at 20 fps. The process read-byte and sampled peak-RSS figures below come from `psutil`; read bytes include process activity beyond HDF5, so their *difference* is the useful measure.

## Repeated state work

Frame application already avoided assigning unchanged layer properties. The remaining repeated work included a bulk dims update on every frame and a second scene preflight/object reconstruction after the same frame had been validated and its managed layers prepared. `ViewerState.apply()` now updates only changed dims and camera fields and passes its same-frame prepared scene to `SceneRenderer.apply_snapshot()`. This does not cache across frames or viewer edits. The renderer still reconciles managed layers and recomputes clipping planes for every frame; it retains the original transactional restoration on completion, failure, and cancellation.

In the earlier optimized run, the first full export spent 0.833 s in `state.dims` and 5.877 s in the inclusive `apply` phase. With this change, the first full export in the alternating prefetch matrix spent 0.020 s and 4.845 s respectively. The later phase also includes other work and varies by run; the clear causal saving is the skipped no-op dims update. Restoration was already about 0.03 s per full export, so it does not warrant a separate optimization.

## Prefetch read audit

The saved movie displays all three axes of each volume. Before the change, every future-frame prefetch key was a full-array slice of all three Dask sources. All 181 frames shared **one** read signature. The three logical arrays total 12.67 GB, with 1,156 Dask chunks per channel (3,468 total). Napari had already materialized the full arrays for the visible 3D view. The shared interactive Dask cache had a roughly 0.5 GB budget, far below one complete three-channel read.

Nine full exports were run in one visible viewer in alternating order `4, 0, 1, 1, 4, 0, 0, 1, 4`, where the number is prefetch depth. This isolates much of the viewer/load cost but does not eliminate time and GPU-frequency drift. These runs used the state-work improvement above and the old full-volume prefetch behavior.

| Prefetch depth | Full-export times | Median | Process read bytes per export | Sampled peak RSS |
|---|---:|---:|---:|---:|
| 0 | 42.592 / 39.603 / 39.821 s | 39.821 s | 0.50 GB | 50.27–52.67 GiB |
| 1 | 43.869 / 43.287 / 41.163 s | 43.287 s | 14.65–14.79 GB | 56.72–60.16 GiB |
| 4 | 44.495 / 43.055 / 40.871 s | 43.055 s | 14.19–14.77 GB | 56.39–59.70 GiB |

Depth 0 was faster in each comparable part of this sequence and avoided roughly 14 GB of process reads. A separate depth-4 cache-instrumented full export recorded 210 cache hits and 3,261 distinct inserted task keys, alongside 14.17 GB of process reads and 60.15 GiB sampled peak RSS. The small cache could not retain the whole-volume working set.

The prefetcher now excludes a fully materialized volume from read-ahead when all its axes are displayed. If every source is already resident, all frames stay in 3D, and locked multiscale levels stay fixed, it skips the scheduler entirely. That proof reads the saved keyframes for these discrete fields; it does not materialize every interpolated frame before the export. Mixed 2D/3D or changing-slice animations retain their existing prefetch behavior and default depth 4. Render sidecars now report `prefetch_enabled` and a skip reason. A benchmark-only `--eager-prefetch-full-volume` switch reproduces the old behavior. This choice avoids changing the default for workloads that may benefit from read-ahead.

## Encoder assessment

The script's producer waited about 0.003 s total to enqueue 181 frames, while encoding used about 1.9–2.2 s of overlapping worker time. In the Qt Save run, the writer spent 42.4 s waiting for frames. Frame capture remains the limiting stage, so no codec, queue-size, or hardware-encoder change was made. The current H.264 file remains deterministic and playable; output verification stays enabled.

## Final validation and reproducibility

The nine alternating-run MP4 files are byte-identical to the corrected reference, SHA-256 `47c4e114cb018d3a15cc26f8d249357fa867672acb9737557d6ad247830a7113`. The raw matrix, process-I/O samples, and logs are in `.performance-audit/720p-state-prefetch-matrix`; the cache audit is in `.performance-audit/720p-prefetch-cache-audit`. Reproduce the matrix with:

```powershell
../.venv/Scripts/python.exe benchmarks/profile_saved_animation.py ../animation.json .performance-audit/new-matrix --samples 0 --width 1280 --height 720 --prefetch-sequence 4,0,1,1,4,0,0,1,4 --eager-prefetch-full-volume --lossless
```

With the state and resident-volume changes in place, a separate fresh visible viewer rendered the full movie three times at the unchanged default depth 4 in **42.081 / 42.189 / 41.504 s**, median **42.081 s**. The requested depth was automatically skipped because all three full 3D sources were resident; the scheduled-prefetch phase fell from about 0.82 s to effectively zero. Each run read about 0.50 GB at the process level and reached 50.94–51.81 GiB sampled peak RSS. The earlier optimized median was 45.349 s, so the new median is **3.268 s (7.2%) faster**; compared with the original corrected eager-layer/per-frame-viewport reference at 51.546 s, the combined improvement is **9.465 s (18.4%)**. These are process-to-process comparisons and include GPU timing variation. All three final movies match the reference byte-for-byte; their raw reports are in `.performance-audit/720p-final-keyframe-prefetch-fixed`.

The Qt Save path then completed the same 181 frames in **43.054 s**, reported no blank frames, and automatically skipped resident-volume prefetch. Its MP4 and unencoded frames 0, 90, and 180 match the script export byte-for-byte. Its writer spent 40.76 s waiting for frames while encoding took 1.61 s, confirming that encoding remains lower priority. Its raw report is in `.performance-audit/720p-final-qt`.

Validation passed **561 tests** in the repository's main/Qt suite and **9 native capture/clip tests** in a separate clean process; the latter also passed at `QT_SCALE_FACTOR=1.75`. Running every test in one process triggers an unrelated napari SVG-command double-registration during setup of those seven native capture cases, so the split run is the reliable suite configuration on this environment. The full movie, Qt Save movie, and sampled unencoded frames retain exact pixel equality with the corrected reference. Two existing test fixtures were updated to expect the encoder's 16-pixel padding and the projection service's `on_error` keyword.

The delivered [movie](../outputs/ortho-720p/animation.mp4) and [performance sidecar](../outputs/ortho-720p/animation.render_perf.json) come from a final single-run export with the completed logging code. It took 42.378 s, read 0.50 GB at the process level, reported `prefetch_enabled: false` with reason `resident_full_volume`, and passed the 181-frame/nonblank output checks. It has the same SHA-256 and sampled lossless pixels as every corrected reference and final run. Its raw benchmark is in `.performance-audit/720p-final-release`.
