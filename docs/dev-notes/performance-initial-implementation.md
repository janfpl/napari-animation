Performance implementation, first pass — 2026-09-22

This pass addresses repeated work in the four controls the user identified. It also tests whether a parallel CPU projection is useful on their IMS/Luxendo data. Source changes are in `napari_animation/animation.py`, `scene/ortho_slice.py`, and `prefetch.py`; the optional prototype is `numba_projection.py`.

**Changes applied**

* A managed section now retains its bound source array and normalized transforms when unchanged. In particular, a Dask timepoint view may be a new Python object on every access, so the binding also records the base array, level and leading indices. Existing `Image.data` reassignment caused a new slice request and data event for every scene application.
* Keyframe navigation guards reentrant and duplicate selection notifications. The selected snapshot is applied once. Restored live section objects regain their managed-layer listeners and last-written plane state without a second scene render.
* Axis-aligned 2D projection selects a basic array slice before materializing it. This prevents accidental full-array conversion by array-protocol readers when `np.take` is used. It also avoids allocating a NumPy index vector for every slab.
* Prefetch recognizes napari 0.8 `MultiScaleData`, so it selects a pyramid array rather than treating the container as a NumPy array. It skips managed derivative layers; it does not skip source layers hidden by the 2D section compositor.
* An optional, import-on-demand Numba kernel projects max/min/mean slabs across independent output pixels. It is intentionally not enabled in the GUI based on the measurements below.

**Real datasets and measurement scope**

The benchmarks use `D:\081926\single tile.ims` (27,466,337,879 bytes; 6 resolution levels, 3 channels), Embryo8b's `main_st-1.ims` external-link header, its processed `uni_tp-0_ch-0_st-1_obj-left_cam-left_etc.lux.h5` (5,042,123,360 bytes), and a raw `Cam_left_00000.lux.h5` (4,295,501,120 bytes). The link from the Embryo8b IMS header resolves successfully to the Luxendo datasets.

The controller benchmark uses one 256³ IMS pyramid array (level 3) and one 128×512×512 Embryo8b array (level 2). Screenshots return a small stub and the viewer is headless, so timings measure Python/controller, HDF5/Dask, and section projection, not the Qt event loop or OpenGL rendering. The processed Luxendo level 2 was also exercised directly after the changes. Each action runs once per case; these figures are directional, not a statistically sampled latency distribution. File-system caches may differ between runs. Baseline and final JSON files preserve the full call counts.

| Dataset and action | Before | After | Reduction |
|---|---:|---:|---:|
| Single tile IMS, Snap XY | 0.152 s | 0.120 s | 21% |
| Single tile IMS, Add scroll-through | 0.794 s | 0.317 s | 60% |
| Single tile IMS, Add slice-sweep | 0.959 s | 0.323 s | 66% |
| Single tile IMS, select keyframe | 0.323 s | 0.069 s | 79% |
| Embryo8b IMS header, Add scroll-through | 0.634 s | 0.231 s | 64% |
| Embryo8b IMS header, Add slice-sweep | 0.708 s | 0.241 s | 66% |
| Embryo8b IMS header, select keyframe | 0.254 s | 0.057 s | 78% |

The Embryo8b direct Luxendo level 2 measured 0.112 s for Snap XY, 0.232 s for Add scroll-through, 0.240 s for Add slice-sweep, and 0.059 s for keyframe selection after the changes. No separate pre-change direct Luxendo controller run was recorded, so these are not reported as speedups.

For the single tile case, scroll-through and slice-sweep previously invoked 8 and 10 2D projections, respectively. Both now invoke 4. Selecting a keyframe previously invoked 4 and now invokes 1. The same call-count reduction was observed for the Embryo8b IMS-header case. These reductions are a stronger explanation of the timing gains than percentage comparisons alone.

The full-resolution I/O probe deliberately read a 4×64×64 voxel ROI from each dataset. It establishes that the actual native files and external links are readable; that tiny ROI is not representative of a full displayed plane. The separate bounded slab benchmark read 32×512×512 uint16 voxels from the single tile level 2 in 1.05 s and 16×1024×1024 voxels from the Embryo8b IMS/Luxendo level 1 in 2.09 s. A 2D projection can incur a read of this scale, so the remaining synchronous I/O is the main next target.

**Numba result**

The Numba kernel passed numerical parity checks against NumPy for axes 0/1/2, max/min/mean and uint16/float32 data, including NaN propagation. It used eight CPU threads in the benchmark. First warm-up took 2.69 s in a fresh process. After warm-up, on the two real slabs above:

| Computation | Single tile: NumPy → Numba | Embryo8b: NumPy → Numba |
|---|---:|---:|
| max | 0.50 → 2.95 ms | 1.53 → 5.37 ms |
| min | 0.42 → 2.90 ms | 1.36 → 5.12 ms |
| mean | 4.41 → 2.86 ms | 10.37 → 5.45 ms |

Numba helped mean reduction but slowed max/min. The 1–2 s HDF5 read dominates all three compute times. Enabling the kernel in the current GUI path would add a first-use compilation pause and would barely change end-to-end latency. CUDA would also need data transfer and synchronization for these already fast reductions. A GPU path should be reconsidered only after read/projection requests run off the GUI thread or data remains resident on the device for repeated work.

**Validation**

`docs/performance-final-tests.txt` records 121 passing headless tests across scroll-through, prefetch, scene rendering, ortho slices, keyframe behavior and Numba numerical parity. Ten tests were deselected: nine pixel tests need a real OpenGL context that the offscreen Qt platform cannot create, and one file-rollback test uses pytest's temporary directory path, which this sandbox denied. The earlier broad test attempt failed in GUI/OpenGL setup for the same environment reason; its output was discarded after the narrowed relevant suite passed. These are validation gaps, not verified application regressions.

Benchmark commands from the outer workspace directory:

```
.\.venv\Scripts\python.exe .\napari-animation-claude-3d-animation-capabilities-3phx02\examples\benchmark_h5_interactions.py
.\.venv\Scripts\python.exe .\napari-animation-claude-3d-animation-capabilities-3phx02\examples\benchmark_projection_backends.py
```

**Next implementation step**

The GUI still waits for `_apply_2d()` to read and project each requested slab. The next phase should make this an immutable, bounded worker request with a generation ID, keep the previous completed image visible while loading, and publish only the latest completed result on the Qt thread. The request planner needs exact level/time/channel, interval, affine, source revision and projection identity. Capture/export must wait for their own exact result before taking a screenshot; preview may discard obsolete requests. Benchmark this with a real Qt event-loop heartbeat and all three orthogonal views on the user's high-resolution IMS/Luxendo data before enabling any further CPU or CUDA backend.
