# Ortho slice animation: 1280 × 720 implementation and results

The first three actions in the [performance plan](ortho-render-performance-plan-2026-09-25.md) are complete for `../animation.json`: establish a valid image, reduce unnecessary initialization, and keep the export viewport at the requested size. The production output is [animation.mp4](../outputs/ortho-720p/animation.mp4), with lossless [frame 0](../outputs/ortho-720p/frame-0.png), [frame 90](../outputs/ortho-720p/frame-90.png), and [frame 180](../outputs/ortho-720p/frame-180.png). The input was not modified; its SHA-256 is `6bf34903553dbbf76639609be0ef4f8d16475e970d6610611cb435bf6b81a961`.

## Correct image reference

The initial almost-black sample arose from an uninitialized hidden Qt viewer. The benchmark now uses a visible native viewer unless `--hidden` is requested. A second problem was exposed when checking the actual source data: this version of napari stores clipping planes in layer-data coordinates, while its Vispy volume visual expects scene coordinates. With a source z scale of 0.7, the 25 µm slab was rendered at the wrong position. `native_clipping.py` now converts the plane positions and inverse-transpose normals at the visual boundary and resynchronizes them when the clipping model or layer transform changes. The authored model coordinates remain unchanged.

An independent HDF5/NumPy axis-aligned slab MIP calculation for frames 0, 90 and 180 shows content in all three channels; the corrected native frames were visually checked against those projections. The oracle files are in `.performance-audit/720p-work/oracle-*`. The oracle is discrete voxel MIP, while the native GPU renderer samples continuous rays, so it is a content and geometry check, not a pixel-equality target. The full export has 181 frames at 1280 × 720, no reported blank frames, and decoded RGB spans 0–255. Frame 180 is mostly dark because the authored slab has reached z = 0, where source intensity is low.

## Changes and measurement

Clip-only animations now create tiny hidden placeholders for their unused managed plane layers. The full backing volume is materialized if the section switches to plane or 2D mode. `ExportCaptureSession` keeps the native canvas at 1280 × 720 between captures and restores the editor size on completion, failure, or cancellation. It rescales each frame's authored camera zoom against the original canvas; fractional device-pixel ratios retain exact output dimensions. Both the script exporter and Qt Save controller use the session. Render logs separate viewport resize, camera scaling, framebuffer capture, per-frame restore, and final session restore.

The reference used the old eager plane layers and per-frame viewport resizing through benchmark-only switches. Both reference and optimized runs used the same clipping-coordinate correction, source, display, encoder, and target size. They ran in separate visible viewer processes on Windows 11, Python 3.13.14, napari 0.8.0, PyQt5, and an NVIDIA GeForce RTX 3070. Each process loaded the three 491 × 2074 × 2073 uint16 HDF5 volumes once and exported the entire 181-frame, 20-fps sequence three times. Export time excludes load/ready time and includes encoding, verification, and editor restoration.

| Variant | Full export runs | Median | Time to ready | Resident memory after ready |
|---|---:|---:|---:|---:|
| Corrected reference | 51.301 / 51.686 / 51.546 s | 51.546 s | 32.168 s | 70.55 GiB |
| Optimized script | 45.653 / 45.349 / 44.041 s | 45.349 s | 21.582 s | 51.22 GiB |
| Optimized Qt Save | 45.325 s | — | 20.533 s | 51.51 GiB |

The script median is **6.197 s (12.0%) faster**. Ready time fell by 10.59 s and measured resident memory after readiness by 19.33 GiB in this paired run. Those load and memory differences are single-process observations and may include OS caching or allocator effects; repeat them in alternating fresh processes before treating their exact magnitudes as general. The three reference runs, three optimized runs, and Qt Save run produced the same MP4 bytes (SHA-256 `47c4e114cb018d3a15cc26f8d249357fa867672acb9737557d6ad247830a7113`). Unencoded frames 0, 90, and 180 were also byte-identical across paths and repetitions.

The reference spent about 26.6 s resizing the viewport and 2.1 s restoring it per export. The optimized session spent about 0.15 s on the one resize and 0.17 s in per-frame camera restoration; the final viewport restoration took about 0.01 s. Framebuffer calls became more expensive with the larger canvas retained (about 14 s total in the reference versus 36–37 s optimized), so the sum of capture phases, rather than one subphase, explains the net speedup. Frame application remained near 5.8–6.2 s per export. Encoding took roughly 1.9–2.2 s of overlapping worker time; this remains a capture-bound workload. Phase timings are inclusive and overlap across nested scopes and threads.

## Cropped-slab prototype and remaining work

Framebuffer work still dominates. A diagnostic-only exact-voxel crop was tested at frames 0, 90, and 180. Its warm capture calls were approximately 29–65 ms, but its output differed from the full-volume ray result at 9,521–82,040 pixels, with a maximum channel error of 5–23. The likely cause is the native ray sampling lattice changing when the volume bounds change. The measurements and images are in `.performance-audit/720p-work/roi-prototype.json` and `roi-*.png`. Cropping remains **disabled** in production because it does not preserve exact pixels. A future GPU ray-entry/ray-exit clip that retains the original volume bounds is worth investigating; it needs pixel and transform checks across oblique slabs and all projections before adoption.

The state-application/reconciliation audit, alternating prefetch-depth and data-read audit, and encoder assessment are now complete in [ortho-render-performance-followup-720p.md](ortho-render-performance-followup-720p.md). Quality reduction and encoder changes are not indicated by these measurements.

## Reproduce and validation

From the inner repository directory, with the parent virtual environment:

```powershell
../.venv/Scripts/python.exe benchmarks/profile_saved_animation.py ../animation.json .performance-audit/new-reference --samples 0 --repeat 3 --width 1280 --height 720 --lossless --profile --legacy-viewport --eager-clip-planes
../.venv/Scripts/python.exe benchmarks/profile_saved_animation.py ../animation.json .performance-audit/new-optimized --samples 0 --repeat 3 --width 1280 --height 720 --lossless --profile
../.venv/Scripts/python.exe benchmarks/profile_saved_animation.py ../animation.json .performance-audit/new-qt --samples 0 --repeat 1 --width 1280 --height 720 --lossless --qt --timeout 300
```

Each output directory must be new. Raw benchmarks and event-level logs are in `.performance-audit/720p-correct-reference`, `720p-correct-optimized`, and `720p-correct-qt`. A curated performance log and JSON sidecar accompany the video in `outputs/ortho-720p`.

The broad relevant suite passed 130 tests after the clipping fix; a final focused suite including the two new adapter tests passed 73 tests. Nine native capture/adapter tests also passed at `QT_SCALE_FACTOR=1.75`. These cover clip transforms, source/mode transitions, exact canvas dimensions, identical native pixels, fractional scaling, and restoration after cancellation. The native Qt backend emitted its existing PyQt5 deprecation/theme warnings.
