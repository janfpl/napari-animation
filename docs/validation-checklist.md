# Validation checklist

Acceptance tests for everything added on `claude/3d-animation-capabilities-3phx02`.

The automated suite covers the model and widget logic, but it runs against
`ViewerModel` and cannot see a pixel. Anything involving actual rendering, GPU
memory, or real data has to be checked by eye on a machine with a GPU. That is
what most of this list is for.

Each check states what to do and what counts as a pass. Work top-down: a
failure early on will cause confusing failures later.

---

## 0. Environment

| # | Check | Pass |
|---|---|---|
| 0.1 | `python -c "import napari; print(napari.__version__)"` | ≥ 0.5, ideally 0.8.x |
| 0.2 | `python -c "import napari_animation; print(napari_animation.__file__)"` | points at your clone |
| 0.3 | `napari` opens, **Plugins → napari-animation → Wizard** | panel appears with Voxel size and 3D scene sections |
| 0.4 | `python -c "from napari_threedee.manipulators import RenderPlaneManipulator"` | imports without error (Python 3.10 env) |

If 0.4 fails you are on Python ≥ 3.11 — napari-threedee pins `zarr<3`, which
pulls in `asciitree`, which does not build there. Everything except check 6
works regardless.

## 1. Automated suite

```bash
python -m pip install -e ".[testing]"
python -m pytest napari_animation -q
```

**Pass:** no failures. On a GPU machine the full suite runs; expect ~180 tests.

If you are on a headless box, the viewer/screenshot tests will fail or abort on
the missing OpenGL context. Those are environmental. The model and widget tests
run without a GPU:

```bash
python -m pytest napari_animation/_tests napari_animation/_qt/_tests/test_scene_widget.py \
                 napari_animation/_qt/_tests/test_savedialog_widget.py -q
```

**Pass:** 154 passed.

---

## 2. Voxel size

Load the TIFF whose Z spacing was wrong.

| # | Check | Pass |
|---|---|---|
| 2.1 | Voxel size panel shows the layer's current spacing | matches what napari reports for `layer.scale` |
| 2.2 | Enter the correct Z/Y/X, choose units, **Apply voxel size** | the volume's proportions visibly correct in 3D |
| 2.3 | With multiple channels loaded and "Apply to all image layers" ticked | every channel gets the same spacing |
| 2.4 | Untick it, change one layer | only that layer changes |
| 2.5 | Capture 2 keyframes, *then* change voxel size, then click keyframe 1 | spacing stays corrected — **does not** revert |
| 2.6 | Try to enter 0 for an axis | rejected (`ValueError`), not silently applied |

2.5 is the one that matters most. `scale` is part of captured keyframe state,
so without the rewrite an old keyframe would restore the wrong spacing partway
through a render — and you would only notice in the finished movie.

## 3. Clipping planes

| # | Check | Pass |
|---|---|---|
| 3.1 | **+ Clipping plane** | volume is cut; plane appears in the list |
| 3.2 | Drag the Position Z spin box | the cut sweeps smoothly through the volume |
| 3.3 | **Flip side** | the other half of the volume is hidden |
| 3.4 | Orientation preset ±Z/±Y/±X | cut re-orients to that axis |
| 3.5 | Add 4 planes at different positions/orientations | **all four cut simultaneously** |
| 3.6 | Set "Cuts" to one layer, with 2+ layers loaded | only that layer is cut |
| 3.7 | Untick a plane in the list | its cut disappears; others remain |
| 3.8 | Set a custom normal like (1, 1, 0) | cut is oblique, and *perpendicular to the direction you asked for* even with anisotropic voxels |

3.5 is the headline requirement — before this branch, a second plane replaced
the first. 3.8 checks the normal transform: an incorrect implementation gives a
plane visibly skewed away from 45° when Z spacing differs from XY.

## 4. Ortho slices

| # | Check | Pass |
|---|---|---|
| 4.1 | **+ Ortho slice** | a slab appears; a new layer shows in napari's layer list |
| 4.2 | Untick the layer in **napari's own** layer list | slab hides — it is a real layer |
| 4.3 | Drag Centre Z | slab scrolls through the volume |
| 4.4 | Increase Thickness | slab visibly thickens; a max projection through more material |
| 4.5 | Projection max → mean → min | rendering changes accordingly |
| 4.6 | View preset XY / XZ / YZ | slab re-orients |
| 4.7 | Set a custom normal | oblique slab |
| 4.8 | Render mode → `clip` | the slab layer hides and the *source volume* is cut to a slab instead |
| 4.9 | Back to `plane` | slab layer returns |
| 4.10 | Thickness in world units against a known feature | a 10 µm slab really spans 10 µm — check against your voxel size |
| 4.11 | Rename the slice in the list | the backing napari layer is renamed too |
| 4.12 | Add an ortho slice *and* clipping planes | both visible at once; neither erases the other |

4.10 is worth doing carefully with anisotropic voxels: thickness is measured
along the slab normal in world units, so an XY slab of 10 µm at 2 µm Z spacing
should be 5 planes thick, not 10.

## 5. Per-keyframe on/off

| # | Check | Pass |
|---|---|---|
| 5.1 | Add a clipping plane, capture keyframe 1 | — |
| 5.2 | Untick the plane, capture keyframe 2 | — |
| 5.3 | Click keyframe 1, then keyframe 2 | cut present, then absent |
| 5.4 | Scrub the animation slider across the transition | the cut switches **once, cleanly** — no flicker, no half-state |
| 5.5 | Same with an ortho slice | slab appears/disappears at the keyframe |

The switch should land on the destination keyframe, not at the start of the
transition.

## 6. napari-threedee manipulator *(Python 3.10 env only)*

| # | Check | Pass |
|---|---|---|
| 6.1 | Select an ortho slice; **Edit in canvas** is enabled | not greyed out |
| 6.2 | Click it | a manipulator gizmo appears on the slab in the 3D canvas |
| 6.3 | Drag the translation handle | slab moves |
| 6.4 | Drag a rotation ring | slab tilts |
| 6.5 | Capture a keyframe after dragging | the *dragged* position is recorded, not the old spin-box value |
| 6.6 | Re-select the slice in the scene list | spin boxes show the dragged geometry |

6.5 is the integration point: the backing layer is the source of truth, so an
external tool moving the plane must be what gets captured.

## 7. Interpolation quality

| # | Check | Pass |
|---|---|---|
| 7.1 | Keyframe 1: plane at one end. Keyframe 2: plane at the other, 60 steps. Scrub | the cut **sweeps smoothly**; it does not jump at the start |
| 7.2 | Same for an ortho slice centre | slab scrolls smoothly |
| 7.3 | Keyframe a plane normal from (1,0,0) to (0,1,0) | plane **rotates**; it does not shrink or pass through the centre |
| 7.4 | Two keyframes with different numbers of enabled planes | no plane vanishes mid-transition |
| 7.5 | Keyframe 1 in 3D, keyframe 2 in 2D | ndisplay switches once, at the destination |
| 7.6 | 3D keyframe → 2D keyframe → 3D keyframe with distinct camera angles | each 3D keyframe's camera is preserved, not reset |

7.6 is the regression that motivated the apply-ordering fix. A failure looks
like the camera snapping to a default view after every mode change.

## 8. Persistence

| # | Check | Pass |
|---|---|---|
| 8.1 | Build an animation with planes and slices, **Save Keyframes** | writes a `.json` |
| 8.2 | Restart napari, load the data, **Load Keyframes** | keyframes restore |
| 8.3 | Scrub | planes and slices are back in the right places |
| 8.4 | Load a keyframe file saved *before* this branch | loads without error (scene is simply absent) |

## 9. Render output

| # | Check | Pass |
|---|---|---|
| 9.1 | Save Animation → `.mp4` | log ends `Saved animation to ...` |
| 9.2 | Log line `Encoder (compatibility mode): ...` | present, showing `-profile:v high` |
| 9.3 | Diagnostics `Stream #0:0` line | reads `h264 (High)` and `yuv420p` — **not** any 4:4:4 variant |
| 9.4 | Open the file in Windows Media Player / Photos | plays |
| 9.5 | Diagnostics `read back N frame(s)` | N equals the frames rendered |
| 9.6 | No `WARNING` lines in diagnostics | — |
| 9.7 | Resize the napari window so canvas dims are multiples of 16, re-render | no `was encoded as` note; no black strip |
| 9.8 | Deliberately resize the window *while* rendering | diagnostics report `FRAME SIZE CHANGED MID-RENDER` |
| 9.9 | Render with two identical keyframes | diagnostics report `EVERY FRAME IS IDENTICAL` |
| 9.10 | Cancel the save dialog | nothing happens — no traceback |
| 9.11 | Save to a path with no extension | loud warning that a PNG folder is being written |

9.8 and 9.9 are checking the *diagnostics* work, so deliberately break things.

## 10. Performance

```bash
python examples/benchmark_scene.py --real --shape 64 512 512
```

Substitute your own volume shape. Expected shape of the results:

| Measure | Expected |
|---|---|
| Clipping plane, per-frame apply cost | well under 0.1 ms each — essentially free |
| Ortho slice, per-frame apply cost | a few ms each |
| Ortho slice VRAM | one extra copy of the volume per plane-mode slice |
| Render throughput | dominated by `screenshot`, not by scene work |

For reference, headless on a small volume: clipping planes ~0.03 ms each,
ortho slices ~1.8 ms each.

The point of this benchmark is that clipping planes and plane depiction are
evaluated on the GPU against an already-uploaded texture, so sweeping them
across hundreds of frames should trigger no extra data reads. If your
per-frame cost grows with the number of planes in any significant way, or the
`apply` phase of the render log balloons when you add scene objects, something
is wrong.

### The real constraint: VRAM

Each `plane`-mode ortho slice is a separate napari layer and therefore a
separate GPU texture of the same volume. Three slices on a 500³ float32 volume
is roughly 3 × 500 MB **on top of** the source volume.

| # | Check | Pass |
|---|---|---|
| 10.1 | Add slices one at a time on your real data, watching GPU memory | grows by about one volume per slice |
| 10.2 | At the point it no longer fits, switch a slice to `clip` mode | memory drops back; the slab still renders |
| 10.3 | Render a long animation with slices active | no slowdown over time, no out-of-memory |

If VRAM is tight on whole-brain data, the mitigations are `clip` mode, or
pointing slices at a downsampled copy of the volume. A pyramid-level selector
for the latter is specified in the plan but not yet built.

### Render log baseline

Keep a `.render_log.txt` from a known-good run. `apply` and `screenshot` should
dominate; `interpolate` should be ~0 and `encode/write` a few percent. A sudden
jump in `apply` after adding scene objects is the signal worth watching.

---

## Known gaps

Not yet implemented, so do not test for them:

* 2D/3D snap buttons and scroll-through helpers (Phase 3). The *state* handling
  works — check 7.5 and 7.6 — but there is no button for it.
* The keyframe × object toggle matrix (Phase 4). Per-keyframe toggling works
  via capture; there is no grid view.
* Pyramid-level selection and a VRAM readout on the ortho slice panel.
* The legacy "Ortho slicer" group box still exists alongside the new 3D scene
  panel. Two overlapping concepts in one UI; the old one is slated to fold into
  the new.
