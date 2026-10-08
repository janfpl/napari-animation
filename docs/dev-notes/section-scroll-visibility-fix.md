# 2D section flicker while scrolling

The recording `2026-09-23 15-40-49.mp4` shows the managed 2D ortho
section disappearing temporarily while the built-in napari plane slider moves,
revealing other channels underneath.

## Cause

`OrthoSlice.apply()` unconditionally set the projection layer's visibility to
false on every scene update. The slider triggers that update, and the
asynchronous projection callback later set visibility back to true. Even a
cached projection emitted an unnecessary false/true event pair.

The projection is also represented as a volume with one voxel along its hidden
axis. Leaving it at the previous plane position would allow napari to slice it
away while a replacement was pending, even with visibility kept true.

## Fix

An enabled, active section keeps its last completed pixels visible while the
replacement loads. Its singleton plane follows the slider position; the
background result then replaces the pixels. Existing revision checks reject
obsolete completions without moving the slider backward.

The section still hides when disabled, inactive, outside its source extent, or
in 3D mode. A preview from a different source or orientation is not reused.
This changes the pending preview only; the completed projection and exported
pixels retain the requested plane and resolution.

## Regression checks

Before the fix, five new regression cases failed: delayed reads for all three
axes, cached scrolling, and the actual native Qt slider operation. All nine
new cases pass after the fix, including the four legitimate hide transitions.
The delayed-read checks assert both visibility and nonempty displayed pixels,
then compare the completed image against the projection reference.

Full native Windows Qt/OpenGL suite: **489 passed** (64.17 seconds).
