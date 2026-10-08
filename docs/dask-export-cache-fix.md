# Dask finalizer KeyError during animation export

Reported symptom: export stops with an error containing only a quoted
`finalize-hlgfinalizecompute-...` task key.

## Reproduced cause

The installed Dask `Cache` stores task start times and dependency durations
on a shared callback instance. Overlapping rendering and prefetch computations
can use that instance concurrently. When one finishes, `Cache._finish()`
clears timings required by the other; its next `_posttask()` raises
`KeyError` for the unfinished task. A deterministic test pauses an export
finalizer, completes another computation, then resumes the first and reproduces
the reported key format.

Both the animation plugin's cache and napari's own slicing cache used this
implementation. The task key is not an output filename or an FFmpeg error.

## Changes

- Task timings now belong to each scheduler's state, so finishing one
  computation cannot erase another's timings.
- A short lock protects shared cached values, scores, and eviction. Image reads
  and computations remain concurrent.
- Concurrent completion of the same task inserts its result only once,
  preventing cache-size accounting from counting it twice.
- Cache contexts tolerate overlap, and unregistering during a computation does
  not resurrect the retired callback when Dask restores its global callback set.
- The existing napari cache instance is upgraded in memory, retaining its
  contents and budget. Already-created layer closures see the same corrected
  instance. No dependency files or global Dask classes are patched.
- GUI export failures now log the frame number, output path, and traceback,
  rather than leaving only an opaque task key for diagnosis.

## Verification

Tests cover the deterministic legacy failure, the corrected overlap, real
parallel Dask arrays with eviction, cached task reuse, failed computations,
context lifetime, and existing napari cache references.

A native GUI export also reads chunked, compressed HDF5 data while four
prefetch workers run, writes an MP4, and decodes it to verify the expected
frame count and distinct endpoint images.

Full native Windows Qt/OpenGL suite: **498 passed** (65.16 seconds).

Restart napari before retrying an interrupted export so all workers and cache
callbacks use the corrected implementation. This fix does not resume a
partially written animation.
