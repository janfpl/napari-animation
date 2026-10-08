# Performance review fixes — 2026-09-23

All seven findings from the code review are fixed in the source tree.

| Finding | Change | Regression coverage |
| --- | --- | --- |
| Cached keyframe switches can hide both sections | The renderer updates the active section before any snapshot projection can publish. | Repeated A → B → A navigation with cached projections checks visibility. |
| Script exports screenshot before a section finishes | `Animation.animate()` and `ViewerState.render()` use an exact, synchronous section-render context. It cancels obsolete interactive work and restores the previous processing mode even on errors. | Delayed multiscale reads, exact output pixels, export frame order, and exception cleanup. |
| Interactive edits can enter GUI exports | Export freezes the keyframe timeline, pauses native slice playback, and shows a cancellable window-modal progress dialog. Before capture it detects intervening display changes, reapplies the intended frame and waits again. Adding, removing, reordering or replacing source layers stops export with an error. | Slider/camera/timing edits between apply and capture; source additions/removals/replacement; modal progress and cancellation. |
| Delayed refinement can revive a retired section | A section application has a revision independent of worker requests. Replacement, disable, removal, inactive selection, empty scenes, and shutdown invalidate earlier refinement/publication callbacks. | Disabled/removed/empty/closed views and switching from a pending preview to a cached exact snapshot. |
| Scroll-through leaves the active section stationary | Generated endpoints update the matching active 2D section coordinate as well as `dims.point`. | First, middle, and final frame coordinates and projected pixels. |
| HDF5 bypass skips a custom getter | Both direct HDF5 graphs and the intermediate reader layer of recognized Imaris graphs must use Dask's unchanged getter. Custom or unknown getters retain the original Dask path. | A custom getter that doubles voxel values; unchanged direct and singleton Imaris graphs; external Luxendo links. |
| Boolean means lose fractions | Mean projections allocate float64 output for Boolean input, matching NumPy. | Exact NumPy comparisons on all three axes. |

## Validation

- **462 tests passed** in the complete native Windows Qt/OpenGL suite, in 58.99 seconds. This includes 19 new regression cases. Warnings concern Qt5 theme detection and napari's existing non-orthogonal slicing limitation.
- The new real OpenGL check compares frames produced by `QtExportController` against exact synchronous screenshots of the same multiscale section frames, before encoding. The arrays match exactly.
- The production HDF5 benchmark passed on `single tile.ims`, the supplied `.lux.h5` file, and the externally linked `main_st-1.ims` layout. Compressed IMS projections retained four worker processes; uncompressed Luxendo retained direct reads. No backend failures or surviving workers were reported.
- The largest measured Qt heartbeat gap during these HDF5 worker checks was **17.58 ms**. These are worker/event-loop measurements, not live-reader GPU paint or input-latency measurements.

Raw real-file results: [performance-review-fixes-h5.json](performance-review-fixes-h5.json).

The remaining performance-plan items in [performance-implementation-status.md](performance-implementation-status.md) still apply. This correctness pass does not enable or claim CUDA acceleration.
