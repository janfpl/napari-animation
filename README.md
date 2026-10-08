# napari-animation (WIP under active development)

[![License](https://img.shields.io/pypi/l/napari-animation.svg?color=green)](https://github.com/napari/napari-animation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-animation.svg?color=green)](https://pypi.org/project/napari-animation)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-animation.svg?color=green)](https://python.org)
[![tests](https://github.com/napari/napari-animation/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/napari/napari-animation/actions)
[![codecov](https://codecov.io/gh/napari/napari-animation/branch/main/graph/badge.svg)](https://codecov.io/gh/napari/napari-animation)

**napari-animation** is a plugin for making animations in [napari].

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

It is built off of great work from @guiwitz in [naparimovie](https://github.com/guiwitz/naparimovie) which was initially submitted to napari in [PR#851](https://github.com/napari/napari/pull/780).

----------------------------------
## Overview

**napari-animation** provides a framework for the creation of animations in napari and features:
- an easy to use GUI for interactive creation of animations
- Python tools for programmatic creation of animations

This plugin is currently pre-release and under active development. APIs are likely to change before it's first 0.0.1 release,
but feedback and contributions are welcome.

## Installation

Requires Python 3.11 or newer and napari 0.8 or newer with a Qt backend.
Clone this repository and install it locally with

    pip install "napari[pyqt5]"
    pip install -e .

See [docs/installation.md](docs/installation.md) for details, and
[docs/user-guide.md](docs/user-guide.md) for the 3D scene objects (clipping
planes, ortho slices, scroll-throughs).

## Examples
Examples can be found in our [examples](examples) folder. Simple examples for both interactive and headless 
use of the plugin follow.

### Interactive
**napari-animation** can be used interactively by creating an `AnimationWidget` from a napari `Viewer` and adding it to
the viewer as a dock widget.

```python
from napari_animation import AnimationWidget

animation_widget = AnimationWidget(viewer)
viewer.window.add_dock_widget(animation_widget, area='right')
```

![AnimationWidget image](resources/screenshot-animation-widget.png)

### Headless
**napari-animation** can also be run headless, allowing for reproducible, scripted creation of animations.

```python
from napari_animation import Animation

animation = Animation(viewer)

viewer.dims.ndisplay = 3
viewer.camera.angles = (0.0, 0.0, 90.0)
animation.capture_keyframe()
viewer.camera.zoom = 2.4
animation.capture_keyframe()
viewer.camera.angles = (-7.0, 15.7, 62.4)
animation.capture_keyframe(steps=60)
viewer.camera.angles = (2.0, -24.4, -36.7)
animation.capture_keyframe(steps=60)
viewer.reset_view()
viewer.camera.angles = (0.0, 0.0, 90.0)
animation.capture_keyframe()
animation.animate('demo.mov', canvas_only=False)
```

### Importing processed Luxendo folders

Open **Import Luxendo folder** at the top of the animation dock. Paste a folder
path or use **Browse…**, then click **Import channels**. Choose a processed task
folder, or its parent with **Include subfolders** enabled.

Each per-channel `.lux.h5` or `.h5` file becomes an image layer. The importer
reads `processingInformation.voxel_size_um` from the embedded `metadata`
JSON and sets Z/Y/X spacing from depth/height/width in micrometers. It also
fills the **Voxel size** controls automatically. Channels can have different
spacing; no manual **Apply voxel size** step is needed.

Image pyramids stay lazy, metadata preparation runs in a background worker,
and channels are added one at a time. Each new channel gets napari's **Auto
contrast** adjustment as it is added. **Cancel** stops remaining work and keeps
channels already added. Companion `main*` and `*_bdv.h5` headers are excluded.
Files with missing or invalid voxel metadata are skipped with an explanation.
Repeating an import skips files already loaded through this importer.

In Python:

```python
from napari_animation import import_luxendo_folder

layers = import_luxendo_folder(
    viewer, r"D:\data\Embryo8b\processed", recursive=True
)
```

The Python call is synchronous; the dock uses background preparation. Saved
keyframes record each channel's file path so **Load Keyframes** can reopen it.
Separate timepoints or tiles remain separate layers; the importer does not
assemble a time series or apply registration transforms.

### Editing keyframes

Resolution defaults to **level 0 (highest resolution)** for new multiscale
images and optical sections in the animation dock. Changing an image's
**resolution** control or a section's **Resolution** control updates that
image or section in every existing keyframe, including scroll-through endpoints.
The choice is saved with keyframes; selecting another frame refreshes the scene
controls to match its rendered state. Fine 2D sections may briefly show a coarse
preview while their requested resolution loads. Higher resolution requires more
memory and can take longer to load.

Keyframes can be **dragged to reorder** them in the list. Each keyframe row has
an **Overwrite** button that replaces that keyframe's captured view with the
current viewer state (keeping its name, steps and easing), so you can fix up a
single keyframe without deleting and re-adding it.

### Export quality and size estimates

The video export dialog shows the output resolution, duration and approximate
file-size range for the selected quality level. Hover over **Quality** or its
slider to compare all ten levels. **Render size** offers common 16:9 and 4:3
presets, including Full HD and 4K. A preset makes napari rerender each canvas
frame at that size, independent of the window and monitor dimensions. Presets
require **Canvas only**. **Current canvas** keeps the existing on-screen size.
If the preset has a different aspect ratio, export fits the existing view in
the new frame and shows more scene along one axis. The selected data pyramid
level still determines how much source detail is available.
Quality controls compression. Estimates update with the format and selected
render size. GIF and PNG exports do not use the quality slider.

File sizes are rough heuristics based on pixel count, frame count and quality,
without encoding a sample. Current-canvas video frames are padded with black
pixels on the right and bottom to multiples of 16, preserving every captured
pixel. Preset videos retain the selected dimensions exactly. Motion, noise and
image detail can put the actual size outside the displayed range. For Current
canvas, reopen the dialog if you resize the viewer before exporting.

### Rendering performance

Interactive 2D sections read and project large slabs in a worker so napari can
keep handling input. For a large compressed HDF5 dataset exposed directly,
through an unchanged Dask `from_array` wrapper, or through the unmodified
single-time/channel graph of `napari-imaris-loader`, the projection worker starts
a small read-only process pool. Independent HDF5 readers decompress output
stripes concurrently from the first request. The default is up to
four processes when there is enough available RAM. Large uncompressed Luxendo
slabs with many native Dask chunks use one direct HDF5 read to avoid chunk
scheduling overhead; other transformed reader graphs keep the Dask path. Set
`NAPARI_ANIMATION_H5_PROCESS_LANES=0` before starting napari to disable this,
or set a value from 2 to 8 to force a lane count. The process pool is released
when the animation dock closes.

Frames are rendered on the main thread (napari needs its single OpenGL context
there), while encoding and writing to disk happen on a background thread so the
two overlap. By default `animate()` also logs a per-phase performance summary
so you can see what is rate-limiting:

```
napari-animation performance summary:
  interpolate  total=  0.10s  mean=    1.2ms  n=80     2.1%
  apply        total=  0.40s  mean=    5.0ms  n=80     8.3%
  screenshot   total=  4.10s  mean=   51.3ms  n=80    85.4%
  encode/write total=  0.20s  mean=    2.5ms  n=80     4.2%
  finalize     total=  0.05s
  wall-clock   total=  4.30s
```

A large `screenshot`/`apply` share usually means the viewer is waiting on lazy
data (e.g. dask-backed reads); a large `encode/write` share means disk/codec is
the bottleneck. Pass `perf_log=False` to silence it. If a video writer can't be
created, the fallback to a folder of PNGs is now reported rather than silent.
The report is also written to a `render_log.txt` next to the output so it can be
found and shared.

When the `apply` phase dominates (lazily-loaded data such as Imaris/HDF5), the
renderer **prefetches** upcoming frames' data slices on background threads so
several disk reads overlap and land warm in cache:

```python
animation.animate('movie.mp4', prefetch=4, prefetch_workers=4)  # defaults
animation.animate('movie.mp4', prefetch=8, prefetch_workers=8)  # lean in harder
animation.animate('movie.mp4', prefetch=0)                      # disable
```

An opportunistic **dask cache** is also enabled during rendering (`dask_cache`,
default `"auto"`) so a chunk decompressed by a prefetch thread is reused by the
main-thread read instead of being recomputed, and frames that share a chunk are
free. Size it explicitly with `dask_cache=2_000_000_000` or turn it off with
`dask_cache=None`.

Prefetching only warms caches (it never changes the output), and reads go
through dask's scheduler. If your data source is not safe for concurrent reads,
set `prefetch=0`. If your storage serves reads in parallel (SSD/NVMe/RAID),
raise `prefetch`/`prefetch_workers` for more overlap.

### Saving and resuming keyframes

Keyframes can be saved to a file so an animation can be resumed in a later
napari session. In the GUI, use the **Save Keyframes** / **Load Keyframes**
buttons; from Python use:

```python
animation.save_keyframes('my_animation.json')
# ... later, in a new session with the same data loaded ...
animation.load_keyframes('my_animation.json')
```

The file stores each keyframe's viewer state (camera, dims, layer display
settings, optical section), interpolation settings and a thumbnail. Pixel data
is **not** stored; instead the source file path of each layer is recorded and
those layers are re-opened on load (matched to keyframe state by layer name).
Pass `reload_layers=False` to skip re-opening and match against layers already
present in the viewer.

### Ortho slicer (optical sections)

An Imaris-style ortho slicer restricts the display to an optical section of a
chosen thickness (a number of planes) centered on the currently selected plane.
Enable it from the **Ortho slicer** panel in the GUI, or from Python:

```python
from napari_animation.ortho_slicer import view_to_axis

animation.ortho_slicer.enabled = True
animation.ortho_slicer.thickness = 7          # planes
animation.ortho_slicer.mode = 'projection'    # or 'clip'
animation.ortho_slicer.projection_mode = 'max'  # max/mean/min/sum
# orientation: defaults to XY; toggle to XZ / YZ via the GUI or set the axis
animation.ortho_slicer.axis = view_to_axis('YZ', viewer.dims.ndim)
animation.ortho_slicer.apply(viewer)
```

In `projection` mode the slab is projected into the 2D slice (a thick optical
section); in `clip` mode a 3D rendering is kept but only the slab is rendered.
The optical section defaults to the **XY** view (slab along Z) and can be
toggled to the **XZ** or **YZ** orthogonal views (slab along Y or X), which
changes the axis the slab runs along. The widget also shows the physical
section thickness derived from the layer scale/units (e.g. `0.5 µm/plane → 3.5
µm` for a 7-plane section), so you know how thick the optical section is. The optical-section parameters are
captured into keyframes, so the thickness, orientation or slab position can be
animated (e.g. a slab that sweeps through z or grows over the movie).

## Is everything animate-able?

Not everything, but most of what shapes the rendered view. Keyframes capture:

- `Viewer.camera` (angles are interpolated spherically, zoom logarithmically)
- `Viewer.dims`, including switching between 2D and 3D at a keyframe
- layer transforms: `scale`, `translate`, `rotate`, `shear`, `affine`
- layer display: `opacity`, `blending`, `visible`, `contrast_limits`,
  `gamma`, `colormap`, `rendering`, `iso_threshold`, `attenuation`,
  `interpolation2d` / `interpolation3d`, and the pyramid level
- image `depiction` and `plane` (position, normal and thickness)
- the ortho slicer's optical section
- scene objects: clipping planes and ortho slices, whose geometry is
  interpolated and which can be switched on or off per keyframe

Discrete values (2D/3D mode, colormap, rendering mode, visibility) switch on
arrival at a keyframe rather than being blended. Layer data itself is not
stored; saved keyframe files record each layer's source path instead.

## Contributing

Contributions are very welcome and a detailed contributing guide is coming soon. 

Tests are run with `pytest` (`pip install -e ".[testing]"`). Performance
benchmarks are in [benchmarks](benchmarks), and design, review and
performance notes from development are in [docs/dev-notes](docs/dev-notes).

We use [`pre-commit`](https://pre-commit.com) to sort imports with
[`isort`](https://github.com/timothycrosley/isort), format code with
[`black`](https://github.com/psf/black), and lint with
[`flake8`](https://github.com/PyCQA/flake8) automatically prior to each commit.
To minmize test errors when submitting pull requests, please install `pre-commit`
in your environment as follows:

```sh
pre-commit install
```



## License

Distributed under the terms of the [BSD-3] license,
"napari-animation" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/sofroniewn/napari-animation/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
