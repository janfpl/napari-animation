# Installing this fork

Instructions for a fresh conda environment containing only Python 3.11.

## The short version

```bash
conda create -n napari-anim python=3.11 -y
conda activate napari-anim

# napari + a Qt backend. This is the step that is easy to get wrong:
# the plugin does NOT pull in a Qt backend of its own.
python -m pip install "napari[pyqt6]>=0.5"

git clone https://github.com/janfpl/napari-animation.git
cd napari-animation
git checkout claude/3d-animation-capabilities-3phx02
python -m pip install -e .

# check it worked
python -c "import napari, napari_animation; print(napari.__version__)"
napari
```

In napari: **Plugins → napari-animation → Wizard**.

## Why each step is the way it is

**Install napari yourself, before the plugin.** `setup.cfg` lists plain
`napari`, which has no Qt backend as a dependency — install only the plugin and
you get a working import but no GUI. `napari[pyqt6]` pulls the backend in.
PyQt5 also works, but napari 0.8 warns that system theme detection needs Qt6.

**Python 3.11 is fine, but not the floor.** The package declares
`python_requires = >=3.9`. napari 0.5+ is what actually constrains you.

**The napari floor matters.** This fork requires **napari >= 0.5** —
`layer.projection_mode` and `dims.margin_left/right` do not exist before that,
and the ortho slicer uses both. It is developed and tested against **napari
0.8.0**. The declared floor used to be `0.4.8`, which would let pip resolve a
napari too old to run the code; that has been corrected.

**Install from a git clone, not a tarball.** Versioning uses
`setuptools_scm`, which reads the git history. Installing from a directory with
no `.git` fails at build time.

**`-e` (editable) is recommended** while this branch is under development, so
`git pull` is all it takes to update.

**Video export needs no separate ffmpeg.** `imageio-ffmpeg` is a hard
dependency and ships its own binary. A path with no video extension writes a
folder of PNGs instead of a movie — that is a common surprise, and the code
warns loudly when it happens.

## Verifying the install

```bash
python -m pip install -e ".[testing]"
python -m pytest napari_animation/_tests -q
```

Tests that open a viewer or take a screenshot need a working OpenGL context.
On a headless machine they will fail or abort; the model-level tests
(`test_scene.py`, `test_ortho_slice.py`, `test_animatable_state.py`,
`test_ortho_slicer.py`, `test_io.py`) run without one:

```bash
python -m pytest napari_animation/_tests/test_scene.py \
                 napari_animation/_tests/test_ortho_slice.py \
                 napari_animation/_tests/test_animatable_state.py \
                 napari_animation/_tests/test_ortho_slicer.py \
                 napari_animation/_tests/test_io.py -q
```

Confirm napari can see the plugin:

```bash
python -c "
import npe2
pm = npe2.PluginManager.instance(); pm.discover(include_npe1=True)
print([m.name for m in pm.iter_manifests() if 'animation' in m.name])
"
```

This plugin still registers through the old `napari.plugin` entry point.
napari 0.8 has no built-in npe1 support, but npe2 ships an npe1 adapter that
picks it up, so the widget appears in the Plugins menu. Migrating to an npe2
`napari.yaml` manifest is worth doing before that adapter goes away.

## Optional: in-canvas plane dragging

Positioning ortho slice planes by dragging them rather than typing coordinates
uses [napari-threedee](https://github.com/napari-threedee/napari-threedee):

```bash
python -m pip install "napari-animation[threedee]"
```

**This currently fails on Python 3.11+.** napari-threedee pins `zarr<3`, which
pulls in `asciitree`, which does not build on modern Python. Until that pin is
relaxed upstream, either use a Python 3.10 environment for it or skip it —
everything else works without it, and the code degrades gracefully when it is
absent.

## Quick smoke test

```python
import numpy as np, napari
from napari_animation import Animation, ClipPlane, OrthoSlice

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(np.random.random((64, 128, 128)), name="vol", scale=(2, 1, 1))

animation = Animation(viewer)
animation.add_scene_object(ClipPlane(name="cut", position=(64, 64, 64)))
animation.add_scene_object(
    OrthoSlice(name="Ortho XY", source="vol", thickness=10.0)
)

viewer.camera.angles = (0, 0, 90)
animation.capture_keyframe()
viewer.camera.angles = (-20, 40, 70)
animation.capture_keyframe(steps=30)
animation.animate("test.mp4")
```

See `examples/imaris_style_flythrough.py` for a fuller example.
