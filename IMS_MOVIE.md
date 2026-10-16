# Making a movie from an Imaris (.ims) file

This sets up one conda environment with napari, the
[napari-imaris-loader](https://github.com/janfpl/napari-imaris-loader) fork
(opens `.ims` files) and this napari-animation fork (makes the movie).

## 1. Create the environment (once)

Open **Anaconda Prompt** (or **Miniforge Prompt**) and run:

```bat
curl -L -o environment-ims-movie.yml https://raw.githubusercontent.com/janfpl/napari-animation/main/environment-ims-movie.yml
conda env create -f environment-ims-movie.yml
```

If you have `mamba`, `mamba env create -f environment-ims-movie.yml` is faster.

To pick up later changes to either fork:

```bat
conda activate napari-ims-movie
pip install --force-reinstall --no-deps git+https://github.com/janfpl/napari-imaris-loader.git@main git+https://github.com/janfpl/napari-animation.git@main
```

## 2. Open the .ims file

```bat
conda activate napari-ims-movie
napari D:/TH_LCS_4x.ims
```

or start `napari` and use **File > Open File(s)...**. If napari asks which
reader to use, pick **napari-imaris-loader**. Each channel becomes its own
layer, loaded as a multiscale pyramid, so even large files open quickly.

## 3. Record the movie

1. **Plugins > animation: Wizard** opens the animation panel.
2. Set up the first view (2D/3D toggle at the bottom left, rotate, zoom,
   time slider, contrast) and click **Capture**.
3. Change the view and click **Capture** again. Repeat for each key frame.
   *Steps* sets how many frames are interpolated to the next key frame.
4. Use the slider in the panel to preview, then **Save Animation** and pick an
   `.mp4` file name.

## Scripted alternative

```python
import napari
from napari_animation import Animation

viewer = napari.Viewer()
viewer.open("D:/TH_LCS_4x.ims", plugin="napari-imaris-loader")
viewer.dims.ndisplay = 3

animation = Animation(viewer)
animation.capture_keyframe()
viewer.camera.angles = (0, 90, 90)
animation.capture_keyframe(steps=60)
animation.animate("D:/TH_LCS_4x_movie.mp4", canvas_only=True, fps=30)
```

Run it with `python make_movie.py` inside the activated environment. The
viewer window has to stay visible while frames are rendered.
