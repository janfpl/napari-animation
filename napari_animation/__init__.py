"""Public animation API with lightweight imports for reader subprocesses."""

from importlib import import_module

from ._hookimpls import napari_experimental_provide_dock_widget

_EXPORTS = {
    "import_luxendo_folder": ".luxendo",
    "AnimationWidget": "._qt",
    "Animation": ".animation",
    "KeyFrame": ".key_frame",
    "OrthoSlicer": ".ortho_slicer",
    "ViewerState": ".viewer_state",
    "ClipPlane": ".scene",
    "LayerRegistry": ".scene",
    "OrthoSlice": ".scene",
    "Scene": ".scene",
    "SceneObject": ".scene",
    "SceneRenderer": ".scene",
}

__all__ = [
    "napari_experimental_provide_dock_widget",
    *_EXPORTS,
]


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
