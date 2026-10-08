from napari_plugin_engine import napari_hook_implementation


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    from ._qt import AnimationWidget

    widget_options = {
        "name": "Wizard",
        "add_vertical_stretch": False,
    }
    return AnimationWidget, widget_options
