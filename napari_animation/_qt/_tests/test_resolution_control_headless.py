"""Resolution control logic without an OpenGL canvas."""

import numpy as np
from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication, QWidget

from napari_animation import Animation
from napari_animation._qt.scene_widget import SceneWidget
from napari_animation.scene import OrthoSlice


def test_scene_editor_exposes_pyramid_level():
    app = QApplication.instance() or QApplication([])
    viewer = ViewerModel(ndisplay=3)
    viewer.add_image(
        [np.zeros((8, 16, 16)), np.zeros((4, 8, 8))],
        name="pyramid", multiscale=True,
    )
    animation = Animation(viewer)
    section = animation.add_scene_object(
        OrthoSlice(name="section", source="pyramid")
    )
    parent = QWidget()
    parent.animation = animation
    widget = SceneWidget(parent)
    widget.listWidget.setCurrentRow(0)
    control = widget.orthoSliceEditor.sourceLevelComboBox
    assert control.findData(-1) >= 0
    assert control.findData(0) >= 0
    control.setCurrentIndex(control.findData(0))
    assert section.source_level == 0
    widget.close()
    assert app is not None
