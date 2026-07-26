from unittest.mock import patch

import numpy as np
from napari.components import ViewerModel

from napari_animation._qt import AnimationWidget
from napari_animation._qt.savedialog_widget import SaveDialogWidget


class _HeadlessViewer(ViewerModel):
    """Stands in for a real viewer where no OpenGL context is available."""

    def screenshot(self, *args, **kwargs):
        return np.zeros((60, 60, 4), dtype=np.uint8)


def test_panels_are_scrollable(qtbot):
    """The stack of panels is taller than a side dock, so it must scroll.

    Without this the controls at the bottom -- save/load keyframes, the ortho
    slicer -- are simply unreachable in a normal-height napari window.
    """
    viewer = _HeadlessViewer()
    viewer.add_image(np.random.random((8, 16, 16)), name="img")
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.scrollArea.widget() is widget.contentWidget
    # panels stretch to the dock width instead of sitting at minimum width
    assert widget.scrollArea.widgetResizable() is True

    # every panel is inside the scrolling area, so none can be cut off
    for panel in (
        widget.voxelSizeWidget,
        widget.sceneWidget,
        widget.scrollWidget,
        widget.orthoSlicerWidget,
        widget.keyframeIOWidget,
    ):
        assert widget.contentWidget.isAncestorOf(panel)

    # saving and scrubbing stay pinned outside it, always reachable
    assert not widget.contentWidget.isAncestorOf(widget.saveButton)
    assert not widget.contentWidget.isAncestorOf(widget.animationSlider)


def test_panels_fit_a_normal_dock_width(qtbot):
    """Vertical scrolling only: the panel must fit a typical side dock.

    A wide control anywhere in the stack forces a *horizontal* scrollbar
    across the whole dock, which is far more disruptive than the vertical one
    it is paired with. Form rows wrap and spin boxes are width-capped to keep
    this in check; this guards against a new panel undoing that.
    """
    viewer = _HeadlessViewer()
    viewer.add_image(np.random.random((8, 16, 16)), name="img")
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)

    minimum = widget.contentWidget.minimumSizeHint().width()
    assert minimum <= 340, (
        f"controls need {minimum}px, wider than a normal napari dock; "
        "cap spin box widths or let the form rows wrap"
    )


def test_panels_still_reach_their_animation(qtbot):
    """Reparenting into the scroll container must not orphan the panels.

    Each panel resolves ``parentWidget().animation`` in its constructor; if
    that were deferred it would break once the panel is moved into the
    scrolling container.
    """
    viewer = _HeadlessViewer()
    viewer.add_image(np.random.random((8, 16, 16)), name="img")
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)

    for panel in (
        widget.voxelSizeWidget,
        widget.sceneWidget,
        widget.scrollWidget,
        widget.orthoSlicerWidget,
        widget.frameWidget,
    ):
        assert panel.animation is widget.animation

    # and the panels still work after being reparented
    widget.sceneWidget.add_clip_plane()
    assert len(widget.animation.scene) == 1


def test_cancelling_the_save_dialog_is_a_no_op(qtbot):
    """Regression: backing out of the save dialog used to raise.

    The dialog returned "" on cancel and the callback did
    ``animation_kwargs["path"]``, so cancelling raised "string indices must be
    integers" instead of quietly doing nothing.
    """
    viewer = _HeadlessViewer()
    viewer.add_image(np.random.random((8, 16, 16)), name="img")
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)

    with patch.object(
        SaveDialogWidget, "getAnimationParameters", return_value=None
    ), patch.object(widget.animation, "animate") as animate:
        widget._save_callback()

    animate.assert_not_called()


def test_animation_widget(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((28, 28)))
    aw = AnimationWidget(viewer)
    qtbot.addWidget(aw)

    aw._capture_keyframe_callback()
    assert len(aw.animation.key_frames) == 1
    aw._capture_keyframe_callback()
    assert len(aw.animation.key_frames) == 2
    aw._replace_keyframe_callback()
    assert len(aw.animation.key_frames) == 2
    aw._delete_keyframe_callback()
    assert len(aw.animation.key_frames) == 1


def test_animation_widget_has_keyframe_io_and_ortho(make_napari_viewer, qtbot):
    # construction without an image layer avoids requiring an OpenGL context
    viewer = make_napari_viewer()
    aw = AnimationWidget(viewer)
    qtbot.addWidget(aw)

    assert aw.saveKeyframesButton.text() == "Save Keyframes"
    assert aw.loadKeyframesButton.text() == "Load Keyframes"

    # the ortho slicer widget drives the animation's OrthoSlicer
    ortho = aw.orthoSlicerWidget
    ortho.setChecked(True)
    ortho.thicknessSpinBox.setValue(5)
    assert aw.animation.ortho_slicer.enabled is True
    assert aw.animation.ortho_slicer.thickness == 5

    ortho.modeComboBox.setCurrentText("clip")
    assert aw.animation.ortho_slicer.mode == "clip"
    # projection type is only meaningful in projection mode
    assert not ortho.projectionComboBox.isEnabled()

    # the orthogonal view defaults to XY and can be toggled to XZ / YZ
    assert ortho.viewComboBox.currentText() == "XY"
    ndim = viewer.dims.ndim
    from napari_animation.ortho_slicer import view_to_axis

    ortho.viewComboBox.setCurrentText("YZ")
    assert aw.animation.ortho_slicer.axis == view_to_axis("YZ", ndim)
