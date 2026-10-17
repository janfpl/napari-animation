from pathlib import Path

from napari import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QErrorMessage,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..animation import Animation
from .export_controller import QtExportController
from .frame_widget import FrameWidget
from .gui_timing import GuiActionTimer, debug_enabled
from .keyframelistcontrol_widget import KeyFrameListControlWidget
from .keyframeslist_widget import KeyFramesListWidget
from .luxendo_import_widget import LuxendoImportWidget
from .ortho_slicer_widget import OrthoSlicerWidget
from .savedialog_widget import SaveDialogWidget
from .scene_widget import SceneWidget
from .scroll_widget import ScrollThroughWidget
from .voxel_size_widget import VoxelSizeWidget


class AnimationWidget(QWidget):
    """Widget for interatviely making animations using the napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        napari viewer.

    Attributes
    ----------
    viewer : napari.Viewer
        napari viewer.
    animation : napari_animation.Animation
        napari-animation animation in sync with the GUI.
    """

    def __init__(self, viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        # A dock opened from napari's Plugins menu receives a PublicOnlyProxy
        # of the viewer. This plugin drives napari's canvas, layer controls
        # and slicer, and Qt rejects proxied widgets as arguments (e.g. as a
        # dialog parent), so work with the viewer itself.
        viewer = getattr(viewer, "__wrapped__", viewer)
        # Store reference to viewer and create animation
        self.viewer = viewer
        self.animation = Animation(self.viewer)
        self._export_controller = None

        # Initialise User Interface
        self.keyframesListControlWidget = KeyFrameListControlWidget(
            animation=self.animation, parent=self
        )
        self.keyframesListWidget = KeyFramesListWidget(
            self.animation.key_frames, parent=self
        )
        self.keyframesListWidget.set_overwrite_callback(
            self._overwrite_keyframe_callback
        )
        self.frameWidget = FrameWidget(parent=self)
        self.orthoSlicerWidget = OrthoSlicerWidget(parent=self)
        self.voxelSizeWidget = VoxelSizeWidget(parent=self)
        self.luxendoImportWidget = LuxendoImportWidget(parent=self)
        self.luxendoImportWidget.imported.connect(self._on_luxendo_imported)
        self.sceneWidget = SceneWidget(parent=self)
        self.scrollWidget = ScrollThroughWidget(parent=self)
        self.saveButton = QPushButton("Save Animation", parent=self)
        self.saveButton.setEnabled(len(self.animation.key_frames) > 1)
        self.unitsWarningLabel = QLabel(parent=self)
        self.unitsWarningLabel.setWordWrap(True)
        self.unitsWarningLabel.setStyleSheet("color: #b35a00;")
        self.unitsWarningLabel.hide()

        # buttons to persist keyframes so an animation can be resumed later
        self.saveKeyframesButton = QPushButton("Save Keyframes", parent=self)
        self.saveKeyframesButton.setToolTip(
            "Save keyframes to a file to resume editing later"
        )
        self.saveKeyframesButton.setEnabled(bool(self.animation.key_frames))
        self.loadKeyframesButton = QPushButton("Load Keyframes", parent=self)
        self.loadKeyframesButton.setToolTip(
            "Load keyframes previously saved to a file"
        )
        self.keyframeIOWidget = QWidget(parent=self)
        self.keyframeIOWidget.setLayout(QHBoxLayout())
        self.keyframeIOWidget.layout().addWidget(self.saveKeyframesButton)
        self.keyframeIOWidget.layout().addWidget(self.loadKeyframesButton)

        self.animationSlider = QSlider(Qt.Horizontal, parent=self)
        self.animationSlider.setToolTip("Scroll through animation")
        self.animationSlider.setRange(0, len(self.animation._frames) - 1)

        # The panels together are taller than a typical side dock, so they go
        # in a scroll area. Note that adding them to this container reparents
        # them away from the AnimationWidget -- fine, because each one reads
        # ``parentWidget().animation`` in its constructor and caches it.
        self.contentWidget = QWidget(parent=self)
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        for panel in (
            self.luxendoImportWidget,
            self.voxelSizeWidget,
            self.keyframesListControlWidget,
            self.keyframesListWidget,
            self.frameWidget,
            self.sceneWidget,
            self.scrollWidget,
            self.orthoSlicerWidget,
            self.keyframeIOWidget,
        ):
            content_layout.addWidget(panel)
        content_layout.addStretch(1)
        self.contentWidget.setLayout(content_layout)

        self.scrollArea = QScrollArea(parent=self)
        self.scrollArea.setWidget(self.contentWidget)
        # resize the content to the viewport width, so panels fill the dock
        # rather than sitting at their minimum width
        self.scrollArea.setWidgetResizable(True)
        # no extra border: the dock already provides one
        self.scrollArea.setFrameShape(QFrame.NoFrame)

        # Create layout. Saving and scrubbing stay pinned below the scroll
        # area rather than scrolling out of reach with everything else.
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scrollArea)
        self.layout().addWidget(self.unitsWarningLabel)
        self.layout().addWidget(self.saveButton)
        self.layout().addWidget(self.animationSlider)

        # establish key bindings and callbacks
        self._add_keybind_callbacks()
        self._add_callbacks()
        self._update_units_warning()
        self.animation.resolution.attach_controls()

        # NAPARI_ANIMATION_DEBUG=1: time every click from release to repaint
        self._gui_timer = (
            GuiActionTimer(self.viewer, self, parent=self)
            if debug_enabled()
            else None
        )

    def _on_luxendo_imported(self, layers):
        if layers:
            self.voxelSizeWidget._refresh_layers()
            self.voxelSizeWidget.layerComboBox.setCurrentText(layers[0].name)

    def _add_keybind_callbacks(self):
        """Bind keys"""
        self._keybindings = [
            ("Alt-f", self._capture_keyframe_callback),
            ("Alt-r", self._replace_keyframe_callback),
            ("Alt-d", self._delete_keyframe_callback),
            ("Alt-a", lambda e: self.animation.key_frames.select_next()),
            ("Alt-b", lambda e: self.animation.key_frames.select_previous()),
        ]
        for key, cb in self._keybindings:
            self.viewer.bind_key(key, cb)

    def _add_callbacks(self):
        """Establish callbacks"""
        self.keyframesListControlWidget.deleteButton.clicked.connect(
            self._delete_keyframe_callback
        )
        self.keyframesListControlWidget.captureButton.clicked.connect(
            self._capture_keyframe_callback
        )
        self.saveButton.clicked.connect(self._save_callback)
        self.saveKeyframesButton.clicked.connect(self._save_keyframes_callback)
        self.loadKeyframesButton.clicked.connect(self._load_keyframes_callback)
        self.animationSlider.valueChanged.connect(self._on_slider_moved)
        self.animation._frames.events.n_frames.connect(self._nframes_changed)
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.layers.events.units.connect(self._update_units_warning)
        for layer in self.viewer.layers:
            layer.events.units.connect(self._update_units_warning)

        keyframe_list = self.animation.key_frames
        keyframe_list.events.inserted.connect(self._on_keyframes_changed)
        keyframe_list.events.removed.connect(self._on_keyframes_changed)
        keyframe_list.events.changed.connect(self._on_keyframes_changed)
        keyframe_list.selection.events.active.connect(
            self._on_active_keyframe_changed
        )
        self.animation._frames.events._current_index.connect(
            self._on_frame_index_changed
        )

    def _input_state(self):
        """Get current state of input widgets as {key->value} parameters."""
        return {
            "steps": int(self.frameWidget.stepsSpinBox.value()),
            "ease": self.frameWidget.get_easing_func(),
        }

    def _update_units_warning(self, event=None):
        """Mirror napari's layer-units warning in the animation dock."""
        layers = self.viewer.layers
        inconsistent = bool(layers) and layers.extent.units is None
        if inconsistent:
            details = ", ".join(
                f"{layer.name}: {', '.join(map(str, layer.units))}"
                for layer in layers
            )
            self.unitsWarningLabel.setText(
                "Layers have incompatible units. napari will ignore units "
                "while rendering. Set matching units in Voxel size or the "
                "layer controls. This warning alone does not stop export."
            )
            self.unitsWarningLabel.setToolTip(details)
        self.unitsWarningLabel.setVisible(inconsistent)

    def _on_layer_inserted(self, event):
        event.value.events.units.connect(self._update_units_warning)
        self._update_units_warning()

    def _on_layer_removed(self, event):
        event.value.events.units.disconnect(self._update_units_warning)
        self._update_units_warning()

    def _capture_keyframe_callback(self, event=None):
        """Record current key-frame"""
        self.animation.capture_keyframe(**self._input_state())

    def _replace_keyframe_callback(self, event=None):
        """Replace current key-frame with new view"""
        self.animation.capture_keyframe(**self._input_state(), insert=False)

    def _overwrite_keyframe_callback(self, index):
        """Overwrite the key-frame at ``index`` with the current view."""
        self.animation.overwrite_keyframe(index)

    def _delete_keyframe_callback(self, event=None):
        """Delete current key-frame"""
        if self.animation.key_frames.selection.active:
            self.animation.key_frames.remove_selected()
        else:
            raise ValueError("No selected keyframe to delete !")

    def _on_keyframes_changed(self, event=None):
        n_keyframes = len(self.animation.key_frames)
        has_frames = bool(self.animation.key_frames)

        self.keyframesListControlWidget.deleteButton.setEnabled(has_frames)
        self.keyframesListWidget.setEnabled(has_frames)
        self.frameWidget.setEnabled(has_frames)
        self.saveButton.setEnabled(n_keyframes > 1)
        self.saveKeyframesButton.setEnabled(has_frames)

    def _on_frame_index_changed(self, event=None):
        """Callback on change of last set frame index."""
        frame_index = event.value
        self.animationSlider.blockSignals(True)
        self.animationSlider.setValue(frame_index)
        self.animationSlider.blockSignals(False)

    def _on_active_keyframe_changed(self, event):
        """Callback on change of active keyframe in the key frames list."""
        active_keyframe = event.value
        self.keyframesListControlWidget.deleteButton.setEnabled(
            bool(active_keyframe)
        )

    def _on_slider_moved(self, event=None):
        frame_index = event
        if frame_index < len(self.animation._frames):
            with self.animation.key_frames.selection.events.active.blocker():
                self.animation.set_movie_frame_index(frame_index)

    def _save_callback(self, event=None):

        filters = (
            "MP4 (*.mp4)"
            ";;GIF (*.gif)"
            ";;MOV (*.mov)"
            ";;AVI (*.avi)"
            ";;MPEG (*.mpg *.mpeg)"
            ";;MKV (*.mkv)"
            ";;WMV (*.wmv)"
            ";;Folder of PNGs (*)"  # sep filters with ";;"
        )

        saveDialogWidget = SaveDialogWidget(self)
        animation_kwargs = saveDialogWidget.getAnimationParameters(
            self,
            "Save animation",
            str(Path.home()),
            filters,
            animation=self.animation,
        )

        # None when the user cancelled the dialog: nothing to do
        if not animation_kwargs:
            return

        if animation_kwargs.get("path"):
            try:
                controller = QtExportController(
                    self.animation, animation_kwargs, parent=self
                )
                self._export_controller = controller
                self.saveButton.setEnabled(False)

                def finished(_output, error):
                    self._export_controller = None
                    controller.deleteLater()
                    self.saveButton.setEnabled(
                        len(self.animation.key_frames) > 1
                    )
                    if error is not None and str(error) != "Export cancelled":
                        dialog = QErrorMessage(self)
                        dialog.showMessage(str(error))
                        dialog.exec_()

                controller.finished.connect(finished)
                controller.start()
            except Exception as err:  # noqa: BLE001 - surface, don't crash
                self._export_controller = None
                self.saveButton.setEnabled(len(self.animation.key_frames) > 1)
                # Surface any rendering/encoding error to the user instead of
                # failing silently in the Qt event loop.
                error_dialog = QErrorMessage()
                error_dialog.showMessage(str(err))
                error_dialog.exec_()

    def _save_keyframes_callback(self, event=None):
        """Save keyframes to a file so the animation can be resumed later."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save keyframes",
            str(Path.home() / "animation.json"),
            "napari-animation keyframes (*.json)",
        )
        if path:
            try:
                self.animation.save_keyframes(path)
            except (OSError, ValueError) as err:
                error_dialog = QErrorMessage()
                error_dialog.showMessage(str(err))
                error_dialog.exec_()

    def _load_keyframes_callback(self, event=None):
        """Load keyframes previously saved to a file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load keyframes",
            str(Path.home()),
            "napari-animation keyframes (*.json)",
        )
        if path:
            try:
                self.animation.load_keyframes(path)
            except (OSError, ValueError, KeyError) as err:
                error_dialog = QErrorMessage()
                error_dialog.showMessage(str(err))
                error_dialog.exec_()

    def _nframes_changed(self, event):
        has_frames = bool(event.value)
        previous = self.animationSlider.blockSignals(True)
        try:
            maximum = event.value - 1 if has_frames else 0
            self.animationSlider.setEnabled(has_frames)
            self.animationSlider.setMaximum(maximum)
            # Rebuilding a timeline can shrink its length below the old
            # slider value. Clamp while signals are blocked so a stale value
            # cannot trigger a second render against an invalid frame.
            self.animationSlider.setValue(
                min(self.animationSlider.value(), maximum)
            )
        finally:
            self.animationSlider.blockSignals(previous)

    def closeEvent(self, ev) -> None:
        if self._gui_timer is not None:
            self._gui_timer.close()
        self.luxendoImportWidget.shutdown()
        if self._export_controller is not None:
            self._export_controller.finished.connect(
                lambda *_: self.animation.close()
            )
            self._export_controller.cancel()
        else:
            self.animation.close()
        # release callbacks
        for key, _ in self._keybindings:
            self.viewer.bind_key(key, None)
        return super().closeEvent(ev)
