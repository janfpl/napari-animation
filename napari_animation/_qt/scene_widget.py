"""Controls for the 3D scene objects: clipping planes and ortho slices.

Mirrors an Imaris object tree: a list of objects that can be added, removed,
renamed and individually switched on or off, with a properties panel for
whichever is selected. Edits are pushed to the viewer immediately, and are
recorded when a keyframe is captured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..scene import ORIENTATIONS, VIEW_NORMALS, ClipPlane, OrthoSlice
from ..scene._manipulators import INSTALL_HINT, is_available

if TYPE_CHECKING:
    from ..animation import Animation

#: Sentinel in a target-layer combo meaning "every eligible layer".
ALL_LAYERS = "<all layers>"


def _vector_widget(tooltip):
    """A Z/Y/X triple of spin boxes, returned with its container."""
    container = QWidget()
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    boxes = []
    for axis in ("Z", "Y", "X"):
        spin_box = QDoubleSpinBox()
        spin_box.setDecimals(3)
        spin_box.setRange(-1e6, 1e6)
        spin_box.setToolTip(f"{tooltip} ({axis})")
        boxes.append(spin_box)
        layout.addWidget(QLabel(axis))
        layout.addWidget(spin_box)
    container.setLayout(layout)
    return container, boxes


def _set_vector(boxes, values):
    for spin_box, value in zip(boxes, values):
        spin_box.blockSignals(True)
        spin_box.setValue(float(value))
        spin_box.blockSignals(False)


def _get_vector(boxes):
    return tuple(spin_box.value() for spin_box in boxes)


class _ObjectEditor(QWidget):
    """Base for the per-object property panels."""

    def __init__(self, scene_widget):
        super().__init__()
        self.scene_widget = scene_widget
        self.object = None

    def _changed(self, *args):
        if self.object is None:
            return
        self.write()
        self.scene_widget.apply()

    def _layer_names(self):
        return [
            layer.name
            for layer in self.scene_widget.viewer.layers
            if getattr(layer, "ndim", 0) >= 3
        ]


class ClipPlaneEditor(_ObjectEditor):
    """Position and orientation of a single cutaway plane."""

    def __init__(self, scene_widget):
        super().__init__(scene_widget)

        self.orientationComboBox = QComboBox()
        self.orientationComboBox.addItems([*ORIENTATIONS, "custom"])
        self.orientationComboBox.setToolTip(
            "Axis the plane faces; data behind it is cut away"
        )

        position_widget, self.positionBoxes = _vector_widget(
            "A point on the plane, in world units"
        )
        normal_widget, self.normalBoxes = _vector_widget(
            "Plane normal, in world units"
        )

        self.targetComboBox = QComboBox()
        self.flipButton = QPushButton("Flip side")
        self.flipButton.setToolTip("Cut away the other side of the plane")

        layout = QFormLayout()
        layout.addRow("Facing", self.orientationComboBox)
        layout.addRow("Position", position_widget)
        layout.addRow("Normal", normal_widget)
        layout.addRow("Cuts", self.targetComboBox)
        layout.addRow(self.flipButton)
        self.setLayout(layout)

        self.orientationComboBox.currentIndexChanged.connect(
            self._orientation_chosen
        )
        for spin_box in (*self.positionBoxes, *self.normalBoxes):
            spin_box.valueChanged.connect(self._changed)
        self.targetComboBox.currentIndexChanged.connect(self._changed)
        self.flipButton.clicked.connect(self._flip)

    def read(self, clip_plane):
        """Populate the panel from ``clip_plane``."""
        self.object = None  # suppress write-back while loading
        _set_vector(self.positionBoxes, clip_plane.position)
        _set_vector(self.normalBoxes, clip_plane.normal)

        name = next(
            (
                key
                for key, value in ORIENTATIONS.items()
                if tuple(value) == tuple(clip_plane.normal)
            ),
            "custom",
        )
        self.orientationComboBox.blockSignals(True)
        self.orientationComboBox.setCurrentText(name)
        self.orientationComboBox.blockSignals(False)

        self.targetComboBox.blockSignals(True)
        self.targetComboBox.clear()
        self.targetComboBox.addItems([ALL_LAYERS, *self._layer_names()])
        self.targetComboBox.setCurrentText(
            clip_plane.targets[0] if clip_plane.targets else ALL_LAYERS
        )
        self.targetComboBox.blockSignals(False)

        self.object = clip_plane

    def write(self):
        self.object.position = _get_vector(self.positionBoxes)
        self.object.normal = _get_vector(self.normalBoxes)
        target = self.targetComboBox.currentText()
        self.object.targets = () if target == ALL_LAYERS else (target,)

    def _orientation_chosen(self, *args):
        name = self.orientationComboBox.currentText()
        if name in ORIENTATIONS:
            _set_vector(self.normalBoxes, ORIENTATIONS[name])
        self._changed()

    def _flip(self, *args):
        _set_vector(
            self.normalBoxes,
            [-value for value in _get_vector(self.normalBoxes)],
        )
        self._changed()


class OrthoSliceEditor(_ObjectEditor):
    """Geometry and appearance of a thick optical section."""

    def __init__(self, scene_widget):
        super().__init__(scene_widget)

        self.sourceComboBox = QComboBox()
        self.sourceComboBox.setToolTip("Layer whose data the slab shows")

        self.viewComboBox = QComboBox()
        self.viewComboBox.addItems([*VIEW_NORMALS, "custom"])
        self.viewComboBox.setToolTip(
            "Orthogonal view the section faces "
            "(XY slices Z, XZ slices Y, YZ slices X)"
        )

        position_widget, self.positionBoxes = _vector_widget(
            "Centre of the slab, in world units"
        )
        normal_widget, self.normalBoxes = _vector_widget(
            "Slab normal, in world units"
        )

        self.thicknessSpinBox = QDoubleSpinBox()
        self.thicknessSpinBox.setDecimals(3)
        self.thicknessSpinBox.setRange(0.001, 1e6)
        self.thicknessSpinBox.setToolTip(
            "Slab thickness in world units, measured along the normal"
        )

        self.projectionComboBox = QComboBox()
        self.projectionComboBox.addItems(["max", "mean", "min"])

        self.renderComboBox = QComboBox()
        self.renderComboBox.addItems(["plane", "clip"])
        self.renderComboBox.setToolTip(
            "plane: a slab layer drawn alongside the volume (one extra GPU "
            "texture)\nclip: the slab carved out of the source layer (no "
            "extra memory)"
        )

        self.manipulatorButton = QPushButton("Edit in canvas")
        if is_available():
            self.manipulatorButton.setToolTip(
                "Drag the plane directly in the 3D canvas"
            )
        else:
            self.manipulatorButton.setEnabled(False)
            self.manipulatorButton.setToolTip(INSTALL_HINT)

        layout = QFormLayout()
        layout.addRow("Source", self.sourceComboBox)
        layout.addRow("View", self.viewComboBox)
        layout.addRow("Centre", position_widget)
        layout.addRow("Normal", normal_widget)
        layout.addRow("Thickness", self.thicknessSpinBox)
        layout.addRow("Projection", self.projectionComboBox)
        layout.addRow("Render as", self.renderComboBox)
        layout.addRow(self.manipulatorButton)
        self.setLayout(layout)

        self.viewComboBox.currentIndexChanged.connect(self._view_chosen)
        for spin_box in (
            *self.positionBoxes,
            *self.normalBoxes,
            self.thicknessSpinBox,
        ):
            spin_box.valueChanged.connect(self._changed)
        for combo in (
            self.sourceComboBox,
            self.projectionComboBox,
            self.renderComboBox,
        ):
            combo.currentIndexChanged.connect(self._changed)
        self.manipulatorButton.clicked.connect(self._edit_in_canvas)

    def read(self, ortho_slice):
        self.object = None
        _set_vector(self.positionBoxes, ortho_slice.position)
        _set_vector(self.normalBoxes, ortho_slice.normal)

        self.thicknessSpinBox.blockSignals(True)
        self.thicknessSpinBox.setValue(float(ortho_slice.thickness))
        self.thicknessSpinBox.blockSignals(False)

        name = next(
            (
                key
                for key, value in VIEW_NORMALS.items()
                if tuple(value) == tuple(ortho_slice.normal)
            ),
            "custom",
        )
        for combo, value in (
            (self.viewComboBox, name),
            (self.projectionComboBox, ortho_slice.projection),
            (self.renderComboBox, ortho_slice.render_mode),
        ):
            combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(False)

        self.sourceComboBox.blockSignals(True)
        self.sourceComboBox.clear()
        names = [
            name
            for name in self._layer_names()
            if name != ortho_slice.layer_name
        ]
        self.sourceComboBox.addItems(names)
        if ortho_slice.source:
            self.sourceComboBox.setCurrentText(ortho_slice.source)
        self.sourceComboBox.blockSignals(False)

        self.object = ortho_slice

    def write(self):
        self.object.position = _get_vector(self.positionBoxes)
        self.object.normal = _get_vector(self.normalBoxes)
        self.object.thickness = self.thicknessSpinBox.value()
        self.object.projection = self.projectionComboBox.currentText()
        self.object.render_mode = self.renderComboBox.currentText()
        source = self.sourceComboBox.currentText()
        self.object.source = source or None

    def _view_chosen(self, *args):
        name = self.viewComboBox.currentText()
        if name in VIEW_NORMALS:
            _set_vector(self.normalBoxes, VIEW_NORMALS[name])
        self._changed()

    def _edit_in_canvas(self, *args):
        """Attach a napari-threedee manipulator to the backing layer."""
        from ..scene._manipulators import attach_plane_manipulator

        if self.object is None:
            return
        viewer = self.scene_widget.viewer
        layer = viewer.layers.get(self.object.layer_name)
        if layer is None:
            return
        # keep a reference: the manipulator stops working if it is collected
        self.scene_widget._manipulators[self.object.id] = (
            attach_plane_manipulator(viewer, layer)
        )


class SceneWidget(QGroupBox):
    """List of clipping planes and ortho slices, with their properties."""

    def __init__(self, parent=None):
        super().__init__("3D scene", parent=parent)
        self.animation: Animation = self.parentWidget().animation
        self.viewer = self.animation.viewer
        self._manipulators = {}

        self.listWidget = QListWidget()
        self.listWidget.setToolTip(
            "Objects in the 3D scene. Untick to switch one off; capture a "
            "keyframe to record the state."
        )

        self.addPlaneButton = QPushButton("+ Clipping plane")
        self.addSliceButton = QPushButton("+ Ortho slice")
        self.removeButton = QPushButton("Remove")
        self.removeButton.setEnabled(False)

        buttons = QWidget()
        buttons.setLayout(QHBoxLayout())
        buttons.layout().setContentsMargins(0, 0, 0, 0)
        for button in (
            self.addPlaneButton,
            self.addSliceButton,
            self.removeButton,
        ):
            buttons.layout().addWidget(button)

        self.clipPlaneEditor = ClipPlaneEditor(self)
        self.orthoSliceEditor = OrthoSliceEditor(self)
        self.editorStack = QStackedWidget()
        self.editorStack.addWidget(QWidget())  # nothing selected
        self.editorStack.addWidget(self.clipPlaneEditor)
        self.editorStack.addWidget(self.orthoSliceEditor)

        layout = QVBoxLayout()
        layout.addWidget(self.listWidget)
        layout.addWidget(buttons)
        layout.addWidget(self.editorStack)
        self.setLayout(layout)

        self.addPlaneButton.clicked.connect(self.add_clip_plane)
        self.addSliceButton.clicked.connect(self.add_ortho_slice)
        self.removeButton.clicked.connect(self.remove_selected)
        self.listWidget.currentRowChanged.connect(self._selection_changed)
        self.listWidget.itemChanged.connect(self._item_changed)

        self.refresh()

    # ---------------------------------------------------------------- state

    def apply(self):
        """Push the live scene onto the viewer."""
        ortho, _ = self.animation._capture_state()
        self.animation.scene.apply(self.viewer, ortho=ortho)

    def refresh(self):
        """Rebuild the object list from the scene."""
        self.listWidget.blockSignals(True)
        self.listWidget.clear()
        for scene_object in self.animation.scene:
            item = QListWidgetItem(scene_object.name)
            item.setFlags(
                item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable
            )
            item.setCheckState(
                Qt.Checked if scene_object.enabled else Qt.Unchecked
            )
            self.listWidget.addItem(item)
        self.listWidget.blockSignals(False)
        self._selection_changed(self.listWidget.currentRow())

    def selected_object(self):
        row = self.listWidget.currentRow()
        if 0 <= row < len(self.animation.scene):
            return self.animation.scene[row]
        return None

    # -------------------------------------------------------------- actions

    def _volume_extent(self):
        """A sensible default position: the centre of the data."""
        try:
            extent = self.viewer.layers.extent.world
            return tuple(
                float(value) for value in ((extent[0] + extent[1]) / 2)[-3:]
            )
        except Exception:  # noqa: BLE001 - empty viewer, or no spatial layers
            return (0.0, 0.0, 0.0)

    def _unique_name(self, stem):
        existing = {obj.name for obj in self.animation.scene}
        if stem not in existing:
            return stem
        index = 2
        while f"{stem} {index}" in existing:
            index += 1
        return f"{stem} {index}"

    def add_clip_plane(self, *args):
        plane = ClipPlane(
            name=self._unique_name("Clipping plane"),
            position=self._volume_extent(),
        )
        self.animation.add_scene_object(plane)
        self.refresh()
        self.listWidget.setCurrentRow(len(self.animation.scene) - 1)

    def add_ortho_slice(self, *args):
        centre = self._volume_extent()
        slice_ = OrthoSlice(
            name=self._unique_name("Ortho slice"),
            position=centre,
            thickness=self._default_thickness(),
        )
        self.animation.add_scene_object(slice_)
        self.refresh()
        self.listWidget.setCurrentRow(len(self.animation.scene) - 1)

    def _default_thickness(self):
        """One voxel along Z, so the slab starts as a single plane."""
        for layer in self.viewer.layers:
            if getattr(layer, "ndim", 0) >= 3:
                return float(layer.scale[-3])
        return 1.0

    def remove_selected(self, *args):
        scene_object = self.selected_object()
        if scene_object is None:
            return
        self._manipulators.pop(scene_object.id, None)
        self.animation.remove_scene_object(scene_object)
        self.refresh()

    # -------------------------------------------------------------- signals

    def _selection_changed(self, row):
        scene_object = self.selected_object()
        self.removeButton.setEnabled(scene_object is not None)
        if isinstance(scene_object, OrthoSlice):
            self.orthoSliceEditor.read(scene_object)
            self.editorStack.setCurrentWidget(self.orthoSliceEditor)
        elif isinstance(scene_object, ClipPlane):
            self.clipPlaneEditor.read(scene_object)
            self.editorStack.setCurrentWidget(self.clipPlaneEditor)
        else:
            self.editorStack.setCurrentIndex(0)

    def _item_changed(self, item):
        """A name edit or an on/off tick in the list."""
        row = self.listWidget.row(item)
        if not 0 <= row < len(self.animation.scene):
            return
        scene_object = self.animation.scene[row]
        enabled = item.checkState() == Qt.Checked
        name = item.text()

        if isinstance(scene_object, OrthoSlice) and name != scene_object.name:
            # the backing layer is named after the object; move it too
            old_name = scene_object.layer_name
            if old_name in self.viewer.layers:
                self.viewer.layers[old_name].name = name

        scene_object.enabled = enabled
        scene_object.name = name
        self.apply()
