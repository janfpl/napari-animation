"""Correct the physical voxel spacing of imported layers.

File metadata is often wrong -- most commonly a TIFF whose Z spacing does not
match the microscope's step size, leaving the volume stretched or squashed
along Z. Everything downstream depends on getting this right: an ortho slice's
thickness is specified in world units, and clipping planes are positioned in
world coordinates, so both are only physically meaningful once the voxel size
is correct.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

if TYPE_CHECKING:
    from ..animation import Animation

#: Units offered for voxel spacing. napari stores these as pint units.
UNITS = ("um", "nm", "mm", "pixel")


class VoxelSizeWidget(QGroupBox):
    """Set the physical size of a voxel for a layer (or every image layer)."""

    def __init__(self, parent=None):
        super().__init__("Voxel size", parent=parent)
        self.animation: Animation = self.parentWidget().animation
        self.viewer = self.animation.viewer
        self.setToolTip(
            "Correct the physical voxel spacing when a file's metadata is "
            "wrong, e.g. the Z step of an imported TIFF"
        )

        self.layerComboBox = QComboBox()
        self.layerComboBox.setToolTip("Layer to read the current spacing from")

        self.spinBoxes = {}
        axes_widget = QWidget()
        axes_layout = QHBoxLayout()
        axes_layout.setContentsMargins(0, 0, 0, 0)
        for axis in ("Z", "Y", "X"):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(6)
            spin_box.setRange(1e-6, 1e6)
            spin_box.setValue(1.0)
            # six decimals over a wide range is a very wide natural width;
            # three side by side would force the dock to scroll horizontally
            spin_box.setMinimumWidth(70)
            spin_box.setMaximumWidth(120)
            spin_box.setToolTip(f"Voxel size along {axis}")
            self.spinBoxes[axis] = spin_box
            axes_layout.addWidget(QLabel(axis))
            axes_layout.addWidget(spin_box)
        axes_widget.setLayout(axes_layout)

        self.unitsComboBox = QComboBox()
        self.unitsComboBox.addItems(UNITS)

        self.allLayersCheckBox = QCheckBox("Apply to all image layers")
        self.allLayersCheckBox.setChecked(True)
        self.allLayersCheckBox.setToolTip(
            "Channels of the same acquisition normally share a voxel size"
        )

        self.applyButton = QPushButton("Apply voxel size")
        self.applyButton.setToolTip(
            "Applies to the viewer and to every keyframe already captured, "
            "since voxel size describes the data rather than the animation"
        )

        layout = QFormLayout()
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addRow("Layer", self.layerComboBox)
        layout.addRow("Spacing", axes_widget)
        layout.addRow("Units", self.unitsComboBox)
        layout.addRow(self.allLayersCheckBox)
        layout.addRow(self.applyButton)
        self.setLayout(layout)

        self.layerComboBox.currentIndexChanged.connect(self._load_from_layer)
        self.applyButton.clicked.connect(self._apply)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)

        self._refresh_layers()

    # ------------------------------------------------------------- helpers

    def _image_layers(self):
        """Layers that have a spatial scale worth editing."""
        return [
            layer
            for layer in self.viewer.layers
            if getattr(layer, "ndim", 0) >= 2
        ]

    def _selected_layer(self):
        name = self.layerComboBox.currentText()
        if name and name in self.viewer.layers:
            return self.viewer.layers[name]
        return None

    def _refresh_layers(self, event=None):
        current = self.layerComboBox.currentText()
        self.layerComboBox.blockSignals(True)
        self.layerComboBox.clear()
        self.layerComboBox.addItems(
            [layer.name for layer in self._image_layers()]
        )
        index = self.layerComboBox.findText(current)
        if index >= 0:
            self.layerComboBox.setCurrentIndex(index)
        self.layerComboBox.blockSignals(False)
        self._load_from_layer()

    def _load_from_layer(self, *args):
        """Show the selected layer's current spacing."""
        layer = self._selected_layer()
        self.applyButton.setEnabled(layer is not None)
        if layer is None:
            return

        scale = list(layer.scale)
        # map the layer's trailing axes onto Z, Y, X
        values = ([1.0] * 3 + [float(v) for v in scale])[-3:]
        for axis, value in zip(("Z", "Y", "X"), values):
            spin_box = self.spinBoxes[axis]
            spin_box.blockSignals(True)
            spin_box.setValue(value)
            spin_box.blockSignals(False)

        # a 2D layer has no Z spacing to set
        self.spinBoxes["Z"].setEnabled(layer.ndim >= 3)

        unit = str(next(iter(getattr(layer, "units", []) or []), "")).lower()
        for index, candidate in enumerate(UNITS):
            if unit.startswith(candidate.rstrip("s")) or unit == candidate:
                self.unitsComboBox.setCurrentIndex(index)
                break

    # -------------------------------------------------------------- action

    def _apply(self, *args):
        layer = self._selected_layer()
        if layer is None:
            return

        spacing = [self.spinBoxes[axis].value() for axis in ("Z", "Y", "X")]
        units = self.unitsComboBox.currentText()

        if self.allLayersCheckBox.isChecked():
            targets = [
                other
                for other in self._image_layers()
                if other.ndim == layer.ndim
            ]
        else:
            targets = [layer]

        for target in targets:
            values = spacing if target.ndim >= 3 else spacing[1:]
            self.animation.set_voxel_size(target.name, values, units=units)

        self._load_from_layer()
