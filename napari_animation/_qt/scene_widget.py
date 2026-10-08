"""Controls for the 3D scene objects: clipping planes and ortho slices.

Mirrors an Imaris object tree: a list of objects that can be added, removed,
renamed and individually switched on or off, with a properties panel for
whichever is selected. Edits are pushed to the viewer immediately, and are
recorded when a keyframe is captured.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..scene import ORIENTATIONS, VIEW_NORMALS, ClipPlane, OrthoSlice
from ..layer_registry import LayerRegistry, OBJECT_ID_METADATA
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
        # a wide range gives a very wide natural width; three of these side by
        # side would force the whole dock to scroll horizontally
        spin_box.setMinimumWidth(62)
        spin_box.setMaximumWidth(110)
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
            and not getattr(layer, "metadata", {}).get(OBJECT_ID_METADATA)
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
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
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

        self.channelSearch = QLineEdit()
        self.channelSearch.setPlaceholderText("Search channels")
        self.channelList = QListWidget()
        self.channelList.setMaximumHeight(150)
        self.channelList.setToolTip("Check image channels to show in this slice")
        self.channelAllButton = QPushButton("All")
        self.channelNoneButton = QPushButton("None")
        self.channelRelink = QComboBox()
        self.channelRelinkButton = QPushButton("Relink missing")
        channel_buttons = QWidget()
        channel_buttons.setLayout(QHBoxLayout())
        channel_buttons.layout().setContentsMargins(0, 0, 0, 0)
        channel_buttons.layout().addWidget(self.channelAllButton)
        channel_buttons.layout().addWidget(self.channelNoneButton)
        self.channelAppearance = QComboBox()
        self.channelAppearance.addItems(["Linked to source", "Custom"])
        self.channelColor = QComboBox()
        self.channelColor.addItems([
            "gray", "red", "green", "blue", "cyan", "magenta", "yellow",
        ])
        self.channelOpacity = QDoubleSpinBox()
        self.channelOpacity.setRange(0.0, 1.0)
        self.channelOpacity.setSingleStep(0.05)
        self.channelOpacity.setDecimals(2)
        self.channelGamma = QDoubleSpinBox()
        self.channelGamma.setRange(0.01, 20.0)
        self.channelGamma.setSingleStep(0.1)
        self.channelGamma.setDecimals(2)
        self.channelContrastMin = QDoubleSpinBox()
        self.channelContrastMax = QDoubleSpinBox()
        for box in (self.channelContrastMin, self.channelContrastMax):
            box.setRange(-1e15, 1e15)
            box.setDecimals(4)

        self.sourceLevelComboBox = QComboBox()
        self.sourceLevelComboBox.setToolTip(
            "Resolution for this section in every keyframe. Level 0 is the "
            "highest resolution. Fine levels refine after scrolling stops."
        )

        self.viewComboBox = QComboBox()
        self.viewComboBox.addItems([*VIEW_NORMALS, "custom"])
        self.viewComboBox.setToolTip(
            "Orthogonal view the section faces "
            "(XY slices Z, XZ slices Y, YZ slices X)"
        )

        position_widget, self.positionBoxes = _vector_widget(
            "Centre of the slab, in world units"
        )
        self.centreZSlider = QSlider(Qt.Horizontal)
        self.centreZSlider.setRange(0, 10000)
        self.centreZSlider.setToolTip(
            "Scroll the slab through the source volume along Z"
        )
        self.centreZSlider.setEnabled(False)
        self._z_slider_bounds = (0.0, 0.0)
        normal_widget, self.normalBoxes = _vector_widget(
            "Slab normal, in world units"
        )

        self.thicknessSpinBox = QDoubleSpinBox()
        self.thicknessSpinBox.setDecimals(3)
        self.thicknessSpinBox.setRange(0.001, 1e6)
        self.thicknessSpinBox.setMaximumWidth(130)
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
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addRow("Source", self.sourceComboBox)
        layout.addRow("Find", self.channelSearch)
        layout.addRow("Channels", self.channelList)
        layout.addRow(channel_buttons)
        layout.addRow("Relink to", self.channelRelink)
        layout.addRow(self.channelRelinkButton)
        layout.addRow("Appearance", self.channelAppearance)
        layout.addRow("Color", self.channelColor)
        layout.addRow("Opacity", self.channelOpacity)
        layout.addRow("Gamma", self.channelGamma)
        layout.addRow("Contrast min", self.channelContrastMin)
        layout.addRow("Contrast max", self.channelContrastMax)
        layout.addRow("Resolution", self.sourceLevelComboBox)
        layout.addRow("View", self.viewComboBox)
        layout.addRow("Centre", position_widget)
        layout.addRow("Centre Z", self.centreZSlider)
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
        self.positionBoxes[0].valueChanged.connect(self._sync_z_slider)
        self.centreZSlider.valueChanged.connect(self._z_slider_changed)
        self.sourceComboBox.currentIndexChanged.connect(self._source_chosen)
        self.channelList.itemChanged.connect(self._channel_toggled)
        self.channelList.currentItemChanged.connect(self._read_channel_appearance)
        self.channelSearch.textChanged.connect(self._filter_channels)
        self.channelRelinkButton.clicked.connect(self._relink_missing_channel)
        self.channelAllButton.clicked.connect(
            lambda: self._set_visible_channels(True)
        )
        self.channelNoneButton.clicked.connect(
            lambda: self._set_visible_channels(False)
        )
        for combo in (self.channelAppearance, self.channelColor):
            combo.currentIndexChanged.connect(self._channel_appearance_changed)
        for box in (
            self.channelOpacity, self.channelGamma,
            self.channelContrastMin, self.channelContrastMax,
        ):
            box.valueChanged.connect(self._channel_appearance_changed)
        for combo in (
            self.projectionComboBox,
            self.renderComboBox,
        ):
            combo.currentIndexChanged.connect(self._changed)
        self.sourceLevelComboBox.currentIndexChanged.connect(self._resolution_changed)
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
        self._refresh_channels(ortho_slice)
        self._refresh_levels(ortho_slice.source_level)

        self.object = ortho_slice
        self._refresh_z_slider()
        if self.channelList.count():
            self.channelList.setCurrentRow(0)
        self._read_channel_appearance(self.channelList.currentItem())

    def _refresh_channels(self, section):
        registry = LayerRegistry(self.scene_widget.viewer)
        self.channelList.blockSignals(True)
        self.channelList.clear()
        selected = section.channels
        if selected is None:
            selected = {}
            if section.source_id:
                selected[section.source_id] = {"enabled": True}
        seen = set()
        for layer in self.scene_widget.viewer.layers:
            if layer.metadata.get(OBJECT_ID_METADATA) or not hasattr(layer, "plane"):
                continue
            if getattr(layer, "ndim", 0) < 3 or getattr(layer, "rgb", False):
                continue
            source_id = registry.source_id(layer)
            seen.add(source_id)
            item = QListWidgetItem(layer.name)
            settings = selected.get(source_id, {})
            custom = settings.get("appearance") or {}
            color_name = custom.get("colormap") if settings.get(
                "appearance_mode"
            ) == "custom" else None
            if isinstance(color_name, dict):
                color_name = color_name.get("name")
            swatch = QColor(color_name) if color_name else QColor()
            if not swatch.isValid():
                rgba = np.asarray(layer.colormap.map(1.0)).reshape(-1)[:4]
                swatch = QColor.fromRgbF(*[float(v) for v in rgba])
            pixmap = QPixmap(12, 12)
            pixmap.fill(swatch)
            item.setIcon(QIcon(pixmap))
            item.setData(Qt.UserRole, source_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if selected.get(source_id, {}).get("enabled") else Qt.Unchecked
            )
            self.channelList.addItem(item)
        for source_id, settings in selected.items():
            if source_id in seen:
                continue
            item = QListWidgetItem(f"{settings.get('source_name', source_id)} (missing)")
            item.setData(Qt.UserRole, source_id)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            item.setCheckState(
                Qt.Checked if settings.get("enabled") else Qt.Unchecked
            )
            self.channelList.addItem(item)
        self.channelList.blockSignals(False)
        self.channelRelink.blockSignals(True)
        self.channelRelink.clear()
        for layer in self.scene_widget.viewer.layers:
            if layer.metadata.get(OBJECT_ID_METADATA) or not hasattr(layer, "plane"):
                continue
            if layer.ndim >= 3 and not getattr(layer, "rgb", False):
                self.channelRelink.addItem(layer.name, registry.source_id(layer))
        self.channelRelink.blockSignals(False)
        self._filter_channels(self.channelSearch.text())

    def _editable_channels(self):
        channels = {
            key: dict(value) for key, value in (self.object.channels or {}).items()
        }
        if self.object.channels is None and self.object.source_id:
            settings = {
                "source_id": self.object.source_id,
                "source_name": self.object.source,
                "enabled": True,
                "source_level": self.object.source_level,
            }
            managed = LayerRegistry(self.scene_widget.viewer).find_managed(
                self.object.id, "section_3d"
            )
            source = self.object.source_layer(self.scene_widget.viewer)
            if managed is not None and source is not None and (
                OrthoSlice._colormap_spec(managed.colormap)
                != OrthoSlice._colormap_spec(source.colormap)
                or not np.array_equal(managed.contrast_limits, source.contrast_limits)
                or managed.gamma != source.gamma
                or managed.opacity != source.opacity
            ):
                settings["appearance_mode"] = "custom"
                settings["appearance"] = {
                    "colormap": OrthoSlice._colormap_spec(managed.colormap),
                    "contrast_limits": list(managed.contrast_limits),
                    "gamma": float(managed.gamma),
                    "opacity": float(managed.opacity),
                }
            channels[self.object.source_id] = settings
        return channels

    def _read_channel_appearance(self, item, *args):
        source_id = item.data(Qt.UserRole) if item is not None else None
        source = LayerRegistry(self.scene_widget.viewer).find(source_id)
        enabled = source is not None and self.object is not None
        self.channelRelinkButton.setEnabled(
            source is None and item is not None and self.channelRelink.count() > 0
        )
        settings = (
            (self.object.channels or {}).get(source_id, {})
            if self.object is not None else {}
        )
        custom = settings.get("appearance") or {}
        mode = settings.get("appearance_mode", "source")
        for widget in (
            self.channelAppearance, self.channelColor, self.channelOpacity,
            self.channelGamma, self.channelContrastMin, self.channelContrastMax,
        ):
            widget.blockSignals(True)
        self.channelAppearance.setCurrentIndex(1 if mode == "custom" else 0)
        if source is not None:
            colormap = custom.get("colormap", source.colormap.name)
            if isinstance(colormap, dict):
                colormap = colormap.get("name", "custom")
            if self.channelColor.findText(colormap) < 0:
                self.channelColor.addItem(colormap)
            self.channelColor.setCurrentText(colormap)
            self.channelOpacity.setValue(float(custom.get("opacity", source.opacity)))
            self.channelGamma.setValue(float(custom.get("gamma", source.gamma)))
            limits = custom.get("contrast_limits", source.contrast_limits)
            self.channelContrastMin.setValue(float(limits[0]))
            self.channelContrastMax.setValue(float(limits[1]))
        self.channelAppearance.setEnabled(enabled)
        custom_enabled = enabled and mode == "custom"
        for widget in (
            self.channelColor, self.channelOpacity, self.channelGamma,
            self.channelContrastMin, self.channelContrastMax,
        ):
            widget.setEnabled(custom_enabled)
            widget.blockSignals(False)
        self.channelAppearance.blockSignals(False)

    def _relink_missing_channel(self):
        item = self.channelList.currentItem()
        if self.object is None or item is None:
            return
        old_id = item.data(Qt.UserRole)
        registry = LayerRegistry(self.scene_widget.viewer)
        if registry.find(old_id) is not None:
            return
        new_id = self.channelRelink.currentData()
        source = registry.find(new_id)
        if source is None:
            return
        channels = self._editable_channels()
        if new_id in channels:
            return
        settings = channels.pop(old_id)
        settings["source_id"] = new_id
        settings["source_name"] = source.name
        channels[new_id] = settings
        changes = {"channels": channels}
        if self.object.source_id == old_id:
            changes.update(source_id=new_id, source=source.name)
        self.scene_widget.animation.update_scene_object(
            self.object.id, **changes
        )
        self._refresh_channels(self.object)

    def _channel_appearance_changed(self, *args):
        item = self.channelList.currentItem()
        if self.object is None or item is None:
            return
        source_id = item.data(Qt.UserRole)
        source = LayerRegistry(self.scene_widget.viewer).find(source_id)
        if source is None:
            return
        channels = self._editable_channels()
        settings = channels.setdefault(source_id, {
            "source_id": source_id,
            "source_name": source.name,
            "enabled": item.checkState() == Qt.Checked,
            "source_level": self.object.source_level,
        })
        custom = self.channelAppearance.currentIndex() == 1
        settings["appearance_mode"] = "custom" if custom else "source"
        old_colormap = (settings.get("appearance") or {}).get("colormap")
        colormap = self.channelColor.currentText()
        if isinstance(old_colormap, dict) and old_colormap.get("name") == colormap:
            colormap = old_colormap
        low = self.channelContrastMin.value()
        high = max(self.channelContrastMax.value(), low + 1e-6)
        settings["appearance"] = {
            "colormap": colormap,
            "opacity": self.channelOpacity.value(),
            "gamma": self.channelGamma.value(),
            "contrast_limits": [low, high],
        } if custom else None
        self.scene_widget.animation.update_scene_object(
            self.object.id, channels=channels
        )
        self._read_channel_appearance(item)

    def _filter_channels(self, text):
        query = text.casefold()
        for index in range(self.channelList.count()):
            item = self.channelList.item(index)
            item.setHidden(query not in item.text().casefold())

    def _set_visible_channels(self, enabled):
        if self.object is None:
            return
        registry = LayerRegistry(self.scene_widget.viewer)
        channels = self._editable_channels()
        changed = False
        for index in range(self.channelList.count()):
            item = self.channelList.item(index)
            if item.isHidden():
                continue
            source_id = item.data(Qt.UserRole)
            source = registry.find(source_id)
            if source is None:
                continue
            settings = channels.setdefault(source_id, {
                "source_id": source_id,
                "source_name": source.name,
                "source_level": self.object.source_level,
            })
            if settings.get("enabled") != enabled:
                settings["enabled"] = enabled
                changed = True
        if changed:
            self.scene_widget.animation.update_scene_object(
                self.object.id, channels=channels
            )
            self._refresh_channels(self.object)
            self._refresh_z_slider()

    def _channel_toggled(self, item):
        if self.object is None:
            return
        registry = LayerRegistry(self.scene_widget.viewer)
        channels = self._editable_channels()
        source_id = item.data(Qt.UserRole)
        source = registry.find(source_id)
        if source is None:
            return
        settings = channels.setdefault(source_id, {
            "source_id": source_id,
            "source_name": source.name,
            "source_level": self.object.source_level,
        })
        settings["enabled"] = item.checkState() == Qt.Checked
        self.scene_widget.animation.update_scene_object(
            self.object.id, channels=channels
        )
        self._refresh_z_slider()

    def _refresh_z_slider(self):
        """Use the selected source volume's world-space Z extent."""
        source = self.sourceComboBox.currentText()
        layer = (
            self.scene_widget.viewer.layers[source]
            if source in self.scene_widget.viewer.layers else None
        )
        bounds = None
        if layer is not None and getattr(layer, "ndim", 0) >= 3:
            lower, upper = sorted(float(v) for v in layer.extent.world[:, -3])
            if math.isfinite(lower) and math.isfinite(upper):
                bounds = (lower, upper)
        if self.object is not None and self.object.channels is not None:
            extents = []
            for source in self.object.source_layers(
                self.scene_widget.viewer, include_disabled=True
            ):
                extent = sorted(float(v) for v in source.extent.world[:, -3])
                if all(math.isfinite(value) for value in extent):
                    extents.append(extent)
            if extents:
                bounds = min(extent[0] for extent in extents), max(
                    extent[1] for extent in extents
                )
        self._z_slider_bounds = bounds or (0.0, 0.0)
        lower, upper = self._z_slider_bounds
        self.centreZSlider.setEnabled(bounds is not None and upper > lower)
        self._sync_z_slider(self.positionBoxes[0].value())

    def _sync_z_slider(self, z):
        lower, upper = self._z_slider_bounds
        if upper <= lower:
            return
        fraction = (z - lower) / (upper - lower)
        value = round(max(0.0, min(1.0, fraction)) * self.centreZSlider.maximum())
        self.centreZSlider.blockSignals(True)
        self.centreZSlider.setValue(value)
        self.centreZSlider.blockSignals(False)

    def _z_slider_changed(self, value):
        lower, upper = self._z_slider_bounds
        if upper > lower:
            z = lower + (upper - lower) * value / self.centreZSlider.maximum()
            self.positionBoxes[0].setValue(z)

    def _refresh_levels(self, selected=None):
        source = self.sourceComboBox.currentText()
        layer = (
            self.scene_widget.viewer.layers[source]
            if source in self.scene_widget.viewer.layers else None
        )
        if selected is None:
            selected = self.sourceLevelComboBox.currentData()
        self.sourceLevelComboBox.blockSignals(True)
        self.sourceLevelComboBox.clear()
        if layer is not None and getattr(layer, "multiscale", False):
            self.sourceLevelComboBox.addItem("Coarsest (fast)", -1)
            for level in range(len(layer.data)):
                label = f"Level {level}"
                if level == 0:
                    label += " (finest)"
                self.sourceLevelComboBox.addItem(label, level)
        else:
            self.sourceLevelComboBox.addItem("Native", 0)
        index = self.sourceLevelComboBox.findData(selected)
        if index < 0:
            index = self.sourceLevelComboBox.findData(0)
        self.sourceLevelComboBox.setCurrentIndex(max(0, index))
        self.sourceLevelComboBox.blockSignals(False)

    def _source_chosen(self, *args):
        self._refresh_levels()
        self._refresh_z_slider()
        self._changed()

    def _resolution_changed(self, *args):
        if self.object is not None:
            self.scene_widget.animation.resolution.set_section_level(
                self.object.id, self.sourceLevelComboBox.currentData()
            )

    def write(self):
        self.object.position = _get_vector(self.positionBoxes)
        self.object.normal = _get_vector(self.normalBoxes)
        self.object.thickness = self.thicknessSpinBox.value()
        self.object.projection = self.projectionComboBox.currentText()
        self.object.render_mode = self.renderComboBox.currentText()
        self.object.source_level = int(self.sourceLevelComboBox.currentData())
        if self.object.channels is not None:
            return
        source = self.sourceComboBox.currentText()
        self.object.source = source or None
        source_layer = (
            self.scene_widget.viewer.layers[source]
            if source and source in self.scene_widget.viewer.layers
            else None
        )
        self.object.source_id = (
            LayerRegistry(self.scene_widget.viewer).source_id(source_layer)
            if source_layer is not None
            else None
        )

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
        layer = LayerRegistry(viewer).find_managed(self.object.id, "section_3d")
        if layer is None and self.object.channels is not None:
            for child in self.object._channel_sections(viewer):
                layer = LayerRegistry(viewer).find_managed(
                    child.id, "section_3d"
                )
                if layer is not None:
                    break
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
        # keep the list compact: it sits inside a scrolling panel, and a tall
        # object list would push the editor below it out of view
        self.listWidget.setMaximumHeight(140)

        self.addPlaneButton = QPushButton("+ Plane")
        self.addPlaneButton.setToolTip("Add a clipping plane")
        self.addSliceButton = QPushButton("+ Slice")
        self.addSliceButton.setToolTip("Add an ortho slice")
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
        self.animation.events.state_applied.connect(self._state_applied)

        self.refresh()

    # ---------------------------------------------------------------- state

    def apply(self):
        """Push the live scene onto the viewer."""
        ortho = (
            self.animation.ortho_slicer.to_dict()
            if self.animation.ortho_slicer.enabled else None
        )
        self.animation.renderer.apply_working(ortho=ortho)

    def refresh(self):
        """Rebuild the object list from the scene."""
        previous = self.selected_object()
        selected_id = previous.id if previous is not None else None
        selected_id = self.animation.active_section_id or selected_id
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
            if scene_object.id == selected_id:
                self.listWidget.setCurrentRow(self.listWidget.count() - 1)
        self.listWidget.blockSignals(False)
        self._selection_changed(self.listWidget.currentRow(), activate=False)

    def _state_applied(self, event=None):
        self.refresh()

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

    def _selection_changed(self, row, *, activate=True):
        scene_object = self.selected_object()
        self.removeButton.setEnabled(scene_object is not None)
        if isinstance(scene_object, OrthoSlice):
            self.orthoSliceEditor.read(scene_object)
            self.editorStack.setCurrentWidget(self.orthoSliceEditor)
            if activate and self.animation.active_section_id != scene_object.id:
                self.animation.set_active_section(scene_object.id)
        elif isinstance(scene_object, ClipPlane):
            self.clipPlaneEditor.read(scene_object)
            self.editorStack.setCurrentWidget(self.clipPlaneEditor)
        else:
            self.orthoSliceEditor.object = None
            self.clipPlaneEditor.object = None
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
            registry = LayerRegistry(self.viewer)
            layer = registry.find_managed(scene_object.id, "section_3d")
            if layer is not None:
                layer.name = registry.unique_name(name)

        scene_object.enabled = enabled
        scene_object.name = name
        self.apply()
