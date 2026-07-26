"""View snapping and scroll-through generation.

Two motions that are tedious to keyframe by hand:

* walking the dims slider through a stack, and
* sliding an ortho slice through a volume along its own normal.

Both produce a pair of keyframes that differ in exactly one thing, so whatever
else is set up -- camera, clipping planes, layer settings -- is carried through
untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..ortho_slicer import ORTHO_VIEWS
from ..scene import OrthoSlice

if TYPE_CHECKING:
    from ..animation import Animation


class ScrollThroughWidget(QGroupBox):
    """Snap between 2D and 3D, and generate scroll-through keyframes."""

    def __init__(self, parent=None):
        super().__init__("View and scroll-through", parent=parent)
        self.animation: Animation = self.parentWidget().animation
        self.viewer = self.animation.viewer
        # until the user picks an axis themselves, follow the slider axis so
        # that snapping to a different view scrolls the axis you just exposed
        self._axis_pinned = False

        # -- view snapping --
        self.view3dButton = QPushButton("3D")
        self.view3dButton.setToolTip("Snap to the 3D volume view")
        # four short buttons in a row that cannot wrap; without an explicit
        # minimum, Qt's default button width forces the whole dock wider
        self.view3dButton.setMinimumWidth(36)
        self.view3dButton.setMaximumWidth(80)
        self.viewButtons = {}
        view_row = QWidget()
        view_row.setLayout(QHBoxLayout())
        view_row.layout().setContentsMargins(0, 0, 0, 0)
        view_row.layout().addWidget(self.view3dButton)
        for view in ORTHO_VIEWS:
            button = QPushButton(view)
            button.setToolTip(f"Snap to a flat 2D {view} view")
            button.setMinimumWidth(36)
            button.setMaximumWidth(80)
            self.viewButtons[view] = button
            view_row.layout().addWidget(button)

        # -- dims scroll-through --
        self.axisComboBox = QComboBox()
        self.axisComboBox.setToolTip("Axis to scroll along")

        self.startSpinBox = QDoubleSpinBox()
        self.stopSpinBox = QDoubleSpinBox()
        for spin_box in (self.startSpinBox, self.stopSpinBox):
            spin_box.setDecimals(3)
            spin_box.setRange(-1e6, 1e6)
            spin_box.setMinimumWidth(70)
            spin_box.setMaximumWidth(120)
        range_row = QWidget()
        range_row.setLayout(QHBoxLayout())
        range_row.layout().setContentsMargins(0, 0, 0, 0)
        range_row.layout().addWidget(QLabel("from"))
        range_row.layout().addWidget(self.startSpinBox)
        range_row.layout().addWidget(QLabel("to"))
        range_row.layout().addWidget(self.stopSpinBox)

        self.stepsSpinBox = QSpinBox()
        self.stepsSpinBox.setRange(2, 100000)
        self.stepsSpinBox.setValue(30)
        self.stepsSpinBox.setToolTip("Frames the sweep takes")

        self.scrollButton = QPushButton("Add scroll-through")
        self.scrollButton.setToolTip(
            "Two keyframes that differ only in the dims slider"
        )

        # -- ortho slice sweep --
        self.sliceComboBox = QComboBox()
        self.sliceComboBox.setToolTip(
            "Ortho slice to slide through the volume along its own normal"
        )
        self.sweepButton = QPushButton("Add slice sweep")
        self.sweepButton.setToolTip(
            "Two keyframes that sweep the slice across the whole volume"
        )

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.addRow("Snap view", view_row)
        form.addRow("Scroll axis", self.axisComboBox)
        form.addRow("Range", range_row)
        form.addRow("Steps", self.stepsSpinBox)
        form.addRow(self.scrollButton)
        form.addRow("Slice", self.sliceComboBox)
        form.addRow(self.sweepButton)

        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(form)
        layout.addWidget(container)
        self.setLayout(layout)

        self.view3dButton.clicked.connect(self._snap_3d)
        for view, button in self.viewButtons.items():
            button.clicked.connect(
                lambda *args, view=view: self._snap_2d(view)
            )
        self.axisComboBox.currentIndexChanged.connect(self._load_axis_range)
        # `activated` fires only for a real user selection, not for the
        # programmatic repopulation in _refresh_axes
        self.axisComboBox.activated.connect(self._pin_axis)
        self.scrollButton.clicked.connect(self._add_scroll_through)
        self.sweepButton.clicked.connect(self._add_slice_sweep)

        self.viewer.dims.events.ndisplay.connect(self.refresh)
        self.viewer.dims.events.order.connect(self.refresh)
        self.viewer.layers.events.inserted.connect(self.refresh)
        self.viewer.layers.events.removed.connect(self.refresh)

        self.refresh()

    # ------------------------------------------------------------- refresh

    def refresh(self, event=None):
        """Repopulate the axis and slice choices."""
        self._refresh_axes()
        self._refresh_slices()

    def _refresh_axes(self):
        dims = self.viewer.dims
        current = self.axisComboBox.currentData()
        self.axisComboBox.blockSignals(True)
        self.axisComboBox.clear()
        labels = [str(label) for label in dims.axis_labels]
        for axis in range(dims.ndim):
            self.axisComboBox.addItem(f"{axis}: {labels[axis]}", axis)
        # default to the current slider axis, which is what a user scrolling
        # by hand would be moving
        not_displayed = tuple(dims.not_displayed)
        default = (
            int(not_displayed[0]) if not_displayed else max(0, dims.ndim - 3)
        )
        wanted = (
            current
            if (self._axis_pinned and current is not None)
            else (default)
        )
        index = self.axisComboBox.findData(wanted)
        self.axisComboBox.setCurrentIndex(max(0, index))
        self.axisComboBox.blockSignals(False)
        self._load_axis_range()

    def _pin_axis(self, *args):
        """Stop following the slider axis once the user has chosen one."""
        self._axis_pinned = True

    def _refresh_slices(self):
        current = self.sliceComboBox.currentData()
        self.sliceComboBox.blockSignals(True)
        self.sliceComboBox.clear()
        for scene_object in self.animation.scene:
            if isinstance(scene_object, OrthoSlice):
                self.sliceComboBox.addItem(scene_object.name, scene_object.id)
        index = self.sliceComboBox.findData(current)
        if index >= 0:
            self.sliceComboBox.setCurrentIndex(index)
        self.sliceComboBox.blockSignals(False)
        self.sweepButton.setEnabled(self.sliceComboBox.count() > 0)

    def _load_axis_range(self, *args):
        """Fill the range boxes with the full extent of the chosen axis."""
        axis = self.axisComboBox.currentData()
        if axis is None:
            return
        try:
            start, stop = self.animation._axis_limits(int(axis))
        except (IndexError, TypeError):
            return
        for spin_box, value in (
            (self.startSpinBox, start),
            (self.stopSpinBox, stop),
        ):
            spin_box.blockSignals(True)
            spin_box.setValue(value)
            spin_box.blockSignals(False)

    # -------------------------------------------------------------- actions

    def _snap_2d(self, view):
        self.animation.snap_2d(view, capture=False)
        self.refresh()

    def _snap_3d(self, *args):
        self.animation.snap_3d(capture=False)
        self.refresh()

    def _add_scroll_through(self, *args):
        axis = self.axisComboBox.currentData()
        if axis is None:
            return
        self.animation.add_scroll_through(
            axis=int(axis),
            start=self.startSpinBox.value(),
            stop=self.stopSpinBox.value(),
            steps=self.stepsSpinBox.value(),
        )

    def _selected_slice(self):
        object_id = self.sliceComboBox.currentData()
        for scene_object in self.animation.scene:
            if scene_object.id == object_id:
                return scene_object
        return None

    def _add_slice_sweep(self, *args):
        scene_object = self._selected_slice()
        if scene_object is None:
            return
        self.animation.add_slice_sweep(
            scene_object, steps=self.stepsSpinBox.value()
        )
