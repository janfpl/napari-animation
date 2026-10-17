"""Pick, apply and save preset routines (:mod:`napari_animation.routines`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from .. import routines

if TYPE_CHECKING:
    from ..animation import Animation


class RoutinesWidget(QGroupBox):
    """List built-in and saved routines; apply one or save the timeline."""

    def __init__(self, parent=None):
        super().__init__("Routines", parent=parent)
        self.animation: Animation = self.parentWidget().animation
        self.viewer = self.animation.viewer
        self._routines = {}

        self.routineComboBox = QComboBox()
        self.routineComboBox.setToolTip(
            "Built-in routines, then routines saved in "
            f"{routines.routines_dir()}"
        )
        self.descriptionLabel = QLabel()
        self.descriptionLabel.setWordWrap(True)
        self.thicknessSpinBox = QDoubleSpinBox()
        self.thicknessSpinBox.setRange(1.0, 100.0)
        self.thicknessSpinBox.setDecimals(1)
        self.thicknessSpinBox.setSuffix(" %")
        self.thicknessSpinBox.setValue(10.0)
        self.thicknessSpinBox.setToolTip(
            "Thickness of every slab, as a percentage of the data extent "
            "along the slab's normal"
        )
        self.passSpinBox = QDoubleSpinBox()
        self.passSpinBox.setRange(0.5, 600.0)
        self.passSpinBox.setDecimals(1)
        self.passSpinBox.setSuffix(" s")
        self.passSpinBox.setValue(10.0)
        self.passSpinBox.setToolTip(
            "Duration of one end-to-end pass of a slab (built-in routines; "
            "export at 24 fps for these timings)"
        )
        self.applyButton = QPushButton("Apply to current data")
        self.applyButton.setToolTip(
            "Replace the keyframes and scene objects with this routine, "
            "sized to the loaded 3D image layers"
        )
        self.saveButton = QPushButton("Save keyframes as routine…")
        self.saveButton.setToolTip(
            "Store the current keyframes as a routine that can be applied "
            "to other data (ortho slice positions are saved relative to the "
            "data size)"
        )

        options = QFormLayout()
        options.addRow("Slab thickness", self.thicknessSpinBox)
        options.addRow("Pass duration", self.passSpinBox)

        layout = QVBoxLayout()
        layout.addWidget(self.routineComboBox)
        layout.addWidget(self.descriptionLabel)
        layout.addLayout(options)
        layout.addWidget(self.applyButton)
        layout.addWidget(self.saveButton)
        self.setLayout(layout)

        self.routineComboBox.currentIndexChanged.connect(self._describe)
        self.passSpinBox.valueChanged.connect(self._describe)
        self.applyButton.clicked.connect(self._apply)
        self.saveButton.clicked.connect(self._save)
        self.refresh()

    def refresh(self, select=None):
        self._routines = routines.available_routines()
        current = select or self.routineComboBox.currentText()
        self.routineComboBox.blockSignals(True)
        self.routineComboBox.clear()
        self.routineComboBox.addItems(list(self._routines))
        index = self.routineComboBox.findText(current)
        self.routineComboBox.setCurrentIndex(max(0, index))
        self.routineComboBox.blockSignals(False)
        self._describe()

    def selected_routine(self):
        """The selected routine, built-ins generated with the panel timing."""
        name = self.routineComboBox.currentText()
        make = routines.BUILTIN_ROUTINES.get(name)
        if make is not None and self._routines.get(name) is not None:
            return make(pass_seconds=self.passSpinBox.value())
        return self._routines.get(name)

    def thickness(self) -> float:
        return self.thicknessSpinBox.value() / 100.0

    def _describe(self, *args):
        routine = self.selected_routine()
        if routine is None:
            self.descriptionLabel.setText("")
            return
        builtin = self.routineComboBox.currentText() in (
            routines.BUILTIN_ROUTINES
        )
        self.passSpinBox.setEnabled(builtin)
        sections = routine.get("sections", [])
        if not builtin and sections and not args[1:]:
            # a saved routine starts from the thickness it was saved with
            self.thicknessSpinBox.setValue(
                100.0 * float(sections[0].get("thickness", 0.1))
            )
        text = routine.get("description", "")
        summary = routines.routine_summary(routine)
        self.descriptionLabel.setText(f"{text}\n{summary}".strip())

    def _confirm(self, title, text) -> bool:
        answer = QMessageBox.question(
            self, title, text, QMessageBox.Yes | QMessageBox.Cancel
        )
        return answer == QMessageBox.Yes

    def _apply(self, *args):
        routine = self.selected_routine()
        if routine is None:
            return
        if (len(self.animation.key_frames) or len(self.animation.scene)) and (
            not self._confirm(
                "Apply routine",
                "Applying a routine replaces the current keyframes and "
                "scene objects. Continue?",
            )
        ):
            return
        try:
            routines.apply_routine(
                self.animation, routine, thickness=self.thickness()
            )
        except ValueError as error:
            QMessageBox.warning(self, "Apply routine", str(error))

    def _save(self, *args):
        name, ok = QInputDialog.getText(self, "Save routine", "Routine name:")
        name = name.strip()
        if not ok or not name:
            return
        try:
            routine = routines.routine_from_keyframes(self.animation, name)
            path = routines.save_routine(routine)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Save routine", str(error))
            return
        self.refresh(select=name)
        self.descriptionLabel.setText(
            f"{self.descriptionLabel.text()}\nSaved to {path}"
        )
