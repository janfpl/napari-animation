"""Pick, apply and save preset routines (:mod:`napari_animation.routines`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QComboBox,
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

        layout = QVBoxLayout()
        layout.addWidget(self.routineComboBox)
        layout.addWidget(self.descriptionLabel)
        layout.addWidget(self.applyButton)
        layout.addWidget(self.saveButton)
        self.setLayout(layout)

        self.routineComboBox.currentIndexChanged.connect(self._describe)
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
        return self._routines.get(self.routineComboBox.currentText())

    def _describe(self, *args):
        routine = self.selected_routine()
        if routine is None:
            self.descriptionLabel.setText("")
            return
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
            routines.apply_routine(self.animation, routine)
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
