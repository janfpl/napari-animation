"""Folder import controls for processed Luxendo HDF5 channels."""

from __future__ import annotations

import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from qtpy.QtCore import QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout,
)


class LuxendoImportWidget(QGroupBox):
    imported = Signal(object)
    _prepared = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__("Import Luxendo folder", parent=parent)
        self.animation = parent.animation
        self.viewer = self.animation.viewer
        self._pool = None
        self._future = None
        self._closed = False
        self._busy = False
        self._cancel = threading.Event()
        self._pending = deque()
        self._added = []
        self._skipped = []
        self.pathEdit = QLineEdit()
        self.pathEdit.setPlaceholderText("Processed Luxendo data folder")
        self.pathEdit.setMinimumWidth(80)
        self.browseButton = QPushButton("Browse…")
        self.recursiveCheckBox = QCheckBox("Include subfolders")
        self.recursiveCheckBox.setChecked(True)
        self.importButton = QPushButton("Import channels")
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setEnabled(False)
        self.statusLabel = QLabel("Voxel sizes are read from each image's metadata (µm).")
        self.statusLabel.setWordWrap(True)
        layout = QVBoxLayout(self)
        path_row = QHBoxLayout()
        path_row.addWidget(self.pathEdit, 1)
        path_row.addWidget(self.browseButton)
        layout.addLayout(path_row)
        layout.addWidget(self.recursiveCheckBox)
        buttons = QHBoxLayout()
        buttons.addWidget(self.importButton)
        buttons.addWidget(self.cancelButton)
        layout.addLayout(buttons)
        layout.addWidget(self.statusLabel)
        self.browseButton.clicked.connect(self._browse)
        self.importButton.clicked.connect(self.start_import)
        self.pathEdit.returnPressed.connect(self.start_import)
        self.cancelButton.clicked.connect(self._cancel_import)
        self._prepared.connect(self._install_plan)
        self.destroyed.connect(self.shutdown)
        qt_window = getattr(getattr(self.viewer, "window", None), "_qt_window", None)
        if qt_window is not None:
            qt_window.destroyed.connect(self.shutdown)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select processed Luxendo folder", self.pathEdit.text(),
        )
        if path:
            self.pathEdit.setText(path)

    def _set_busy(self, busy):
        self._busy = busy
        for widget in (self.pathEdit, self.browseButton, self.recursiveCheckBox, self.importButton):
            widget.setEnabled(not busy)
        self.cancelButton.setEnabled(busy)

    def start_import(self):
        if self._closed or self._busy:
            return
        path = self.pathEdit.text().strip().strip('"')
        if not path:
            self.statusLabel.setText("Choose a processed Luxendo folder first.")
            return
        from ..luxendo import prepare_luxendo_folder

        self._cancel = threading.Event()
        self._added = []
        self._skipped = []
        self._set_busy(True)
        self.statusLabel.setText("Reading channel metadata…")
        self.statusLabel.setToolTip("")
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="luxendo-import")
        self._future = self._pool.submit(
            prepare_luxendo_folder, path, self.recursiveCheckBox.isChecked(), self._cancel,
        )

        def ready(future):
            try:
                plan, error = future.result(), None
            except Exception as error_result:
                plan, error = None, error_result
            if not self._closed:
                try:
                    self._prepared.emit(plan, error)
                except RuntimeError:  # parent window was destroyed
                    pass

        self._future.add_done_callback(ready)

    @Slot(object, object)
    def _install_plan(self, plan, error):
        if self._closed:
            return
        self._future = None
        if self._cancel.is_set():
            self._finish("Import cancelled.")
            return
        if error is not None:
            self._finish(f"Import failed: {error}")
            return
        from ..luxendo import loaded_luxendo_paths

        existing = loaded_luxendo_paths(self.viewer)
        self._pending = deque(
            channel for channel in plan.channels
            if os.path.normcase(str(channel.path)) not in existing
        )
        self._already_loaded = len(plan.channels) - len(self._pending)
        self._total = len(self._pending)
        self._skipped = plan.skipped
        QTimer.singleShot(0, self._add_next)

    def _add_next(self):
        if self._closed:
            return
        if self._cancel.is_set():
            self._pending.clear()
            self._finish(f"Import cancelled; kept {len(self._added)} imported channel(s).")
            return
        if not self._pending:
            message = f"Imported {len(self._added)} channel(s), with voxel sizes in µm."
            if self._already_loaded:
                message += f" {self._already_loaded} already loaded."
            if self._skipped:
                message += f" Skipped {len(self._skipped)} file(s):\n" + "\n".join(self._skipped[:3])
            self._finish(message)
            self.statusLabel.setToolTip("\n".join(self._skipped))
            return
        from ..luxendo import add_luxendo_channel

        channel = self._pending.popleft()
        self.statusLabel.setText(f"Adding channel {len(self._added) + 1} of {self._total}: {channel.name}")
        try:
            self._added.append(add_luxendo_channel(self.viewer, channel, len(self._added)))
        except Exception as error:
            self._pending.clear()
            self._finish(f"Import stopped after {len(self._added)} channel(s): {error}")
            return
        QTimer.singleShot(0, self._add_next)

    def _finish(self, message):
        self._set_busy(False)
        self.statusLabel.setText(message)
        self.statusLabel.setToolTip(message)
        layers, self._added = self._added, []
        if layers:
            self.imported.emit(layers)

    def _cancel_import(self):
        self._cancel.set()
        self.statusLabel.setText("Cancelling import…")

    def shutdown(self):
        self._closed = True
        self._cancel.set()
        self._future = None
        self._pending.clear()
        self._added = []
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
