from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledSlider

from ..export_estimates import estimate_export, format_size_range


# Width, height. All preset dimensions are even for yuv420p video encoding.
RESOLUTION_PRESETS = (
    ("1280 × 720 (16:9 HD)", (1280, 720)),
    ("1920 × 1080 (16:9 Full HD)", (1920, 1080)),
    ("2560 × 1440 (16:9 QHD)", (2560, 1440)),
    ("3840 × 2160 (16:9 4K)", (3840, 2160)),
    ("1024 × 768 (4:3)", (1024, 768)),
    ("1600 × 1200 (4:3)", (1600, 1200)),
    ("2048 × 1536 (4:3)", (2048, 1536)),
)


class SaveDialogWidget(QFileDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getAnimationParameters(
        self,
        parent=None,
        caption="Select a file :",
        dir=".",
        filter="All files (*.*)",
        selectedFilter="",
        options=None,
        animation=None,
    ):
        """Prompt for a destination and render settings.

        Returns
        -------
        dict or None
            The keyword arguments for :meth:`Animation.animate`, or ``None``
            if the user cancelled the dialog.
        """

        # Set dialog parameters
        self.setWindowTitle(caption)
        self.setDirectory(dir)
        self.setNameFilter(filter)
        self.setFileMode(QFileDialog.AnyFile)
        self.setAcceptMode(QFileDialog.AcceptSave)

        if selectedFilter != "":
            self.selectNameFilter(selectedFilter)
        if options is not None:
            self.setOptions(options)

        # add OptionsWidget
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        layout = self.layout()
        self.optionsWidget = OptionsWidget(self, animation=animation)

        self.filterSelected.connect(self._update_estimate_format)
        self.currentChanged.connect(self._update_estimate_format)
        filename = self.findChild(QLineEdit, "fileNameEdit")
        if filename is not None:
            filename.textChanged.connect(self._update_estimate_format)
        self._update_estimate_format()

        layout.addWidget(self.optionsWidget, 4, 1)
        self.setLayout(layout)

        # Get info back from user
        if self.exec_():
            animation_kwargs = {}

            selected_path = list(self.selectedFiles())[0]
            # If the user didn't type an extension, append the one implied by
            # the chosen format filter (e.g. "MP4 (*.mp4)" -> ".mp4"). Without
            # this, a bare name silently saves a folder of PNGs instead of the
            # requested video.
            animation_kwargs["path"] = self._ensure_extension(selected_path)
            animation_kwargs["fps"] = self.optionsWidget.fpsSpinBox.value()
            animation_kwargs["quality"] = int(
                self.optionsWidget.qualitySlider.value()
            )
            animation_kwargs["canvas_only"] = (
                self.optionsWidget.canvasCheckBox.isChecked()
            )
            preset = self.optionsWidget.resolutionComboBox.currentData()
            if preset is not None:
                animation_kwargs["render_size"] = (preset[1], preset[0])

            return animation_kwargs

        # cancelled: None rather than "", so a caller that reaches for a key
        # gets a clear AttributeError instead of "string indices must be
        # integers" from indexing into an empty string.
        return None

    def _update_estimate_format(self, *_):
        filename = self.findChild(QLineEdit, "fileNameEdit")
        typed = filename.text() if filename is not None else ""
        paths = self.selectedFiles()
        path = paths[0] if typed and paths else "animation"
        suffix = Path(self._ensure_extension(path)).suffix.lower()
        self.optionsWidget.set_output_suffix(suffix)

    def _ensure_extension(self, path):
        """Append the selected filter's extension if ``path`` has none."""
        import re

        if Path(path).suffix:
            return path
        # selectedNameFilter() looks like "MP4 (*.mp4)" or "Folder of PNGs (*)"
        match = re.search(r"\*(\.[A-Za-z0-9]+)", self.selectedNameFilter())
        if match:
            return path + match.group(1)
        return path


class OptionsWidget(QWidget):
    def __init__(self, *args, animation=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._animation = animation
        self._frame_sizes = {}
        self._suffix = ".mp4"

        layout = QHBoxLayout()

        # "Canvas only" check box
        self.canvasCheckBox = QCheckBox("Canvas only", self)
        self.canvasCheckBox.toggle()
        layout.addWidget(self.canvasCheckBox)

        # Quality list
        self.qualitySlider = QLabeledSlider(Qt.Horizontal, self)
        self.qualitySlider.setRange(1, 10)
        self.qualitySlider.setValue(5)
        self.qualitySlider.setSingleStep(1)
        self.qualitySlider.setTickInterval(1)
        self.qualitySlider.setTickPosition(QSlider.TicksBelow)

        self.qualitySlider._slider.setMinimumWidth(70)
        self.qualitySlider._slider.setMaximumWidth(210)

        self.qualitySlider._label.setAlignment(Qt.AlignCenter)
        self.qualitySlider._label.setStyleSheet(
            "SliderLabel {background:transparent; border: 0; min-width: 20px; max-width: 20px;}"
            "SliderLabel::up-button, SliderLabel::down-button {subcontrol-origin: margin; width: 0px; height: 0px; }"
        )

        quality_label = self.qualityLabel = QLabel("Quality", self)
        quality_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(quality_label)
        layout.addWidget(self.qualitySlider)

        # fps
        self.fpsSpinBox = QSpinBox(self)
        self.fpsSpinBox.setRange(1, 100000)
        self.fpsSpinBox.setValue(20)
        fps_label = QLabel("FPS", self)
        fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(fps_label)
        layout.addWidget(self.fpsSpinBox)

        self.resolutionComboBox = QComboBox(self)
        self.resolutionComboBox.addItem("Current canvas", None)
        for label, size in RESOLUTION_PRESETS:
            self.resolutionComboBox.addItem(label, size)
        resolution_label = QLabel("Render size", self)
        resolution_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        render_layout = QHBoxLayout()
        render_layout.addWidget(resolution_label)
        render_layout.addWidget(self.resolutionComboBox, 1)

        outer = QVBoxLayout()
        outer.addLayout(layout)
        outer.addLayout(render_layout)
        self.resolutionLabel = QLabel(self)
        self.estimateLabel = QLabel(self)
        self.estimateNote = QLabel(self)
        for label in (self.resolutionLabel, self.estimateLabel, self.estimateNote):
            label.setWordWrap(True)
            outer.addWidget(label)
        self.setLayout(outer)
        self.setMaximumWidth(700)
        self.qualitySlider.valueChanged.connect(self._update_estimates)
        self.fpsSpinBox.valueChanged.connect(self._update_estimates)
        self.resolutionComboBox.currentIndexChanged.connect(self._preset_changed)
        self.canvasCheckBox.toggled.connect(self._update_estimates)
        self._update_estimates()

    def set_output_suffix(self, suffix):
        self._suffix = suffix.lower()
        self._update_estimates()

    def _preset_changed(self, *_):
        if self.resolutionComboBox.currentData() is not None:
            self.canvasCheckBox.setChecked(True)
        self.canvasCheckBox.setEnabled(
            self.resolutionComboBox.currentData() is None
        )
        self._update_estimates()

    def _frame_size(self):
        canvas_only = self.canvasCheckBox.isChecked()
        if canvas_only not in self._frame_sizes:
            if self._animation is None:
                return None
            viewer = self._animation.viewer
            window = getattr(viewer, "window", None)
            widget = None
            if window is not None:
                try:
                    widget = (
                        window._qt_viewer.canvas.native
                        if canvas_only else window._qt_window
                    )
                except AttributeError:
                    pass
            if widget is not None:
                ratio = float(widget.devicePixelRatioF())
                width = int(round(widget.width() * ratio))
                height = int(round(widget.height() * ratio))
                if width > 0 and height > 0:
                    self._frame_sizes[canvas_only] = (width, height)
                    return self._frame_sizes[canvas_only]
            try:
                frame = viewer.screenshot(
                    canvas_only=canvas_only, flash=False
                )
            except Exception:
                return None
            self._frame_sizes[canvas_only] = (frame.shape[1], frame.shape[0])
        return self._frame_sizes[canvas_only]

    def _update_estimates(self, *_):
        preset = self.resolutionComboBox.currentData()
        frame_size = preset if preset is not None else self._frame_size()
        frames = len(self._animation._frames) if self._animation is not None else 0
        if frame_size is None:
            self.resolutionLabel.setText("Resolution unavailable: no viewer preview.")
            self.estimateLabel.setText("Estimated file size unavailable.")
            self.estimateNote.setText("Choose a render size; quality sets compression.")
            self.qualitySlider.setToolTip("")
            self.qualityLabel.setToolTip("")
            return
        estimates = [
            estimate_export(
                frame_size, frames, self.fpsSpinBox.value(),
                1, quality, self._suffix,
                video_block_size=2 if preset is not None else 16,
            )
            for quality in range(1, 11)
        ]
        quality = int(self.qualitySlider.value())
        selected = estimates[quality - 1]
        width, height = selected.resolution
        resolution = f"{width:,} × {height:,} px"
        self.resolutionLabel.setText(
            f"Output resolution: {resolution} · "
            f"{selected.duration:,.1f} s ({frames:,} frames)"
        )
        self.qualitySlider.setEnabled(selected.quality_applies)
        size = format_size_range(selected.size_range) if selected.size_range else "unavailable"
        if selected.quality_applies:
            self.estimateLabel.setText(f"Quality {quality}/10 · Estimated file size: {size}")
        elif self._suffix in {"", ".png", ".gif"}:
            label = "PNG folder total" if self._suffix == "" else "file size"
            self.estimateLabel.setText(f"Estimated {label}: {size} · Quality slider unused for this format.")
        else:
            self.estimateLabel.setText("File-size estimate unavailable for this format.")
        if width <= 0 or height <= 0:
            self.estimateNote.setText("The current canvas has an empty dimension.")
        else:
            self.estimateNote.setText(
                "Rough size range for the current view; motion and image detail can put the final size outside it. "
                "Presets rerender the canvas at the selected size; quality sets compression. "
                "Current-canvas video may get black padding to multiples of 16. "
                "Hover over Quality to compare all levels."
            )
        rows = "".join(
            f"<tr><td>{level}</td><td>{resolution}</td><td>"
            f"{format_size_range(estimate.size_range) if estimate.size_range else 'unavailable'}</td></tr>"
            for level, estimate in enumerate(estimates, 1)
        )
        self.qualitySlider.setToolTip(
            "<table cellspacing='6'><tr><th>Quality</th><th>Resolution</th>"
            f"<th>Approximate size</th></tr>{rows}</table>"
        )
        self.qualityLabel.setToolTip(self.qualitySlider.toolTip())
