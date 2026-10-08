"""Tests for the save-dialog extension handling (no GUI interaction)."""

from unittest.mock import patch

from types import SimpleNamespace

import numpy as np

from napari_animation._qt.savedialog_widget import OptionsWidget, SaveDialogWidget

FILTERS = "MP4 (*.mp4);;GIF (*.gif);;Folder of PNGs (*)"


def test_cancelling_returns_none(qtbot):
    """Cancelling must not return something a caller will index into.

    Returning "" meant `animation_kwargs["path"]` raised "string indices must
    be integers", so backing out of the save dialog looked like a crash.
    """
    dialog = SaveDialogWidget()
    qtbot.addWidget(dialog)

    # exec_() returns 0 when the dialog is rejected
    with patch.object(SaveDialogWidget, "exec_", return_value=0):
        result = dialog.getAnimationParameters(filter=FILTERS)

    assert result is None


def test_ensure_extension_appends_from_filter(qtbot):
    dialog = SaveDialogWidget()
    qtbot.addWidget(dialog)
    dialog.setNameFilter(FILTERS)

    dialog.selectNameFilter("MP4 (*.mp4)")
    assert dialog._ensure_extension("C:/tmp/test") == "C:/tmp/test.mp4"
    # an explicit extension is left untouched
    assert dialog._ensure_extension("C:/tmp/test.mov") == "C:/tmp/test.mov"

    dialog.selectNameFilter("GIF (*.gif)")
    assert dialog._ensure_extension("movie") == "movie.gif"


def test_ensure_extension_png_folder_filter_keeps_bare_name(qtbot):
    dialog = SaveDialogWidget()
    qtbot.addWidget(dialog)
    dialog.setNameFilter(FILTERS)
    # "Folder of PNGs (*)" has no concrete extension -> keep the bare name so
    # animate() writes a folder of PNGs.
    dialog.selectNameFilter("Folder of PNGs (*)")
    assert dialog._ensure_extension("C:/tmp/frames") == "C:/tmp/frames"


def _animation_context():
    calls = []

    def screenshot(canvas_only, flash):
        calls.append(canvas_only)
        shape = (600, 800, 4) if canvas_only else (900, 1200, 4)
        return np.zeros(shape, dtype=np.uint8)

    animation = SimpleNamespace(
        viewer=SimpleNamespace(screenshot=screenshot), _frames=range(300)
    )
    return animation, calls


def test_estimates_react_to_presets_without_recapturing_each_level(qtbot):
    animation, calls = _animation_context()
    widget = OptionsWidget(animation=animation)
    qtbot.addWidget(widget)
    assert "800 × 608" in widget.resolutionLabel.text()
    assert "15.0 s (300 frames)" in widget.resolutionLabel.text()
    before = widget.estimateLabel.text()
    widget.qualitySlider.setValue(10)
    assert "Quality 10/10" in widget.estimateLabel.text()
    assert before != widget.estimateLabel.text()
    assert "<td>1</td>" in widget.qualitySlider.toolTip()
    assert "<td>10</td>" in widget.qualitySlider.toolTip()
    widget.resolutionComboBox.setCurrentIndex(
        widget.resolutionComboBox.findText("1920 × 1080 (16:9 Full HD)")
    )
    assert "1,920 × 1,080" in widget.resolutionLabel.text()
    assert not widget.canvasCheckBox.isEnabled()
    widget.fpsSpinBox.setValue(30)
    assert "10.0 s" in widget.resolutionLabel.text()
    widget.resolutionComboBox.setCurrentIndex(0)
    assert widget.canvasCheckBox.isEnabled()
    widget.canvasCheckBox.setChecked(False)
    assert "1,200 × 912" in widget.resolutionLabel.text()
    widget.canvasCheckBox.setChecked(True)
    assert calls == [True, False]


def test_save_dialog_returns_selected_render_size(qtbot):
    animation, _ = _animation_context()
    dialog = SaveDialogWidget()
    qtbot.addWidget(dialog)
    def choose_preset(self):
        self.selectFile("movie.mp4")
        self.optionsWidget.resolutionComboBox.setCurrentIndex(
            self.optionsWidget.resolutionComboBox.findText("2048 × 1536 (4:3)")
        )
        return 1

    with patch.object(SaveDialogWidget, "exec_", choose_preset):
        options = dialog.getAnimationParameters(filter=FILTERS, animation=animation)
    assert options["render_size"] == (1536, 2048)
    assert options["canvas_only"] is True


def test_format_and_typed_extension_update_estimates(qtbot):
    animation, _ = _animation_context()
    dialog = SaveDialogWidget()
    qtbot.addWidget(dialog)
    with patch.object(SaveDialogWidget, "exec_", return_value=0):
        dialog.getAnimationParameters(filter=FILTERS, animation=animation)
    dialog.selectNameFilter("GIF (*.gif)")
    dialog._update_estimate_format()
    assert not dialog.optionsWidget.qualitySlider.isEnabled()
    assert "800 × 600" in dialog.optionsWidget.resolutionLabel.text()
    # A typed extension wins over the selected filter, as it does on export.
    from qtpy.QtWidgets import QLineEdit
    dialog.findChild(QLineEdit, "fileNameEdit").setText("movie.mp4")
    assert dialog.optionsWidget.qualitySlider.isEnabled()
    assert "800 × 608" in dialog.optionsWidget.resolutionLabel.text()
    dialog.findChild(QLineEdit, "fileNameEdit").clear()
    dialog.selectNameFilter("Folder of PNGs (*)")
    dialog._update_estimate_format()
    assert "PNG folder total" in dialog.optionsWidget.estimateLabel.text()
