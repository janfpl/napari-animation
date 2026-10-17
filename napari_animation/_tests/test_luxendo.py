"""Luxendo metadata, lazy file ownership, and asynchronous folder import."""

import gc
import json
import threading
import time

import dask.array as da
import h5py
import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtCore import QTimer

from napari_animation import (
    Animation,
    KeyFrame,
    ViewerState,
    import_luxendo_folder,
)
from napari_animation._qt import AnimationWidget
from napari_animation.luxendo import (
    LUXENDO_METADATA,
    ImportCancelled,
    add_luxendo_channel,
    prepare_luxendo_folder,
    read_luxendo_channel,
    scan_luxendo_files,
)
from napari_animation.scene.h5_process import describe_h5_volume


@pytest.fixture
def channel_file(tmp_path):
    def create(
        name="ch0.lux.h5",
        channel=0,
        scale=(0.7, 0.3, 0.2),
        pyramid=True,
        metadata=True,
        encoding="scalar",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        processing = dict(
            channel=channel,
            channel_description="561" if channel == 0 else "488",
            voxel_size_um=dict(zip(("depth", "height", "width"), scale)),
        )
        raw = json.dumps(dict(processingInformation=processing)).encode()
        with h5py.File(path, "w") as file:
            if metadata:
                if encoding == "uint8":
                    raw = np.frombuffer(raw, dtype=np.uint8)
                elif encoding == "chars":
                    raw = np.frombuffer(raw, dtype="S1")
                file["metadata"] = raw
            factors = (1, 2, 4, 8, 16) if pyramid else (1,)
            for factor in factors:
                shape = tuple(max(1, size // factor) for size in (16, 32, 48))
                data = np.arange(np.prod(shape), dtype=np.uint16).reshape(
                    shape
                )
                key = (
                    "Data"
                    if factor == 1
                    else f"Data_{factor}_{factor}_{factor}"
                )
                file.create_dataset(key, data=data, chunks=True)
        return path

    return create


@pytest.mark.parametrize("encoding", ["scalar", "uint8", "chars"])
def test_metadata_and_pyramids_stay_lazy(channel_file, monkeypatch, encoding):
    path = channel_file(encoding=encoding)
    reads = []
    original = h5py.Dataset.__getitem__

    def record(dataset, key, *args, **kwargs):
        reads.append((dataset.name, key))
        return original(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record)
    channel = read_luxendo_channel(path)
    assert channel.name == "ch0 — 561"
    assert channel.scale == (0.7, 0.3, 0.2)
    assert all(isinstance(array, da.Array) for array in channel.arrays)
    assert [array.shape for array in channel.arrays] == [
        (16, 32, 48),
        (8, 16, 24),
        (4, 8, 12),
        (2, 4, 6),
        (1, 2, 3),
    ]
    image_reads = [(name, key) for name, key in reads if name != "/metadata"]
    assert len(image_reads) == 1
    assert image_reads[0][0] == "/Data_16_16_16"
    assert all(isinstance(part, slice) for part in image_reads[0][1])
    # Remains eligible for the existing parallel HDF5 section backend.
    descriptor = describe_h5_volume(channel.arrays[0])
    assert descriptor[:3] == (str(path), "/Data", (16, 32, 48))
    expected = np.arange(16 * 32 * 48, dtype=np.uint16).reshape(16, 32, 48)
    np.testing.assert_array_equal(
        channel.arrays[0][2:4].compute(), expected[2:4]
    )


def test_folder_channels_recursion_headers_and_rejections(
    channel_file, tmp_path
):
    channel_file()
    channel_file("sub/ch1.h5", channel=1, scale=(1.4, 0.6, 0.4))
    channel_file("main_st-1_bdv.h5", metadata=False)
    channel_file("view_bdv.h5", metadata=False)
    channel_file("missing-metadata.h5", metadata=False)
    channel_file("bad-spacing.h5", scale=(0, 1, 1))
    assert len(scan_luxendo_files(tmp_path, recursive=False)) == 3
    plan = prepare_luxendo_folder(tmp_path)
    assert len(plan.channels) == 2
    assert {channel.scale for channel in plan.channels} == {
        (0.7, 0.3, 0.2),
        (1.4, 0.6, 0.4),
    }
    assert len(plan.skipped) == 2
    assert any("missing embedded" in reason for reason in plan.skipped)
    assert any("positive" in reason for reason in plan.skipped)


def test_empty_folder_and_cancellation(tmp_path):
    with pytest.raises(ValueError, match="No per-channel"):
        prepare_luxendo_folder(tmp_path)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ImportCancelled):
        prepare_luxendo_folder(tmp_path, cancel=cancel)


def test_corrupt_metadata_and_pyramid_are_reported(channel_file, tmp_path):
    path = channel_file()
    with h5py.File(path, "a") as file:
        del file["metadata"]
        file["metadata"] = b"not json"
    with pytest.raises(ValueError, match="ch0.lux.h5"):
        prepare_luxendo_folder(tmp_path)
    path = channel_file()
    with h5py.File(path, "a") as file:
        del file["Data_2_2_2"]
        file["Data_2_2_2"] = np.ones((2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="invalid pyramid"):
        read_luxendo_channel(path)


def test_file_survives_derived_arrays_then_closes(channel_file):
    channel = read_luxendo_channel(channel_file())
    dataset = next(
        value
        for value in channel.arrays[0].dask.values()
        if isinstance(value, h5py.Dataset)
    )
    file_id = dataset.file.id
    derived = channel.arrays[0][1:3] + 1
    del dataset, channel
    gc.collect()
    assert file_id.valid
    assert derived.compute().shape == (2, 32, 48)
    del derived
    gc.collect()
    assert not file_id.valid


def test_import_sets_per_channel_scale_and_deduplicates(
    channel_file, tmp_path
):
    channel_file()
    channel_file("ch1.h5", channel=1, scale=(1.4, 0.6, 0.4), pyramid=False)
    viewer = ViewerModel()
    layers = import_luxendo_folder(viewer, tmp_path)
    assert len(layers) == 2
    assert layers[0].multiscale
    assert not layers[1].multiscale
    for layer, scale in zip(layers, [(0.7, 0.3, 0.2), (1.4, 0.6, 0.4)]):
        np.testing.assert_allclose(layer.scale, scale)
        assert tuple(map(str, layer.units)) == ("micrometer",) * 3
        assert layer.source.path == layer.metadata[LUXENDO_METADATA]["path"]
        assert tuple(layer.contrast_limits) == layer._calc_data_range("slice")
    assert import_luxendo_folder(viewer, tmp_path) == []
    assert len(viewer.layers) == 2


@pytest.mark.parametrize("pyramid", [False, True])
def test_each_luxendo_channel_uses_napari_auto_contrast(channel_file, pyramid):
    channel = read_luxendo_channel(channel_file(pyramid=pyramid))
    channel.contrast_limits = (0, 65535)
    viewer = ViewerModel()

    layer = add_luxendo_channel(viewer, channel)

    assert tuple(layer.contrast_limits) == layer._calc_data_range("slice")
    assert tuple(layer.contrast_limits) != (0, 65535)


def test_saved_keyframes_reopen_luxendo_channels(channel_file, tmp_path):
    channel_file()
    viewer = ViewerModel()
    import_luxendo_folder(viewer, tmp_path)
    animation = Animation(viewer)
    animation.key_frames.append(
        KeyFrame(
            viewer_state=ViewerState.from_viewer(viewer),
            thumbnail=np.zeros((30, 30, 4), dtype=np.uint8),
        )
    )
    path = animation.save_keyframes(tmp_path / "animation.json")
    assert json.loads(path.read_text())["layers"][0]["reader"] == "luxendo"
    restored = Animation(ViewerModel())
    try:
        restored.load_keyframes(path)
        assert len(restored.viewer.layers) == 1
        layer = restored.viewer.layers[0]
        assert layer.name == "ch0 — 561"
        np.testing.assert_allclose(layer.scale, (0.7, 0.3, 0.2))
        assert isinstance(layer.data[0], da.Array)
        np.testing.assert_array_equal(
            layer.data[0][0].compute(), viewer.layers[0].data[0][0].compute()
        )
    finally:
        animation.close()
        restored.close()


def test_gui_import_runs_preparation_off_thread(
    channel_file, tmp_path, qtbot, monkeypatch
):
    import napari_animation.luxendo as backend

    channel_file()
    channel_file("ch1.h5", channel=1, scale=(1.4, 0.6, 0.4))
    viewer = ViewerModel()
    widget = AnimationWidget(viewer)
    qtbot.addWidget(widget)
    importer = widget.luxendoImportWidget
    widget.voxelSizeWidget.unitsComboBox.setCurrentText("nm")
    main_thread = threading.get_ident()
    prepare_threads, install_threads, heartbeats = [], [], []
    prepare, install = (
        backend.prepare_luxendo_folder,
        backend.add_luxendo_channel,
    )

    def slow_prepare(*args, **kwargs):
        prepare_threads.append(threading.get_ident())
        time.sleep(0.15)
        return prepare(*args, **kwargs)

    def record_install(*args, **kwargs):
        install_threads.append(threading.get_ident())
        return install(*args, **kwargs)

    monkeypatch.setattr(backend, "prepare_luxendo_folder", slow_prepare)
    monkeypatch.setattr(backend, "add_luxendo_channel", record_install)
    timer = QTimer(widget)
    timer.timeout.connect(lambda: heartbeats.append(time.perf_counter()))
    timer.start(10)
    importer.pathEdit.setText(str(tmp_path))
    importer.start_import()
    qtbot.waitUntil(lambda: not importer._busy, timeout=10000)
    timer.stop()
    assert len(viewer.layers) == 2
    assert prepare_threads and main_thread not in prepare_threads
    assert install_threads == [main_thread, main_thread]
    assert len(heartbeats) >= 5
    assert widget.voxelSizeWidget.unitsComboBox.currentText() == "um"
    assert widget.voxelSizeWidget.spinBoxes["Z"].value() == 0.7
    assert widget.voxelSizeWidget.spinBoxes["Y"].value() == 0.3
    assert widget.voxelSizeWidget.spinBoxes["X"].value() == 0.2
    assert importer._added == []  # No extra ownership of removed image layers.
    assert "Imported 2" in importer.statusLabel.text()
    importer.start_import()
    qtbot.waitUntil(lambda: not importer._busy, timeout=10000)
    assert len(viewer.layers) == 2
    assert "2 already loaded" in importer.statusLabel.text()
    widget.close()
    # Closing the dock must leave imported channels usable in the viewer.
    assert viewer.layers[0].data[0][0].compute().shape == (32, 48)


@pytest.mark.parametrize("close", [False, True])
def test_cancel_or_close_discards_late_import(
    channel_file, tmp_path, qtbot, monkeypatch, close
):
    import napari_animation.luxendo as backend

    channel_file()
    widget = AnimationWidget(ViewerModel())
    qtbot.addWidget(widget)
    importer = widget.luxendoImportWidget
    ready, release = threading.Event(), threading.Event()
    prepare = backend.prepare_luxendo_folder

    def delayed_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        ready.set()
        release.wait(5)
        return result

    monkeypatch.setattr(backend, "prepare_luxendo_folder", delayed_prepare)
    importer.pathEdit.setText(str(tmp_path))
    importer.start_import()
    qtbot.waitUntil(ready.is_set, timeout=5000)
    future = importer._future
    if close:
        widget.close()
    else:
        importer._cancel_import()
    release.set()
    qtbot.waitUntil(future.done, timeout=5000)
    qtbot.wait(30)
    assert len(widget.viewer.layers) == 0
    assert importer._future is None
    if not close:
        assert not importer._busy
        assert "cancelled" in importer.statusLabel.text()
