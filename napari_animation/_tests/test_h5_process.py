"""Pure planning and exact-pixel checks for parallel HDF5 sections."""

import threading
from uuid import uuid4

import dask.array as da
import h5py
import numpy as np
import pytest

from napari_animation.scene.h5_process import (
    _StripeVolume,
    _tasks,
    describe_h5_volume,
    process_lanes,
)
from napari_animation.scene.processing import project_slab


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("projection", ("max", "min", "mean"))
def test_h5_stripes_reconstruct_exact_projection(axis, projection, tmp_path):
    parent = tmp_path
    path = parent / f"{uuid4().hex}.ims"
    data = np.arange(8 * 10 * 12, dtype=np.uint16).reshape(8, 10, 12)
    try:
        with h5py.File(path, "w") as file:
            dataset = file.create_dataset(
                "Data", data=data, chunks=(4, 5, 6), compression="gzip"
            )
            array = da.from_array(dataset, chunks=dataset.chunks)
            descriptor = describe_h5_volume(array)
            assert descriptor[:2] == (str(path), "/Data")
            assert descriptor[5] == "gzip"
            assert describe_h5_volume(array + 1) is None

            output_axis = next(index for index in range(3) if index != axis)
            tasks = list(_tasks(descriptor, axis, 1, 4, projection, 3))
            assert tasks[0][-2] == 0
            assert tasks[-1][-1] == data.shape[output_axis]
            assert all(a[-1] == b[-2] for a, b in zip(tasks, tasks[1:]))
            parts = [
                project_slab(
                    _StripeVolume(dataset, axis, first, last),
                    axis,
                    1,
                    4,
                    projection,
                )
                for *_prefix, first, last in tasks
            ]
            actual = np.concatenate(parts, axis=0)
            selection = [slice(None)] * 3
            selection[axis] = slice(1, 4)
            expected = getattr(np, projection)(
                data[tuple(selection)], axis=axis
            )
            np.testing.assert_array_equal(actual, expected)
    finally:
        path.unlink(missing_ok=True)


def test_h5_process_lane_policy(monkeypatch):
    from types import SimpleNamespace

    import psutil

    import napari_animation.scene.h5_process as backend

    descriptor = (
        "volume.ims",
        "/Data",
        (1024, 1024, 1024),
        "<u2",
        (128, 64, 64),
        "gzip",
    )
    monkeypatch.setattr(backend, "_MIN_PHYSICAL_BYTES", 1)
    monkeypatch.setenv("NAPARI_ANIMATION_H5_PROCESS_LANES", "auto")
    expected = min(4, backend.os.cpu_count() or 1)
    assert process_lanes(descriptor, 0, 120, 124) == (
        expected if expected >= 2 else 0
    )
    assert process_lanes(descriptor[:-1] + (None,), 0, 120, 124) == 0
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=3 * 1024**3),
    )
    assert process_lanes(descriptor, 0, 120, 124) == 2
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1 * 1024**3),
    )
    assert process_lanes(descriptor, 0, 120, 124) == 0
    monkeypatch.setenv("NAPARI_ANIMATION_H5_PROCESS_LANES", "0")
    assert process_lanes(descriptor, 0, 120, 124) == 0
    monkeypatch.setenv("NAPARI_ANIMATION_H5_PROCESS_LANES", "2")
    assert process_lanes(descriptor[:-1] + (None,), 0, 120, 124) == 2


def test_external_link_descriptor_opens_the_data_file(tmp_path):
    parent = tmp_path
    stem = uuid4().hex
    data_path = parent / f"{stem}.lux.h5"
    header_path = parent / f"{stem}.ims"
    try:
        with h5py.File(data_path, "w") as file:
            file.create_dataset("Data", data=np.zeros((4, 4, 4)))
        with h5py.File(header_path, "w") as file:
            file["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"] = (
                h5py.ExternalLink(data_path.name, "/Data")
            )
        with h5py.File(header_path, "r") as file:
            dataset = file[
                "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"
            ]
            descriptor = describe_h5_volume(dataset)
            assert descriptor[0] == str(data_path)
            assert descriptor[1] == "/Data"
            dataset.id.close()
    finally:
        header_path.unlink(missing_ok=True)
        data_path.unlink(missing_ok=True)


def test_singleton_imaris_loader_graph_maps_to_exact_h5_dataset(tmp_path):
    parent = tmp_path
    path = parent / f"{uuid4().hex}.ims"

    class ims_reader:
        shape = (1, 1, 8, 10, 12)
        chunks = (1, 1, 4, 5, 6)
        ndim = 5
        dtype = np.dtype("uint16")
        ResolutionLevelLock = 0
        squeeze_output = False
        write = False

        def __init__(self, file_path):
            self.filePathComplete = str(file_path)

        def __getitem__(self, key):
            return np.zeros((1, 1, 4, 5, 6), dtype=self.dtype)

    ims_reader.__module__ = "imaris_ims_file_reader.ims"

    class CachingReader:
        def __init__(self, reader):
            self._reader = reader

        def __getattr__(self, name):
            return getattr(self._reader, name)

        def __getitem__(self, key):
            return self._reader[key]

    CachingReader.__module__ = "napari_imaris_loader._cache"

    try:
        with h5py.File(path, "w") as file:
            file.create_dataset(
                "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
                shape=(8, 10, 12),
                dtype="u2",
                chunks=(4, 5, 6),
                compression="gzip",
            )
            file.create_dataset(
                "DataSet/ResolutionLevel 1/TimePoint 0/Channel 0/Data",
                shape=(4, 5, 6),
                dtype="u2",
                chunks=(2, 5, 6),
                compression="gzip",
            )
        reader = CachingReader(ims_reader(path))
        volume = da.from_array(reader, chunks=reader.chunks, fancy=False)[0, 0]
        descriptor = describe_h5_volume(volume)
        assert descriptor == (
            str(path),
            "/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
            (8, 10, 12),
            "<u2",
            (4, 5, 6),
            "gzip",
        )
        assert describe_h5_volume(volume[::-1]) is None
        assert describe_h5_volume(volume + 1) is None
        reader._reader.squeeze_output = True
        assert describe_h5_volume(volume) is None
    finally:
        path.unlink(missing_ok=True)


def test_uncompressed_native_chunk_graph_uses_exact_direct_read(
    monkeypatch, tmp_path
):
    import napari_animation.scene.processing as processing

    parent = tmp_path
    path = parent / f"{uuid4().hex}.lux.h5"
    voxels = np.arange(8 * 40 * 40, dtype=np.uint16).reshape(8, 40, 40)
    try:
        with h5py.File(path, "w") as file:
            dataset = file.create_dataset(
                "Data", data=voxels, chunks=(4, 5, 5)
            )
            volume = da.from_array(dataset, chunks=dataset.chunks)
            seen = []
            original = processing.project_slab

            def track(source, *args):
                seen.append(isinstance(source, h5py.Dataset))
                return original(source, *args)

            monkeypatch.setattr(processing, "project_slab", track)
            service = processing.ProjectionService(asynchronous=False)
            request = [
                ("source", 0, id(volume)),
                "section",
                1,
                threading.Event(),
                volume,
                0,
                1,
                4,
                "max",
                None,
            ]
            actual = service._project(request)
            np.testing.assert_array_equal(actual, voxels[1:4].max(axis=0))
            assert seen == [True]
            service.shutdown()
    finally:
        path.unlink(missing_ok=True)


def test_custom_h5_getter_preserves_calibrated_voxels(tmp_path):
    from napari_animation.scene.processing import ProjectionService

    parent = tmp_path
    path = parent / f"{uuid4().hex}.lux.h5"
    voxels = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16)
    try:
        with h5py.File(path, "w") as file:
            dataset = file.create_dataset(
                "Data", data=voxels, chunks=(1, 2, 2)
            )
            volume = da.from_array(
                dataset,
                chunks=dataset.chunks,
                asarray=False,
                lock=True,
                getitem=lambda array, key: array[key] * 2,
            )
            assert describe_h5_volume(volume) is None
            service = ProjectionService()
            request = [
                ("source", 0, id(volume)),
                "section",
                1,
                threading.Event(),
                volume,
                0,
                0,
                2,
                "max",
                None,
            ]
            try:
                actual = service._project(request)
                np.testing.assert_array_equal(actual, (voxels * 2).max(axis=0))
            finally:
                service.shutdown()
    finally:
        path.unlink(missing_ok=True)
