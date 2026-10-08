"""Small import surface for Windows-spawned HDF5 reader processes."""

from __future__ import annotations

from ._projection_core import project_slab_core

_WORKER_FILE = None
_WORKER_DATASET = None


def open_worker(path, dataset_name):
    global _WORKER_FILE, _WORKER_DATASET
    import h5py

    _WORKER_FILE = h5py.File(path, "r")
    _WORKER_DATASET = _WORKER_FILE[dataset_name]


class StripeVolume:
    """Expose an output-axis stripe with local indices to the core reducer."""

    def __init__(self, dataset, axis, first, last):
        self.dataset = dataset
        self.output_axis = next(index for index in range(3) if index != axis)
        self.first = first
        self.shape = list(dataset.shape)
        self.shape[self.output_axis] = last - first
        self.shape = tuple(self.shape)
        self.dtype = dataset.dtype
        self.chunks = dataset.chunks

    def __getitem__(self, key):
        selection = list(key)
        local = selection[self.output_axis]
        selection[self.output_axis] = slice(
            self.first + local.start, self.first + local.stop, local.step
        )
        return self.dataset[tuple(selection)]


def project_stripe(task):
    axis, start, stop, projection, first, last = task
    volume = StripeVolume(_WORKER_DATASET, axis, first, last)
    return project_slab_core(volume, axis, start, stop, projection)
