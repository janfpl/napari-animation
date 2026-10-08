"""Reproduce overlapping export/prefetch cache finalization deterministically."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import dask.array as da
import numpy as np
import pytest
from dask.cache import Cache
from dask.callbacks import Callback

from napari_animation._dask_cache import ConcurrentCache, protect_napari_cache


@pytest.mark.parametrize("cache_type", [Cache, ConcurrentCache])
def test_overlapping_finalizers(cache_type):
    reached, release = Event(), Event()

    class PausingCache(cache_type):
        def _posttask(self, key, value, dsk, state, worker_id):
            if str(key).startswith(
                "finalize-"
            ) and current_thread().name.startswith("export"):
                reached.set()
                assert release.wait(5)
            return super()._posttask(key, value, dsk, state, worker_id)

    cache = PausingCache(10**6)
    array = da.from_array(np.arange(64), chunks=8)
    # Explicit callbacks make the overlap deterministic regardless of Dask's
    # global callback suppression for nested/concurrent schedulers.

    def compute():
        return array.compute(
            scheduler="synchronous", callbacks=[cache._callback]
        )

    with ThreadPoolExecutor(1, thread_name_prefix="export") as pool:
        future = pool.submit(compute)
        try:
            assert reached.wait(5)
            np.testing.assert_array_equal(compute(), np.arange(64))
        finally:
            release.set()
        if cache_type is Cache:
            with pytest.raises(KeyError, match="finalize-hlgfinalizecompute"):
                future.result(timeout=5)
        else:
            np.testing.assert_array_equal(
                future.result(timeout=5), np.arange(64)
            )
            assert cache.cache.total_bytes == sum(cache.cache.nbytes.values())


def test_concurrent_cache_reuses_tasks():
    from dask import delayed

    reads = []

    @delayed(pure=True)
    def read():
        reads.append(1)
        return np.arange(10)

    array = da.from_delayed(read(), shape=(10,), dtype=int)
    cache = ConcurrentCache(10**6)
    for _ in range(2):
        np.testing.assert_array_equal(
            array.compute(
                scheduler="synchronous",
                callbacks=[cache._callback],
                optimize_graph=False,
            ),
            np.arange(10),
        )
    assert reads == [1]


def test_contexts_and_unregister_do_not_remove_another_user():
    cache = ConcurrentCache(10**6)
    cache.register()
    with cache:
        with cache:
            cache.unregister()
            assert cache._callback in Callback.active
        assert cache._callback in Callback.active
    assert cache._callback not in Callback.active
    cache.unregister()  # Safe even when Dask has temporarily removed callbacks.


def test_existing_napari_layer_cache_references_are_upgraded(monkeypatch):
    from napari.utils import _dask_utils

    old = Cache(1024)
    monkeypatch.setattr(_dask_utils, "_DASK_CACHE", old)
    context = _dask_utils.configure_dask(da.ones((2, 3), chunks=1))
    old.cache.put("existing", 42, cost=1, nbytes=100)
    cache = protect_napari_cache()
    assert cache is old
    assert isinstance(cache, ConcurrentCache)
    assert cache.cache.available_bytes == 1024
    assert cache.cache.data["existing"] == 42
    with context() as (_, captured):
        assert captured is cache
    assert protect_napari_cache() is cache


def test_unregister_during_compute_does_not_resurrect_cache():
    from dask.callbacks import local_callbacks

    cache = ConcurrentCache(10**6)
    cache.register()
    with local_callbacks():
        cache.unregister()
    assert cache._callback not in Callback.active


def test_failed_computation_does_not_poison_the_cache():
    from dask import delayed

    @delayed
    def fail():
        raise ValueError("source read failed")

    cache = ConcurrentCache(10**6)
    with pytest.raises(ValueError, match="source read failed"):
        fail().compute(callbacks=[cache._callback])
    array = da.from_array(np.arange(16), chunks=4)
    np.testing.assert_array_equal(
        array.compute(callbacks=[cache._callback]), np.arange(16)
    )


def test_real_parallel_arrays_and_eviction():
    cache = ConcurrentCache(4096)
    data = np.arange(32 * 16).reshape(32, 16)
    array = da.from_array(data, chunks=(2, 8))

    def compute(index):
        return (array[index % 32] * 3).compute(callbacks=[cache._callback])

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(compute, range(100)))
    for index, result in enumerate(results):
        np.testing.assert_array_equal(result, data[index % 32] * 3)
    assert cache.cache.total_bytes == sum(cache.cache.nbytes.values())
    assert cache.cache.total_bytes <= cache.cache.available_bytes


def test_gui_hdf5_export_finishes_with_prefetch(
    make_napari_viewer, qtbot, tmp_path
):
    import h5py
    import imageio.v2 as imageio

    from napari_animation import Animation
    from napari_animation._qt.export_controller import QtExportController

    viewer = make_napari_viewer(ndisplay=2, show=True)
    animation = Animation(viewer)
    path = tmp_path / "source.h5"
    z, y, x = np.indices((24, 32, 32))
    data = (z * 20 + x + y).astype(np.uint16)
    results = []
    try:
        with h5py.File(path, "w") as file:
            dataset = file.create_dataset(
                "Data", data=data, chunks=(2, 16, 16), compression="gzip"
            )
            array = da.from_array(dataset, chunks=dataset.chunks, lock=True)
            viewer.add_image(array, contrast_limits=(0, 600))
            viewer.reset_view()
            qtbot.wait(100)
            animation.add_scroll_through(start=0, stop=23, steps=8)
            output = tmp_path / "out.mp4"
            controller = QtExportController(
                animation,
                {
                    "path": str(output),
                    "fps": 10,
                    "prefetch": 4,
                    "prefetch_workers": 4,
                    "dask_cache": 1024**2,
                    "diagnose": False,
                },
            )
            controller.finished.connect(
                lambda path, error: results.append(error)
            )
            controller.start()
            qtbot.waitUntil(lambda: bool(results), timeout=30000)
            assert results == [None]
            with imageio.get_reader(output) as reader:
                assert reader.count_frames() == len(animation._frames)
                assert not np.array_equal(
                    reader.get_data(0),
                    reader.get_data(len(animation._frames) - 1),
                )
    finally:
        animation.close()
