from types import SimpleNamespace

import numpy as np
import pytest

from napari_animation.layer_registry import SOURCE_ID_METADATA
from napari_animation.prefetch import (
    SlicePrefetcher,
    dask_cache_context,
    read_keys,
    resolve_cache_bytes,
)


class CountingArray:
    """Array-like that counts how many times it is sliced (a 'read')."""

    def __init__(self, arr):
        self._arr = arr
        self.reads = 0

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def dtype(self):
        return self._arr.dtype

    def __getitem__(self, idx):
        self.reads += 1
        return self._arr[idx]


def _dims(point, ranges, ndisplay=2, ndim=3):
    return {
        "ndim": ndim,
        "ndisplay": ndisplay,
        "order": list(range(ndim)),
        "point": point,
        "range": ranges,
    }


def test_read_keys_slices_displayed_axes_indexes_rest():
    arr = CountingArray(np.zeros((10, 20, 20)))
    layer = SimpleNamespace(data=arr, ndim=3, multiscale=False, data_level=0)
    viewer = SimpleNamespace(layers=[layer])
    dims = _dims([5, 10, 10], [(0, 9, 1), (0, 19, 1), (0, 19, 1)])

    keys = read_keys(viewer, dims)
    assert len(keys) == 1
    array, index = keys[0]
    assert array is arr
    # axis 0 (Z, not displayed) -> integer; axes 1,2 (displayed) -> full slice
    assert index[0] == 5
    assert index[1] == slice(None) and index[2] == slice(None)


def test_read_keys_skips_resident_full_volume_but_keeps_unloaded_volume():
    import dask.array as da

    source = np.arange(8 * 8 * 8).reshape(8, 8, 8)
    array = da.from_array(source, chunks=(4, 8, 8))
    layer = SimpleNamespace(
        data=array,
        ndim=3,
        multiscale=False,
        data_level=0,
        loaded=True,
        _slice=SimpleNamespace(image=SimpleNamespace(view=source)),
    )
    viewer = SimpleNamespace(layers=[layer])
    dims = _dims([0, 0, 0], [(0, 7, 1)] * 3, ndisplay=3)

    assert read_keys(viewer, dims) == []
    layer.loaded = False
    assert len(read_keys(viewer, dims)) == 1
    layer.loaded = True
    assert len(read_keys(viewer, {**dims, "ndisplay": 2})) == 1


def test_prefetcher_disables_only_an_entirely_resident_3d_sequence():
    source = np.zeros((8, 8, 8), dtype=np.uint8)
    layer = SimpleNamespace(
        data=source,
        ndim=3,
        multiscale=False,
        loaded=True,
        metadata={SOURCE_ID_METADATA: "source"},
        _slice=SimpleNamespace(image=SimpleNamespace(view=source)),
    )
    viewer = SimpleNamespace(layers=[layer])
    dims = _dims([0, 0, 0], [(0, 7, 1)] * 3, ndisplay=3)
    frames = [
        SimpleNamespace(dims=dims, layers={"source": {}}) for _ in range(3)
    ]

    prefetcher = SlicePrefetcher(viewer, frames, depth=4)
    assert not prefetcher.enabled
    assert prefetcher._pool is None
    prefetcher.advance(0)
    assert not prefetcher._submitted

    frames[-1] = SimpleNamespace(
        dims={**dims, "ndisplay": 2}, layers={"source": {}}
    )
    prefetcher = SlicePrefetcher(viewer, frames, depth=4)
    try:
        assert prefetcher.enabled
    finally:
        prefetcher.shutdown()


def test_resident_prefetch_check_does_not_materialize_interpolated_states():
    from napari_animation.frame_sequence import FrameSequence
    from napari_animation.key_frame import KeyFrame, KeyFrameList
    from napari_animation.viewer_state import ViewerState

    source = np.zeros((8, 8, 8), dtype=np.uint8)
    layer = SimpleNamespace(
        data=source,
        ndim=3,
        multiscale=False,
        loaded=True,
        metadata={SOURCE_ID_METADATA: "source"},
        _slice=SimpleNamespace(image=SimpleNamespace(view=source)),
    )
    dims = _dims([0, 0, 0], [(0, 7, 1)] * 3, ndisplay=3)
    other = SimpleNamespace(
        data=source,
        ndim=3,
        multiscale=False,
        loaded=True,
        metadata={SOURCE_ID_METADATA: "other"},
        _slice=SimpleNamespace(image=SimpleNamespace(view=source)),
    )
    state = ViewerState(
        camera={}, dims=dims, layers={"source": {}, "other": {}}
    )
    keyframes = KeyFrameList()
    for _ in range(2):
        keyframes.append(KeyFrame(state, np.zeros((1, 1, 4)), steps=10))
    frames = FrameSequence(keyframes)

    prefetcher = SlicePrefetcher(
        SimpleNamespace(layers=[layer, other]), frames, depth=4
    )

    assert len(frames) == 11
    assert not prefetcher.enabled
    assert not frames._cache


def test_read_keys_multiscale_uses_current_level():
    a0 = CountingArray(np.zeros((10, 40, 40)))
    a1 = CountingArray(np.zeros((10, 20, 20)))
    layer = SimpleNamespace(
        data=[a0, a1], ndim=3, multiscale=True, data_level=1
    )
    viewer = SimpleNamespace(layers=[layer])
    dims = _dims([5, 10, 10], [(0, 9, 1), (0, 39, 1), (0, 39, 1)])

    array, _ = read_keys(viewer, dims)[0]
    assert array is a1


@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize(
    "path", ["dims.ndisplay", "layers.source.locked_data_level"]
)
def test_custom_residency_rules_keep_frames_lazy(loaded, path):
    from napari_animation.frame_sequence import FrameSequence
    from napari_animation.interpolation import Interpolation
    from napari_animation.key_frame import KeyFrame, KeyFrameList
    from napari_animation.viewer_state import ViewerState

    source = np.zeros((8, 8, 8), dtype=np.uint8)
    layer = SimpleNamespace(
        data=source,
        ndim=3,
        loaded=loaded,
        metadata={SOURCE_ID_METADATA: "source"},
        _slice=SimpleNamespace(image=SimpleNamespace(view=source)),
    )
    state = ViewerState(camera={}, dims={"ndisplay": 3}, layers={"source": {}})
    keys = KeyFrameList()
    keys.extend(
        [
            KeyFrame(state, np.zeros((1, 1, 3)), steps=1),
            KeyFrame(state, np.zeros((1, 1, 3)), steps=1000),
        ]
    )
    frames = FrameSequence(keys)
    frames.state_interpolation_map[path] = Interpolation.DEFAULT
    prefetcher = SlicePrefetcher(
        SimpleNamespace(layers=[layer]), frames, depth=4
    )
    try:
        assert prefetcher.enabled
        assert not frames._cache
    finally:
        prefetcher.shutdown()


def test_residency_check_does_not_iterate_unknown_lazy_sequence():
    from collections.abc import Sequence

    class LazyFrames(Sequence):
        def __len__(self):
            return 1000

        def __getitem__(self, index):
            raise AssertionError(
                "Residency check must not evaluate lazy frames"
            )

    source = np.zeros((8, 8, 8), dtype=np.uint8)
    layer = SimpleNamespace(
        data=source,
        ndim=3,
        loaded=True,
        metadata={SOURCE_ID_METADATA: "source"},
        _slice=SimpleNamespace(image=SimpleNamespace(view=source)),
    )
    from napari_animation.prefetch import _all_sources_resident

    assert not _all_sources_resident(
        SimpleNamespace(layers=[layer]), LazyFrames()
    )


def test_read_keys_applies_layer_transform_and_pyramid_factor():
    a0 = CountingArray(np.zeros((12, 16, 16)))
    a1 = CountingArray(np.zeros((6, 8, 8)))
    layer = SimpleNamespace(
        data=[a0, a1],
        ndim=3,
        multiscale=True,
        data_level=1,
        downsample_factors=[(1, 1, 1), (2, 2, 2)],
        world_to_data=lambda point: np.asarray(point) / (2, 3, 4),
    )
    viewer = SimpleNamespace(layers=[layer])
    dims = _dims([8, 0, 0], [(0, 22, 2), (0, 45, 3), (0, 60, 4)])
    array, index = read_keys(viewer, dims)[0]
    assert array is a1
    assert index[0] == 2  # world 8 / source scale 2 / pyramid factor 2


def test_read_keys_accepts_napari_multiscale_container():
    """Napari wraps lists in MultiScaleData, which is a Sequence."""
    import dask.array as da
    from napari.components import ViewerModel

    viewer = ViewerModel(ndisplay=2)
    data = da.from_array(np.ones((8, 16, 16)), chunks=(2, 8, 8))
    layer = viewer.add_image([data, data[::2, ::2, ::2]], multiscale=True)
    array, index = read_keys(viewer, viewer.dims.model_dump())[0]
    assert array is layer.data[layer.data_level]
    assert np.asarray(array[index]).ndim == 2


def test_warm_materializes_slice():
    arr = CountingArray(np.zeros((10, 20, 20)))
    layer = SimpleNamespace(data=arr, ndim=3, multiscale=False, data_level=0)
    viewer = SimpleNamespace(layers=[layer])
    pf = SlicePrefetcher(viewer, frames=[], depth=2)
    pf._warm(_dims([5, 10, 10], [(0, 9, 1), (0, 19, 1), (0, 19, 1)]))
    assert arr.reads == 1


def test_advance_submits_future_window_only():
    frames = [
        SimpleNamespace(
            dims=_dims([0, 0, 0], [(0, 1, 1), (0, 1, 1), (0, 1, 1)])
        )
        for _ in range(5)
    ]
    pf = SlicePrefetcher(
        SimpleNamespace(layers=[]), frames, depth=2, workers=1
    )
    pf.advance(0)
    # frame 0 (current) is read by the main thread, so only 1 and 2 prefetch
    assert pf._submitted == {1, 2}
    pf.advance(3)  # 4 is valid, 5 is out of range
    # Old lookahead work is retired rather than accumulating across a movie.
    assert pf._submitted == {4}
    pf.shutdown()


def test_scroll_frames_sharing_dask_chunk_get_one_prefetch_read():
    import dask.array as da

    source = CountingArray(np.arange(8 * 8 * 8).reshape(8, 8, 8))
    array = da.from_array(source, chunks=(4, 8, 8), asarray=False, name=False)
    source.reads = 0  # ignore Dask's shape/dtype probe
    layer = SimpleNamespace(data=array, ndim=3, multiscale=False)
    viewer = SimpleNamespace(layers=[layer])
    frames = [
        SimpleNamespace(
            dims=_dims([z, 0, 0], [(0, 7, 1), (0, 7, 1), (0, 7, 1)])
        )
        for z in range(4)
    ]
    pf = SlicePrefetcher(viewer, frames, depth=3, workers=3)
    try:
        with dask_cache_context(2**20):
            pf.advance(0)
            assert pf._submitted == {1, 2, 3}
            assert len(pf._futures) == 1
            for future, _ in pf._futures.values():
                future.result(timeout=5)
            assert source.reads == 1
            pf.advance(1)
            assert len(pf._futures) == 1
            assert source.reads == 1
    finally:
        pf.shutdown()


def test_disabled_when_depth_zero():
    pf = SlicePrefetcher(SimpleNamespace(layers=[]), [], depth=0)
    assert not pf.enabled
    assert pf._pool is None
    pf.advance(0)  # no-op
    assert pf._submitted == set()


def test_resolve_cache_bytes():
    assert resolve_cache_bytes(None) == 0
    assert resolve_cache_bytes(0) == 0
    assert resolve_cache_bytes(False) == 0
    assert resolve_cache_bytes(123456) == 123456
    auto = resolve_cache_bytes("auto")
    assert 0 < auto <= 2 * 1024**3


def test_dask_cache_context_disabled_yields_zero():
    with dask_cache_context(None) as nbytes:
        assert nbytes == 0


def test_dask_cache_context_enabled_caches_results(monkeypatch):
    import dask.array as da

    import napari_animation.prefetch as prefetch

    monkeypatch.setattr(prefetch, "_INTERACTIVE_CACHE", None)
    arr = da.from_array(np.arange(100).reshape(10, 10), chunks=(1, 10))
    with dask_cache_context(10**7) as nbytes:
        assert nbytes == 10**7
        # cache is active and dask still computes correctly
        np.testing.assert_array_equal(np.asarray(arr[3]), np.arange(30, 40))


def test_dask_cache_context_reports_shared_budget(monkeypatch):
    import napari_animation.prefetch as prefetch

    fake = SimpleNamespace(
        cache=SimpleNamespace(
            available_bytes=1024,
            total_bytes=42,
        )
    )
    monkeypatch.setattr(prefetch, "_INTERACTIVE_CACHE", fake)
    with dask_cache_context(10**7) as nbytes:
        assert nbytes == 1024


def test_direct_animation_releases_shared_cache_on_viewer_close(
    make_napari_viewer, monkeypatch
):
    from qtpy.QtWidgets import QApplication

    import napari_animation.prefetch as prefetch
    from napari_animation import Animation

    monkeypatch.setenv("NAPARI_ANIMATION_INTERACTIVE_CACHE_MB", "1")
    before = prefetch._INTERACTIVE_CACHE_USERS
    # A bare napari.Viewer() re-registers every plugin's commands, which
    # fails once the fixture-managed test viewers have registered them.
    viewer = make_napari_viewer()
    animation = Animation(viewer)
    assert prefetch._INTERACTIVE_CACHE_USERS == before + 1
    viewer.close()
    QApplication.processEvents()
    assert animation._closed
    assert prefetch._INTERACTIVE_CACHE_USERS == before
    animation.close()  # cleanup is idempotent when a dock also closes
