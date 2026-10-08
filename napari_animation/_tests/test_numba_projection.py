"""Numerical checks for the optional multi-core projection backend."""

import numpy as np
import pytest

numba = pytest.importorskip("numba")
from napari_animation.numba_projection import project


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("projection", ["max", "min", "mean"])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_projection_matches_numpy(axis, projection, dtype):
    rng = np.random.default_rng(42)
    slab = (rng.random((9, 17, 18)) * 100).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        slab[3, 4, 5] = np.nan
    expected = getattr(np, projection)(slab, axis=axis)
    actual = project(slab, axis, projection)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)


def test_projection_rejects_empty_axis():
    with pytest.raises(ValueError, match="empty"):
        project(np.empty((0, 2, 3), dtype=np.uint16), 0, "max")
