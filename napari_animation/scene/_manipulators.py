"""Optional in-canvas manipulation of ortho-slice planes.

`napari-threedee <https://github.com/napari-threedee/napari-threedee>`_ provides
a vispy gizmo for dragging and rotating an image layer's rendering plane. It
owns no state of its own -- it reads and writes ``layer.plane.position`` and
``layer.plane.normal``, which is exactly what :class:`OrthoSlice` treats as the
source of truth. So the two compose with no glue: drag the gizmo, capture a
keyframe, and the new geometry is recorded.

It is deliberately an *optional* dependency (``pip install
napari-animation[threedee]``). Its install pulls in cryo-EM specific packages
and pins ``zarr<3``, which does not build on current Python, so requiring it
would break napari-animation installs. Everything here degrades to ``None``
when it is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import napari

#: Shown in the UI when the optional dependency is missing.
INSTALL_HINT = (
    "Interactive plane manipulation needs napari-threedee: "
    "pip install 'napari-animation[threedee]'"
)


def is_available() -> bool:
    """Whether napari-threedee is installed."""
    try:
        from napari_threedee.manipulators import (  # noqa: F401
            RenderPlaneManipulator,
        )
    except ImportError:
        return False
    return True


def attach_plane_manipulator(
    viewer: "napari.viewer.Viewer", layer
) -> Optional[object]:
    """Attach an in-canvas manipulator to an ortho slice's backing layer.

    Returns the manipulator, or ``None`` if napari-threedee is not installed
    or the layer is not in plane depiction. The caller keeps the reference
    alive for as long as the manipulator should stay active.
    """
    try:
        from napari_threedee.manipulators import RenderPlaneManipulator
    except ImportError:
        return None

    if getattr(layer, "depiction", None) != "plane":
        # the manipulator only makes sense for a rendering plane, and would
        # immediately disable itself anyway
        return None

    try:
        return RenderPlaneManipulator(viewer, layer=layer, enabled=True)
    except Exception:  # noqa: BLE001 - needs a live canvas; never fatal
        return None
