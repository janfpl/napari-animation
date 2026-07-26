import numbers
from enum import Enum
from functools import partial
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R


def default(a, b, fraction):
    """Default interpolation for the corresponding type;
    linear interpolation for numeric, instantaneous transition otherwise.

    Parameters
    ----------
    a :
        initial value
    b :
        final value
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
        Interpolated value between a and b at fraction.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        # checking this first because booleans are numbers
        return interpolate_bool(a, b, fraction)

    elif isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        return interpolate_num(a, b, fraction)

    elif isinstance(a, dict) and isinstance(b, dict):
        return interpolate_dict(a, b, fraction)

    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return interpolate_seq(a, b, fraction)

    else:
        # strings, etc.
        return interpolate_bool(a, b, fraction)


def interpolate_seq(a, b, fraction):
    """Interpolation of list or tuple.

    Sequences of unequal length interpolate over their common prefix; any
    trailing elements of the longer sequence are held unchanged rather than
    dropped. Without this a clipping plane present in only one of two
    keyframes would silently vanish partway through the transition, because
    ``zip`` truncates to the shorter sequence.

    Parameters
    ----------
    a : list or tuple
        initial sequence
    b : list or tuple
        final sequence
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
        : sequence of type a
    Interpolated sequence between a and b at fraction.
    """
    shared = min(len(a), len(b))
    values = [
        default(v0, v1, fraction) for v0, v1 in zip(a[:shared], b[:shared])
    ]
    longer = a if len(a) > len(b) else b
    values.extend(longer[shared:])
    return type(a)(values)


def interpolate_dict(a, b, fraction):
    """Key-wise interpolation of two dictionaries.

    Keys present in only one of the two dictionaries are carried through
    unchanged. This is what lets structured values -- a napari ``plane``
    (``position`` / ``normal`` / ``thickness``) or a clipping-plane
    definition -- interpolate field by field instead of snapping wholesale to
    the target value.

    Parameters
    ----------
    a : dict
        initial mapping
    b : dict
        final mapping
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
        : dict
    Interpolated mapping between a and b at fraction.
    """
    interpolated = {}
    for key, value in a.items():
        if key in b:
            interpolated[key] = default(value, b[key], fraction)
        else:
            interpolated[key] = value
    for key, value in b.items():
        if key not in a:
            interpolated[key] = value
    return interpolated


def interpolate_num(a, b, fraction):
    """Linear interpolation for numeric types.

    Parameters
    ----------
    a : numeric type
        initial value
    b : numeric type
        final value
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
        : numeric type
    Interpolated value between a and b at fraction.
    """
    return type(a)(a + (b - a) * fraction)


def interpolate_bool(a, b, fraction):
    """Instantaneous transition from a to b.

    Parameters
    ----------
    a :
        initial value
    b :
        final value
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
    a or b :
        b if any step was taken into its direction, otherwise a.
    """
    if fraction > 0.0:
        return b
    else:
        return a


def step_start(a, b, fraction):
    """Discrete transition that switches at the *start* of the interval.

    Parameters
    ----------
    a :
        initial value
    b :
        final value
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
    a or b :
        b for any fraction above zero, otherwise a.
    """
    return b if fraction > 0.0 else a


def step_end(a, b, fraction):
    """Discrete transition that switches at the *end* of the interval.

    The value is held for the whole transition and only changes on arrival at
    the destination keyframe. This is the right behaviour for state that
    cannot be meaningfully blended -- ``dims.ndisplay``, a colormap, a
    rendering mode -- where switching early means most of the transition is
    rendered in the wrong mode.

    Parameters
    ----------
    a :
        initial value
    b :
        final value
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
    a or b :
        a until the interval completes, then b.
    """
    return b if fraction >= 1.0 else a


def slerp_vector(a, b, fraction):
    """Spherical interpolation between two direction vectors.

    Rotates the direction along the shortest arc while interpolating the
    magnitude linearly, so a plane normal sweeps smoothly through
    intermediate orientations. Component-wise linear interpolation would
    instead shrink the vector toward the origin as it passes through the
    halfway point of a large rotation.

    Parameters
    ----------
    a : sequence of float
        initial vector
    b : sequence of float
        final vector
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
        : sequence of type a
    Interpolated vector between a and b at fraction.
    """
    v0 = np.asarray(a, dtype=float)
    v1 = np.asarray(b, dtype=float)
    norm0 = np.linalg.norm(v0)
    norm1 = np.linalg.norm(v1)

    # a zero-length vector has no direction to rotate; fall back to linear
    if norm0 == 0 or norm1 == 0:
        return interpolate_seq(a, b, fraction)

    unit0, unit1 = v0 / norm0, v1 / norm1
    magnitude = norm0 + (norm1 - norm0) * fraction
    dot = float(np.clip(np.dot(unit0, unit1), -1.0, 1.0))

    if dot > 1.0 - 1e-9:
        # (anti)parallel enough that the arc is degenerate
        direction = unit0
    elif dot < -1.0 + 1e-9:
        # opposed: the shortest arc is ambiguous, so rotate about an
        # arbitrary perpendicular axis to keep the motion continuous.
        axis = np.cross(unit0, (1.0, 0.0, 0.0))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(unit0, (0.0, 1.0, 0.0))
        axis /= np.linalg.norm(axis)
        angle = np.pi * fraction
        direction = unit0 * np.cos(angle) + np.cross(axis, unit0) * np.sin(
            angle
        )
    else:
        omega = np.arccos(dot)
        sin_omega = np.sin(omega)
        direction = (np.sin((1.0 - fraction) * omega) / sin_omega) * unit0 + (
            np.sin(fraction * omega) / sin_omega
        ) * unit1

    result = direction * magnitude
    if isinstance(a, (list, tuple)):
        return type(a)(result.tolist())
    return result


def interpolate_log(a, b, fraction):
    """Log interpolation, for camera zoom mostly.

    Parameters
    ----------
    a : float
        initial value
    b : float
        final value
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
        : float
    Log interpolated value between a and b at fraction.
    """
    c = interpolate_num(np.log10(a), np.log10(b), fraction)
    return np.power(10, c)


def slerp(a, b, fraction):
    """Compute Spherical linear interpolation from Euler angles,
    compatible with the napari view.

    Parameters
    ----------
    a : tuple
        initial tuple of Euler angles in degrees.
    b : tuple
        final tuple of Euler angles in degrees.
    fraction : float
        fraction to interpolate to between a and b.

    Returns
    ----------
        : tuple
    Interpolated Euler angles between a and b at fraction.
    """
    initial_rotation, final_rotation = R.from_euler(
        "ZYX", [a, b], degrees=True
    )
    rotation_vector = (initial_rotation.inv() * final_rotation).as_rotvec()
    rotation_vector *= fraction
    c_rotation = initial_rotation * R.from_rotvec(rotation_vector)
    return c_rotation.as_euler("ZYX", degrees=True)


class Interpolation(Enum):
    """Interpolation: interpolation function to use for a transition.

    Selects a preset interpolation function
        * DEFAULT: linear interpolation between start and endpoint.
        * SLERP: spherical linear interpolation on Euler angles.
        * SLERP_VECTOR: spherical interpolation of a direction vector, for
          plane normals.
        * LOG: log interpolation between start and endpoint.
        * STEP_START: discrete switch at the start of the transition.
        * STEP_END: discrete switch on arrival at the destination keyframe.

    """

    DEFAULT = partial(default)
    LOG = partial(interpolate_log)
    SLERP = partial(slerp)
    SLERP_VECTOR = partial(slerp_vector)
    STEP_START = partial(step_start)
    STEP_END = partial(step_end)

    def __call__(self, *args):
        return self.value(*args)


InterpolationMap = Dict[str, Interpolation]
