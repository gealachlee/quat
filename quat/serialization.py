"""Serialization and interop: JSON, binary bytes, scipy Rotation, numpy views."""
from __future__ import annotations
import json
import struct
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix, QuatTensor
from quat.utils import to_ndarray, from_ndarray


def to_json(q: Quaternion | QuatVector | QuatMatrix | QuatTensor) -> str:
    """Serialize any quaternion object to a JSON string.

    The JSON object has ``"type"`` (class name) and ``"data"`` (nested list).

    Example:
        >>> from quat import Quaternion, to_json, from_json
        >>> q = Quaternion(1, 2, 3, 4)
        >>> s = to_json(q)
        >>> q2 = from_json(s)
        >>> q == q2
        True
    """
    data = to_ndarray(q)
    result = {"type": type(q).__name__, "data": data.tolist()}
    return json.dumps(result)


def from_json(s: str) -> Quaternion | QuatVector | QuatMatrix | QuatTensor:
    """Deserialize a JSON string back to a quaternion object."""
    d = json.loads(s)
    arr = np.array(d["data"], dtype=float)
    return from_ndarray(arr)


def to_bytes(q: Quaternion | QuatVector | QuatMatrix | QuatTensor) -> bytes:
    """Serialize a quaternion object to a compact binary format.

    Format: ``<type_id:i4><ndim:i4><shape:i4[]><data:f8[]>``

    type_id: 0=Quaternion, 1=QuatVector, 2=QuatMatrix, 3=QuatTensor.

    Example:
        >>> from quat import Quaternion, to_bytes, from_bytes
        >>> q = Quaternion(1.5, -2.3, 3.7, -4.1)
        >>> b = to_bytes(q)
        >>> q == from_bytes(b)
        True
    """
    if isinstance(q, Quaternion):
        type_id = 0
    elif isinstance(q, QuatVector):
        type_id = 1
    elif isinstance(q, QuatMatrix):
        type_id = 2
    elif isinstance(q, QuatTensor):
        type_id = 3
    else:
        raise TypeError(f"Cannot serialize {type(q)} to bytes")
    data = to_ndarray(q)
    shape = np.array(data.shape, dtype=np.int32)
    header = struct.pack('<ii', type_id, len(shape)) + shape.tobytes()
    return header + data.astype(np.float64).tobytes()


def from_bytes(b: bytes) -> Quaternion | QuatVector | QuatMatrix | QuatTensor:
    """Deserialize bytes back to a quaternion object."""
    type_id, ndim = struct.unpack_from('<ii', b, 0)
    offset = 8
    shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
    offset += ndim * 4
    size = int(np.prod(shape))
    data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64)
    data = data.reshape(shape)
    return from_ndarray(data)


def to_scipy_rotation(q: Quaternion | QuatVector) -> "Rotation":  # noqa: F821
    """Convert quaternion(s) to ``scipy.spatial.transform.Rotation``.

    scipy uses ``(x, y, z, w)`` order; our internal order is ``(w, x, y, z)``
    ≡ ``(r, i, j, k)``.  Only meaningful for unit quaternions.

    Example:
        >>> from quat import Quaternion, to_scipy_rotation
        >>> import numpy as np
        >>> q = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)  # 90° about z
        >>> rot = to_scipy_rotation(q)
        >>> rot.as_quat()  # scipy returns (x, y, z, w)
        array([0.        , 0.        , 0.70710678, 0.70710678])
    """
    from scipy.spatial.transform import Rotation
    data = to_ndarray(q)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return Rotation.from_quat(data[:, [1, 2, 3, 0]])


def from_scipy_rotation(rot: "Rotation") -> Quaternion | QuatVector:  # noqa: F821
    """Convert ``scipy.spatial.transform.Rotation`` to quaternion(s).

    Returns Quaternion for a single rotation, QuatVector for multiple.

    Example:
        >>> from quat import Quaternion, to_scipy_rotation, from_scipy_rotation
        >>> import numpy as np
        >>> q = Quaternion.from_axis_angle((1, 0, 0), np.pi / 4)
        >>> rot = to_scipy_rotation(q)
        >>> q2 = from_scipy_rotation(rot)
        >>> q.normalize().isclose(q2.normalize())
        True
    """
    scipy_quat = rot.as_quat()
    if scipy_quat.ndim == 1:
        scipy_quat = scipy_quat.reshape(1, -1)
    data = np.zeros((scipy_quat.shape[0], 4))
    data[:, 0] = scipy_quat[:, 3]
    data[:, 1] = scipy_quat[:, 0]
    data[:, 2] = scipy_quat[:, 1]
    data[:, 3] = scipy_quat[:, 2]
    if data.shape[0] == 1:
        return Quaternion(data[0])
    return QuatVector(data)
