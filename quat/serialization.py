# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Serialization and interop — thin wrappers over quaternion object methods.

The primary API is via instance and class methods on the quaternion types:

    q = Quaternion(1,2,3,4)
    s = q.to_json()            # instance → JSON string
    q = Quaternion.from_json(s)  # class method ← JSON string
    b = q.to_bytes()           # instance → bytes
    q = Quaternion.from_bytes(b) # class method ← bytes

Standalone functions below delegate to these methods and accept
Quaternion, QuatVector, QuatMatrix, or QuatTensor.
"""
from __future__ import annotations
import json as _json
import struct as _struct
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix, QuatTensor


def to_json(q: Quaternion | QuatVector | QuatMatrix | QuatTensor) -> str:
    return q.to_json()


def from_json(s: str) -> Quaternion | QuatVector | QuatMatrix | QuatTensor:
    d = _json.loads(s)
    cls_map = {
        "Quaternion": Quaternion,
        "QuatVector": QuatVector,
        "QuatMatrix": QuatMatrix,
        "QuatTensor": QuatTensor,
    }
    return cls_map[d["type"]](np.array(d["data"], dtype=float))


def to_bytes(q: Quaternion | QuatVector | QuatMatrix | QuatTensor) -> bytes:
    return q.to_bytes()


def from_bytes(b: bytes) -> Quaternion | QuatVector | QuatMatrix | QuatTensor:
    type_id = _struct.unpack_from('<i', b, 0)[0]
    cls_map = {
        0: Quaternion,
        1: QuatVector,
        2: QuatMatrix,
        3: QuatTensor,
    }
    return cls_map[type_id].from_bytes(b)


def to_scipy_rotation(q: Quaternion | QuatVector):
    """Convert to ``scipy.spatial.transform.Rotation``.

    scipy uses ``(x, y, z, w)``; our order is ``(w, x, y, z)`` = ``(r, i, j, k)``.
    """
    from scipy.spatial.transform import Rotation
    from quat.utils import to_ndarray
    data = to_ndarray(q)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return Rotation.from_quat(data[:, [1, 2, 3, 0]])


def from_scipy_rotation(rot) -> Quaternion | QuatVector:
    """Convert ``scipy.spatial.transform.Rotation`` back to quaternion(s)."""
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
