"""Serialization and interop: JSON, binary bytes, scipy Rotation, numpy views."""
import json
import struct
import numpy as np
from quat.core import Quaternion
from quat.collections import QuatVector, QuatMatrix, QuatTensor
from quat.utils import to_ndarray, from_ndarray


def to_json(q):
    """Serialize quaternion object to JSON string."""
    data = to_ndarray(q)
    result = {"type": type(q).__name__, "data": data.tolist()}
    return json.dumps(result)


def from_json(s):
    """Deserialize JSON string to quaternion object."""
    d = json.loads(s)
    arr = np.array(d["data"], dtype=float)
    return from_ndarray(arr)


def to_bytes(q):
    """Serialize quaternion object to bytes.

    Format: 4-byte header (type_id as int32, ndim as int32) + shape as int32[] + float64 data.
    type_id: 0=Quaternion, 1=QuatVector, 2=QuatMatrix, 3=QuatTensor
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


def from_bytes(b):
    """Deserialize bytes back to quaternion object."""
    type_id, ndim = struct.unpack_from('<ii', b, 0)
    offset = 8
    shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
    offset += ndim * 4
    size = int(np.prod(shape))
    data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64)
    data = data.reshape(shape)
    return from_ndarray(data)


def to_scipy_rotation(q):
    """Convert quaternion to scipy.spatial.transform.Rotation.

    Only works for unit quaternions. scipy uses (x,y,z,w) order.
    """
    from scipy.spatial.transform import Rotation
    data = to_ndarray(q)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return Rotation.from_quat(data[:, [1, 2, 3, 0]])


def from_scipy_rotation(rot):
    """Convert scipy.spatial.transform.Rotation to Quaternion or QuatVector."""
    scipy_quat = rot.as_quat()  # (n, 4) in x,y,z,w order
    if scipy_quat.ndim == 1:
        scipy_quat = scipy_quat.reshape(1, -1)
    data = np.zeros((scipy_quat.shape[0], 4))
    data[:, 0] = scipy_quat[:, 3]  # w -> r
    data[:, 1] = scipy_quat[:, 0]  # x -> i
    data[:, 2] = scipy_quat[:, 1]  # y -> j
    data[:, 3] = scipy_quat[:, 2]  # z -> k
    if data.shape[0] == 1:
        return Quaternion(data[0])
    return QuatVector(data)
