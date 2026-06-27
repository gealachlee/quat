"""Shared serialization primitives for quaternion types."""
from __future__ import annotations
import json as _json
import struct as _struct
import numpy as np

_TYPE_IDS = {0: "Quaternion", 1: "QuatVector", 2: "QuatMatrix", 3: "QuatTensor"}
_TYPE_NAMES = {v: k for k, v in _TYPE_IDS.items()}


def _serialize_to_json(type_name: str, data: np.ndarray) -> str:
    return _json.dumps({"type": type_name, "data": data.tolist()})


def _deserialize_from_json(s: str, cls_map: dict):
    d = _json.loads(s)
    return cls_map[d["type"]](np.array(d["data"], dtype=float))


def _serialize_bytes_shaped(type_id: int, data: np.ndarray) -> bytes:
    """Binary format for collection types: type_id + ndim + shape + float64."""
    data64 = data.astype(np.float64)
    shape = np.array(data64.shape, dtype=np.int32)
    return _struct.pack('<ii', type_id, len(shape)) + shape.tobytes() + data64.tobytes()


def _deserialize_bytes_shaped(b: bytes, cls_map: dict):
    """Deserialize binary for collection types."""
    type_id, ndim = _struct.unpack_from('<ii', b, 0)
    offset = 8
    shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
    offset += ndim * 4
    size = int(np.prod(shape))
    data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64).reshape(shape)
    return cls_map[type_id](data)
