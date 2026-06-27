"""Shared serialization primitives for quaternion types.

Provides JSON and binary serialization helpers used by Quaternion,
QuatVector, QuatMatrix, and QuatTensor.  Each type delegates its
``to_json`` / ``from_json`` / ``to_bytes`` / ``from_bytes`` methods
to these functions, passing only its type-specific label or id.
"""
from __future__ import annotations
import json as _json
import struct as _struct
import numpy as np

_TYPE_IDS: dict[int, str] = {0: "Quaternion", 1: "QuatVector", 2: "QuatMatrix", 3: "QuatTensor"}
"""Mapping from binary type_id to type name string."""

_TYPE_NAMES: dict[str, int] = {v: k for k, v in _TYPE_IDS.items()}
"""Reverse mapping from type name to binary type_id."""


def _serialize_to_json(type_name: str, data: np.ndarray) -> str:
    """Serialize quaternion data to a JSON string.

    Args:
        type_name: one of ``"Quaternion"``, ``"QuatVector"``, etc.
        data: ndarray of shape ``(..., 4)``.

    Returns:
        JSON string ``{"type": "<type_name>", "data": [[...], ...]}``.
    """
    return _json.dumps({"type": type_name, "data": data.tolist()})


def _deserialize_from_json(s: str, cls_map: dict) -> object:
    """Deserialize a JSON string back to a quaternion type instance.

    Args:
        s: JSON string produced by ``_serialize_to_json``.
        cls_map: dict mapping type_name → constructor callable
            (e.g. ``{"QuatVector": QuatVector}``).

    Returns:
        Instance of the type indicated by the JSON ``"type"`` field.
    """
    d = _json.loads(s)
    return cls_map[d["type"]](np.array(d["data"], dtype=float))


def _serialize_bytes_shaped(type_id: int, data: np.ndarray) -> bytes:
    """Serialize quaternion data to compact binary form.

    Format: ``<i`` type_id + ``<i`` ndim + int32[ndim] shape + float64[...].

    Args:
        type_id: binary type identifier (1=QuatVector, 2=QuatMatrix, 3=QuatTensor).
        data: ndarray of shape ``(..., 4)``.

    Returns:
        Packed bytes.
    """
    data64 = data.astype(np.float64)
    shape = np.array(data64.shape, dtype=np.int32)
    return _struct.pack('<ii', type_id, len(shape)) + shape.tobytes() + data64.tobytes()


def _deserialize_bytes_shaped(b: bytes, cls_map: dict) -> object:
    """Deserialize binary bytes back to a quaternion collection type.

    Args:
        b: bytes produced by ``_serialize_bytes_shaped``.
        cls_map: dict mapping type_id → constructor callable.

    Returns:
        Instance of the type indicated by the stored *type_id*.
    """
    type_id, ndim = _struct.unpack_from('<ii', b, 0)
    offset = 8
    shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
    offset += ndim * 4
    size = int(np.prod(shape))
    data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64).reshape(shape)
    return cls_map[type_id](data)
