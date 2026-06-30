# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Quaternion collection types — QuatVector, QuatMatrix, QuatTensor."""
from __future__ import annotations

import numpy as np
from typing import Tuple, List, Generator
from numbers import Real, Complex
from quat.algebra import _hamilton, _CONJ, _REAL_LEFT, _HAMILTON_TENSOR
from quat.core import Quaternion


# ---------------------------------------------------------------------------
# Shared helpers (inlined from _arrayops / _checks / _serialize)
# ---------------------------------------------------------------------------


def _data_copy(data: np.ndarray) -> np.ndarray:
    return data.copy()


def _to_array(data: np.ndarray) -> np.ndarray:
    return data.copy()


def _to_numpy(data: np.ndarray, copy: bool = True,
              dtype: np.dtype | None = None) -> np.ndarray:
    if copy is False and dtype is None:
        return data
    return np.array(data, dtype=dtype, copy=copy)


def _dispatch_collection_ufunc(
    self, ufunc, method: str, *inputs, **kwargs
) -> object:
    if method != '__call__' or kwargs.get('out') is not None:
        return NotImplemented
    a, b = (inputs[0], inputs[1]) if len(inputs) == 2 else (inputs[0], None)
    if ufunc is np.add:
        return a + b
    if ufunc is np.subtract:
        return a - b
    if ufunc is np.multiply:
        return a * b
    if ufunc is np.true_divide or ufunc is np.floor_divide:
        return a / b
    if ufunc is np.negative:
        return -a
    return NotImplemented


def _vec_isnan(data: np.ndarray) -> np.ndarray:
    return np.any(np.isnan(data), axis=-1)


def _vec_isinf(data: np.ndarray) -> np.ndarray:
    return np.any(np.isinf(data), axis=-1)


def _vec_isfinite(data: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(data), axis=-1)


def _vec_isclose(data: np.ndarray, other_data: np.ndarray,
                 rtol: float, atol: float) -> np.ndarray:
    return np.isclose(data, other_data, rtol=rtol, atol=atol).all(axis=-1)


# -- serialization helpers ----------------------------------------------------

import json as _json
import struct as _struct

_TYPE_IDS: dict[int, str] = {0: "Quaternion", 1: "QuatVector", 2: "QuatMatrix", 3: "QuatTensor"}
_TYPE_NAMES: dict[str, int] = {v: k for k, v in _TYPE_IDS.items()}


def _serialize_to_json(type_name: str, data: np.ndarray) -> str:
    return _json.dumps({"type": type_name, "data": data.tolist()})


def _deserialize_from_json(s: str, cls_map: dict) -> object:
    d = _json.loads(s)
    return cls_map[d["type"]](np.array(d["data"], dtype=float))


def _serialize_bytes_shaped(type_id: int, data: np.ndarray) -> bytes:
    data64 = data.astype(np.float64)
    shape = np.array(data64.shape, dtype=np.int32)
    return _struct.pack('<ii', type_id, len(shape)) + shape.tobytes() + data64.tobytes()


def _deserialize_bytes_shaped(b: bytes, cls_map: dict) -> object:
    type_id, ndim = _struct.unpack_from('<ii', b, 0)
    offset = 8
    shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
    offset += ndim * 4
    size = int(np.prod(shape))
    data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64).reshape(shape)
    return cls_map[type_id](data)


# =============================================================================
# Base collection
# =============================================================================


class _BaseCollection:
    """Shared base for QuatVector, QuatMatrix, QuatTensor.

    Provides NumPy interop, component accessors, scalar arithmetic,
    norm/validation, and JSON/binary serialization.  Subclasses define
    their own ``__init__``, ``shape``, ``__len__``, ``__getitem__``,
    ``__add__``, ``__sub__``, ``__mul__``, ``norm_squared``, and
    type-specific methods.
    """
    __slots__ = ('_data',)

    _TYPE_NAME: str = ""
    _TYPE_ID: int = -1

    # -- NumPy interop --------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        return _data_copy(self._data)

    def to_array(self) -> np.ndarray:
        return _to_array(self._data)

    def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
        return _to_numpy(self._data, copy=copy, dtype=dtype)

    def __array__(self, dtype: np.dtype | None = None,
                  copy: bool | None = None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _dispatch_collection_ufunc(self, ufunc, method, *inputs, **kwargs)

    # -- component accessors --------------------------------------------------

    @property
    def real(self) -> np.ndarray:
        return self._data[..., 0]

    @property
    def i(self) -> np.ndarray:
        return self._data[..., 1]

    @property
    def j(self) -> np.ndarray:
        return self._data[..., 2]

    @property
    def k(self) -> np.ndarray:
        return self._data[..., 3]

    # -- scalar arithmetic ----------------------------------------------------

    def __neg__(self):
        return self.__class__(-self._data)

    def __truediv__(self, scalar: Real | Complex):
        if isinstance(scalar, (Real, Complex)):
            s = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return self.__class__(self._data / s)
        return NotImplemented

    def __rmul__(self, other: Real | Complex | Quaternion):
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return self.__class__(self._data * r)
        if isinstance(other, Quaternion):
            return self.__class__(_hamilton(other._data, self._data))
        return NotImplemented

    # -- norms ----------------------------------------------------------------

    def norm(self) -> float:
        """Frobenius norm."""
        return float(np.sqrt(self.norm_squared()))

    def norm_squared(self) -> float:
        """Sum of squared component norms — override in subclasses."""
        raise NotImplementedError

    # -- validation -----------------------------------------------------------

    def isnan(self) -> np.ndarray:
        return _vec_isnan(self._data)

    def isinf(self) -> np.ndarray:
        return _vec_isinf(self._data)

    def isfinite(self) -> np.ndarray:
        return _vec_isfinite(self._data)

    def isclose(self, other, rtol: float = 1e-05, atol: float = 1e-08) -> np.ndarray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return _vec_isclose(self._data, other._data, rtol, atol)

    # -- serialization --------------------------------------------------------

    def to_json(self) -> str:
        return _serialize_to_json(self._TYPE_NAME, self._data)

    def to_bytes(self) -> bytes:
        return _serialize_bytes_shaped(self._TYPE_ID, self._data)

    @classmethod
    def from_json(cls, s: str):
        return _deserialize_from_json(s, {cls._TYPE_NAME: cls})

    @classmethod
    def from_bytes(cls, b: bytes):
        return _deserialize_bytes_shaped(b, {cls._TYPE_ID: cls})


# =============================================================================
# QuatVector
# =============================================================================


class QuatVector(_BaseCollection):
    __slots__ = ()
    _TYPE_NAME = "QuatVector"
    _TYPE_ID = 1

    def __init__(self, data: np.ndarray | list | tuple | QuatVector) -> None:
        if isinstance(data, QuatVector):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1 and data.size % 4 == 0:
                self._data = np.asarray(data.reshape(-1, 4), dtype=float)
            elif data.ndim == 2 and data.shape[1] == 4:
                self._data = np.asarray(data, dtype=float)
            else:
                raise ValueError(f"Invalid array shape {data.shape}")
        elif isinstance(data, (list, tuple)):
            if len(data) == 0:
                self._data = np.empty((0, 4))
            elif isinstance(data[0], Quaternion):
                self._data = np.array([q._data for q in data])
            elif isinstance(data[0], (list, tuple, np.ndarray)):
                self._data = np.asarray(data, dtype=float)
                if self._data.ndim == 1:
                    self._data = self._data.reshape(-1, 4)
            else:
                raise TypeError(f"Unsupported element type {type(data[0])}")
        else:
            raise TypeError(f"Cannot construct QuatVector from {type(data)}")

    @classmethod
    def zeros(cls, n: int) -> QuatVector:
        return cls(np.zeros((n, 4)))

    @classmethod
    def ones(cls, n: int) -> QuatVector:
        data = np.zeros((n, 4))
        data[:, 0] = 1.
        return cls(data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self._data.shape[0],)

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int | slice) -> Quaternion | QuatVector:
        if isinstance(idx, int):
            return Quaternion(self._data[idx])
        return QuatVector(self._data[idx])

    def __setitem__(self, idx: int | slice,
                    value: Quaternion | QuatVector) -> None:
        self._data[idx] = value._data

    def __iter__(self) -> Generator[Quaternion, None, None]:
        return (Quaternion(row) for row in self._data)

    def quaternions(self) -> List[Quaternion]:
        return [Quaternion(row) for row in self._data]

    # -- display --------------------------------------------------------------

    def __repr__(self) -> str:
        inner = ", ".join(str(Quaternion(r)) for r in self._data[:6])
        if len(self) > 6:
            inner += f", ... ({len(self)} total)"
        return f"QuatVector([{inner}])"

    # -- arithmetic -----------------------------------------------------------

    def __add__(self, other: QuatVector) -> QuatVector:
        if not isinstance(other, QuatVector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError(f"Size mismatch: {len(self)} vs {len(other)}")
        return QuatVector(self._data + other._data)

    def __sub__(self, other: QuatVector) -> QuatVector:
        if not isinstance(other, QuatVector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError(f"Size mismatch: {len(self)} vs {len(other)}")
        return QuatVector(self._data - other._data)

    def __mul__(self, other: QuatVector | Real | Complex | Quaternion) -> QuatVector:
        if isinstance(other, QuatVector):
            if len(self) != len(other):
                raise ValueError(f"Size mismatch: {len(self)} vs {len(other)}")
            return QuatVector(_hamilton(self._data, other._data))
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return QuatVector(self._data * r)
        if isinstance(other, Quaternion):
            return QuatVector(_hamilton(self._data, other._data))
        return NotImplemented

    # -- algebraic ------------------------------------------------------------

    def inner(self, other: QuatVector) -> Quaternion:
        if len(self) != len(other):
            raise ValueError("Size mismatch")
        return Quaternion(
            _hamilton(self._data * _CONJ, other._data).sum(axis=0))

    def norm_squared(self) -> float:
        conj_hprod = _hamilton(self._data * _CONJ, self._data).sum(axis=0)
        return float(conj_hprod[0])

    def normalize(self) -> QuatVector:
        n = self.norm()
        if n == 0.0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        return self / n

    # -- matrix representation ------------------------------------------------

    def to_complex_matrix(self) -> np.ndarray:
        n = len(self)
        a, b, c, d = (self._data[:, i] for i in range(4))
        M = np.empty((2 * n, 2), dtype=complex)
        M[0::2, 0] = a + 1j * b
        M[0::2, 1] = c + 1j * d
        M[1::2, 0] = -c + 1j * d
        M[1::2, 1] = a - 1j * b
        return M

    def to_real_matrix_left(self) -> np.ndarray:
        n = len(self)
        a, b, c, d = (self._data[:, i] for i in range(4))
        M = np.empty((4 * n, 4))
        M[0::4, 0] = a;   M[0::4, 1] = -b;  M[0::4, 2] = -c;  M[0::4, 3] = -d
        M[1::4, 0] = b;   M[1::4, 1] =  a;  M[1::4, 2] = -d;  M[1::4, 3] =  c
        M[2::4, 0] = c;   M[2::4, 1] =  d;  M[2::4, 2] =  a;  M[2::4, 3] = -b
        M[3::4, 0] = d;   M[3::4, 1] = -c;  M[3::4, 2] =  b;  M[3::4, 3] =  a
        return M

    @classmethod
    def from_real_matrix_left(cls, M: np.ndarray) -> QuatVector:
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0] % 4 or M.shape[1] != 4:
            raise ValueError(f"Expected (4n,4), got {M.shape}")
        n = M.shape[0] // 4
        data = np.empty((n, 4))
        data[:, 0] = M[0::4, 0]
        data[:, 1] = M[1::4, 0]
        data[:, 2] = M[2::4, 0]
        data[:, 3] = M[3::4, 0]
        return cls(data)


# =============================================================================
# QuatMatrix
# =============================================================================


class QuatMatrix(_BaseCollection):
    __slots__ = ('_m', '_n')
    _TYPE_NAME = "QuatMatrix"
    _TYPE_ID = 2

    def __init__(self, data: np.ndarray | list | tuple | QuatMatrix) -> None:
        if isinstance(data, QuatMatrix):
            self._data = data._data.copy()
            self._m = data._m
            self._n = data._n
        elif isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[2] == 4:
                self._data = np.asarray(data, dtype=float)
                self._m, self._n = data.shape[0], data.shape[1]
            elif data.ndim == 2 and data.shape[1] == 4:
                arr = np.asarray(data, dtype=float)
                self._data = arr.reshape(1, arr.shape[0], 4)
                self._m, self._n = 1, arr.shape[0]
            else:
                raise ValueError(f"Expected (m,n,4), got {data.shape}")
        elif isinstance(data, (list, tuple)):
            if len(data) == 0:
                self._data = np.empty((0, 0, 4))
                self._m, self._n = 0, 0
            else:
                rows = []
                for row in data:
                    r = []
                    for elem in row:
                        if isinstance(elem, Quaternion):
                            r.append(elem._data)
                        elif isinstance(elem, (list, tuple, np.ndarray)):
                            r.append(np.asarray(elem, dtype=float).ravel())
                        else:
                            raise TypeError(
                                f"Unsupported element type {type(elem)}")
                    rows.append(r)
                self._data = np.array(rows, dtype=float)
                self._m, self._n = self._data.shape[0], self._data.shape[1]
        else:
            raise TypeError(f"Cannot construct QuatMatrix from {type(data)}")

    @classmethod
    def zeros(cls, m: int, n: int) -> QuatMatrix:
        return cls(np.zeros((m, n, 4)))

    @classmethod
    def eye(cls, n: int) -> QuatMatrix:
        data = np.zeros((n, n, 4))
        idx = np.arange(n)
        data[idx, idx, 0] = 1.
        return cls(data)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._m, self._n)

    def __len__(self) -> int:
        return self._m

    def __getitem__(self, idx: int | tuple) -> Quaternion | QuatVector:
        if isinstance(idx, tuple):
            i, j = idx
            return Quaternion(self._data[i, j])
        elif isinstance(idx, int):
            return QuatVector(self._data[idx])
        else:
            raise TypeError(f"Invalid index {idx}")

    def __setitem__(self, idx: int | tuple,
                    value: Quaternion | QuatVector) -> None:
        if isinstance(idx, tuple):
            i, j = idx
            self._data[i, j] = value._data
        elif isinstance(idx, int):
            self._data[idx] = value._data
        else:
            raise TypeError(f"Invalid index {idx}")

    def row(self, i: int) -> QuatVector:
        return QuatVector(self._data[i])

    def col(self, j: int) -> QuatVector:
        return QuatVector(self._data[:, j])

    # -- display --------------------------------------------------------------

    def __repr__(self) -> str:
        lines = []
        for i in range(min(self._m, 8)):
            row_str = "  ".join(
                str(Quaternion(self._data[i, j]))
                for j in range(min(self._n, 6)))
            if self._n > 6:
                row_str += f" ... ({self._n} cols)"
            lines.append(f"  [{row_str}]")
        if self._m > 8:
            lines.append(f"  ... ({self._m} rows total)")
        return f"QuatMatrix({self._m}x{self._n}) [\n" + "\n".join(lines) + "\n]"

    # -- arithmetic -----------------------------------------------------------

    def __add__(self, other: QuatMatrix) -> QuatMatrix:
        if not isinstance(other, QuatMatrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return QuatMatrix(self._data + other._data)

    def __sub__(self, other: QuatMatrix) -> QuatMatrix:
        if not isinstance(other, QuatMatrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return QuatMatrix(self._data - other._data)

    def __mul__(self, other: QuatVector | QuatMatrix | Real | Complex | Quaternion) -> QuatVector | QuatMatrix:
        if isinstance(other, QuatVector):
            if self._n != len(other):
                raise ValueError("Shape mismatch")
            return QuatVector(
                _hamilton(self._data, other._data[None, :, :]).sum(axis=1))
        if isinstance(other, QuatMatrix):
            if self._n != other._m:
                raise ValueError("Shape mismatch")
            result = np.einsum(
                'rab,mia,inb->mnr',
                _HAMILTON_TENSOR, self._data, other._data,
                optimize=True)
            return QuatMatrix(result)
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return QuatMatrix(self._data * s)
        if isinstance(other, Quaternion):
            return QuatMatrix(_hamilton(self._data, other._data))
        return NotImplemented

    # -- matrix operations ----------------------------------------------------

    def transpose(self) -> QuatMatrix:
        return QuatMatrix(np.transpose(self._data, (1, 0, 2)))

    @property
    def T(self) -> QuatMatrix:
        return self.transpose()

    def conjugate(self) -> QuatMatrix:
        return QuatMatrix(self._data * _CONJ)

    def adjoint(self) -> QuatMatrix:
        t = np.transpose(self._data, (1, 0, 2))
        return QuatMatrix(t * _CONJ)

    @property
    def H(self) -> QuatMatrix:
        return self.adjoint()

    def norm_squared(self) -> float:
        return float((self._data * self._data).sum())

    # -- matrix representations -----------------------------------------------

    def to_complex_matrix(self) -> np.ndarray:
        a, b, c, d = (self._data[..., i] for i in range(4))
        M = np.empty((2 * self._m, 2 * self._n), dtype=complex)
        M[0::2, 0::2] = a + 1j * b
        M[0::2, 1::2] = c + 1j * d
        M[1::2, 0::2] = -c + 1j * d
        M[1::2, 1::2] = a - 1j * b
        return M

    @classmethod
    def from_complex_matrix(cls, M: np.ndarray) -> QuatMatrix:
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0] % 2 or M.shape[1] % 2:
            raise ValueError(f"Expected (2m,2n), got {M.shape}")
        m, n = M.shape[0] // 2, M.shape[1] // 2
        result = np.empty((m, n, 4))
        result[:, :, 0] = (M[0::2, 0::2].real + M[1::2, 1::2].real) / 2.
        result[:, :, 1] = (M[0::2, 0::2].imag - M[1::2, 1::2].imag) / 2.
        result[:, :, 2] = (M[0::2, 1::2].real - M[1::2, 0::2].real) / 2.
        result[:, :, 3] = (M[0::2, 1::2].imag + M[1::2, 0::2].imag) / 2.
        return cls(result)

    def to_real_matrix_left(self) -> np.ndarray:
        return np.einsum('rck,mnk->mrnc', _REAL_LEFT, self._data,
                         optimize=True).reshape(4 * self._m, 4 * self._n)

    @classmethod
    def from_real_matrix_left(cls, M: np.ndarray) -> QuatMatrix:
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0] % 4 or M.shape[1] % 4:
            raise ValueError(f"Expected (4m,4n), got {M.shape}")
        m, n = M.shape[0] // 4, M.shape[1] // 4
        result = np.empty((m, n, 4))
        result[:, :, 0] = M[0::4, 0::4]
        result[:, :, 1] = M[1::4, 0::4]
        result[:, :, 2] = M[2::4, 0::4]
        result[:, :, 3] = M[3::4, 0::4]
        return cls(result)


# =============================================================================
# QuatTensor
# =============================================================================


class QuatTensor(_BaseCollection):
    __slots__ = ('_p', '_q', '_r')
    _TYPE_NAME = "QuatTensor"
    _TYPE_ID = 3

    def __init__(self, data: np.ndarray | list | tuple | QuatTensor) -> None:
        if isinstance(data, QuatTensor):
            self._data = data._data.copy()
            self._p, self._q, self._r = data._p, data._q, data._r
        elif isinstance(data, np.ndarray):
            if data.ndim == 4 and data.shape[3] == 4:
                self._data = np.asarray(data, dtype=float)
                self._p, self._q, self._r = \
                    data.shape[0], data.shape[1], data.shape[2]
            elif data.ndim == 3 and data.shape[2] == 4:
                self._data = np.asarray(data, dtype=float)
                self._p, self._q, self._r = \
                    data.shape[0], 1, data.shape[1]
            elif data.ndim == 2 and data.shape[1] == 4:
                arr = np.asarray(data, dtype=float)
                self._data = arr.reshape(1, 1, arr.shape[0], 4)
                self._p, self._q, self._r = 1, 1, arr.shape[0]
            else:
                raise ValueError(
                    f"Expected (p,q,r,4), got {data.shape}")
        elif isinstance(data, (list, tuple)):
            if len(data) == 0:
                self._data = np.empty((0, 0, 0, 4))
                self._p, self._q, self._r = 0, 0, 0
            else:
                raw = []
                for page in data:
                    pg = []
                    for row in page:
                        rw = []
                        for elem in row:
                            if isinstance(elem, Quaternion):
                                rw.append(elem._data)
                            elif isinstance(elem, (list, tuple, np.ndarray)):
                                rw.append(
                                    np.asarray(elem, dtype=float).ravel())
                            else:
                                raise TypeError(
                                    f"Unsupported element type {type(elem)}")
                        pg.append(rw)
                    raw.append(pg)
                self._data = np.array(raw, dtype=float)
                self._p = self._data.shape[0]
                self._q = self._data.shape[1]
                self._r = self._data.shape[2]
        else:
            raise TypeError(
                f"Cannot construct QuatTensor from {type(data)}")

    @classmethod
    def zeros(cls, p: int, q: int, r: int) -> QuatTensor:
        return cls(np.zeros((p, q, r, 4)))

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._p, self._q, self._r)

    @property
    def ndim(self) -> int:
        return 3

    def __len__(self) -> int:
        return self._p

    def __getitem__(self, idx: int | tuple) -> Quaternion | QuatMatrix | QuatVector:
        if isinstance(idx, tuple) and len(idx) == 3:
            return Quaternion(self._data[idx[0], idx[1], idx[2]])
        elif isinstance(idx, int):
            return QuatMatrix(self._data[idx])
        elif isinstance(idx, tuple) and len(idx) == 2:
            return QuatVector(self._data[idx[0], idx[1]])
        else:
            raise TypeError(f"Invalid index {idx}")

    def __setitem__(self, idx: int | tuple,
                    value: Quaternion | QuatMatrix | QuatVector) -> None:
        if isinstance(idx, tuple) and len(idx) == 3:
            self._data[idx[0], idx[1], idx[2]] = value._data
        elif isinstance(idx, int):
            self._data[idx] = value._data
        elif isinstance(idx, tuple) and len(idx) == 2:
            self._data[idx[0], idx[1]] = value._data
        else:
            raise TypeError(f"Invalid index {idx}")

    # -- display --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"QuatTensor({self._p}x{self._q}x{self._r})"

    # -- arithmetic -----------------------------------------------------------

    def __add__(self, other: QuatTensor) -> QuatTensor:
        if not isinstance(other, QuatTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return QuatTensor(self._data + other._data)

    def __sub__(self, other: QuatTensor) -> QuatTensor:
        if not isinstance(other, QuatTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return QuatTensor(self._data - other._data)

    def __mul__(self, scalar: Real | Complex | Quaternion) -> QuatTensor:
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data * r)
        if isinstance(scalar, Quaternion):
            return QuatTensor(_hamilton(self._data, scalar._data))
        return NotImplemented

    # -- algebraic ------------------------------------------------------------

    def inner(self, other: QuatTensor) -> Quaternion:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return Quaternion(
            _hamilton(self._data * _CONJ, other._data)
            .sum(axis=(0, 1, 2)))

    def norm_squared(self) -> float:
        conj_sum = _hamilton(
            self._data * _CONJ, self._data).sum(axis=(0, 1, 2))
        return float(conj_sum[0])

    # -- mode-n products ------------------------------------------------------

    def mode_1_product(self, A: QuatMatrix) -> QuatTensor:
        if not isinstance(A, QuatMatrix):
            raise TypeError("A must be QuatMatrix")
        if A.shape[1] != self._p:
            raise ValueError(f"A cols must match first mode {self._p}")
        result = _hamilton(
            A._data[:, :, None, None, :],
            self._data[None, :, :, :, :]
        ).sum(axis=1)
        return QuatTensor(result)

    def mode_2_product(self, A: QuatMatrix) -> QuatTensor:
        if not isinstance(A, QuatMatrix):
            raise TypeError("A must be QuatMatrix")
        if A.shape[1] != self._q:
            raise ValueError(f"A cols must match second mode {self._q}")
        result = _hamilton(
            A._data[None, :, :, None, :],
            self._data[:, None, :, :, :]
        ).sum(axis=2)
        return QuatTensor(result)

    def mode_3_product(self, A: QuatMatrix) -> QuatTensor:
        if not isinstance(A, QuatMatrix):
            raise TypeError("A must be QuatMatrix")
        if A.shape[1] != self._r:
            raise ValueError(f"A cols must match third mode {self._r}")
        result = _hamilton(
            A._data[None, None, :, :, :],
            self._data[:, :, None, :, :]
        ).sum(axis=3)
        return QuatTensor(result)

    # -- unfolding ------------------------------------------------------------

    def unfold(self, mode: int) -> QuatMatrix:
        if mode == 1:
            return QuatMatrix(
                self._data.transpose(0, 1, 2, 3)
                .reshape(self._p, self._q * self._r, 4))
        elif mode == 2:
            return QuatMatrix(
                self._data.transpose(1, 0, 2, 3)
                .reshape(self._q, self._p * self._r, 4))
        elif mode == 3:
            return QuatMatrix(
                self._data.transpose(2, 0, 1, 3)
                .reshape(self._r, self._p * self._q, 4))
        else:
            raise ValueError(f"mode must be 1,2,3, got {mode}")

    def to_complex_matrix(self, mode: int = 1) -> np.ndarray:
        return self.unfold(mode).to_complex_matrix()

    def to_real_matrix_left(self, mode: int = 1) -> np.ndarray:
        return self.unfold(mode).to_real_matrix_left()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def dict_to_quat_matrix(X_dict: dict) -> QuatMatrix:
    """Convert a single-sample quaternion dict to QuatMatrix.

    X_dict: {'real': (H,W), 'i': (H,W), 'j': (H,W), 'k': (H,W)}
    Returns: QuatMatrix of shape (H, W) — one Quaternion per pixel.
    """
    data = np.stack(
        [X_dict['real'], X_dict['i'], X_dict['j'], X_dict['k']],
        axis=-1)
    return QuatMatrix(data)


def dict_to_quat_tensor(X_dict: dict) -> QuatTensor:
    """Convert a batched quaternion dict to QuatTensor.

    X_dict: {'real': (n,H,W), 'i': (n,H,W), 'j': (n,H,W), 'k': (n,H,W)}
    Returns: QuatTensor of shape (n, H, W)
    """
    data = np.stack(
        [X_dict['real'], X_dict['i'], X_dict['j'], X_dict['k']],
        axis=-1)
    return QuatTensor(data)


def labels_to_quat_vector(y: np.ndarray, binary: bool = False) -> QuatVector:
    """Convert integer labels to a pure-real QuatVector.

    y: (n,) int array
    binary: if True, maps 0→-1, 1→+1
    """
    data = np.zeros((len(y), 4))
    vals = np.asarray(y, dtype=float)
    if binary:
        vals = np.where(vals == 0, -1., 1.)
    data[:, 0] = vals
    return QuatVector(data)
