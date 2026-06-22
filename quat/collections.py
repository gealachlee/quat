# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Quaternion collection types — QuatVector, QuatMatrix, QuatTensor."""
from __future__ import annotations

import json as _json
import struct as _struct
import numpy as np
from typing import Tuple, List, Generator
from numbers import Real, Complex
from quat.algebra import _hamilton, _CONJ, _REAL_LEFT, _HAMILTON_TENSOR
from quat.core import Quaternion


# ---------------------------------------------------------------------------
class QuatVector:
    __slots__ = ('_data',)

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

    # -- constructors --------------------------------------------------------
    @classmethod
    def zeros(cls, n: int) -> QuatVector:
        return cls(np.zeros((n, 4)))

    @classmethod
    def ones(cls, n: int) -> QuatVector:
        data = np.zeros((n, 4))
        data[:, 0] = 1.
        return cls(data)

    # -- properties ----------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return (self._data.shape[0],)

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int | slice) -> Quaternion | QuatVector:
        if isinstance(idx, int):
            return Quaternion(self._data[idx])
        return QuatVector(self._data[idx])

    def __setitem__(self, idx: int | slice, value: Quaternion | QuatVector) -> None:
        self._data[idx] = value._data

    def __iter__(self) -> Generator[Quaternion, None, None]:
        return (Quaternion(row) for row in self._data)

    @property
    def data(self) -> np.ndarray:
        return self._data.copy()

    def to_array(self) -> np.ndarray:
        return self._data.copy()

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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

    # -- component arrays (n,) ------------------------------------------------
    @property
    def real(self) -> np.ndarray:
        return self._data[:, 0]

    @property
    def i(self) -> np.ndarray:
        return self._data[:, 1]

    @property
    def j(self) -> np.ndarray:
        return self._data[:, 2]

    @property
    def k(self) -> np.ndarray:
        return self._data[:, 3]

    def quaternions(self) -> List[Quaternion]:
        return [Quaternion(row) for row in self._data]

    # -- display -------------------------------------------------------------
    def __repr__(self) -> str:
        inner = ", ".join(str(Quaternion(r)) for r in self._data[:6])
        if len(self) > 6:
            inner += f", ... ({len(self)} total)"
        return f"QuatVector([{inner}])"

    # -- arithmetic ----------------------------------------------------------
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

    def __neg__(self) -> QuatVector:
        return QuatVector(-self._data)

    def __mul__(self, scalar: Real | Complex | Quaternion) -> QuatVector:
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatVector(self._data * r)
        if isinstance(scalar, Quaternion):
            return QuatVector(_hamilton(self._data, scalar._data))
        return NotImplemented

    def __rmul__(self, scalar: Real | Complex | Quaternion) -> QuatVector:
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatVector(self._data * r)
        if isinstance(scalar, Quaternion):
            return QuatVector(_hamilton(scalar._data, self._data))
        return NotImplemented

    def __truediv__(self, scalar: Real | Complex) -> QuatVector:
        if isinstance(scalar, (Real, Complex)):
            s = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatVector(self._data / s)
        return NotImplemented

    # -- algebraic -----------------------------------------------------------
    def inner(self, other: QuatVector) -> Quaternion:
        """Quaternion-valued inner product.

        Example:
            >>> a = QuatVector.zeros(3)
            >>> b = QuatVector.zeros(3)
            >>> ip = a.inner(b)
        """
        if len(self) != len(other):
            raise ValueError("Size mismatch")
        return Quaternion(
            _hamilton(self._data * _CONJ, other._data).sum(axis=0))


    def norm(self) -> float:
        """Frobenius norm of the vector.

        Example:
            >>> v = QuatVector.ones(3)
            >>> v.norm()
            1.732...
        """
        return float(np.sqrt(self.norm_squared()))

    def norm_squared(self) -> float:
        conj_hprod = _hamilton(self._data * _CONJ, self._data).sum(axis=0)
        return float(conj_hprod[0])

    def normalize(self) -> QuatVector:
        """Return a unit-norm copy of this vector.

        Example:
            >>> v = QuatVector.ones(3)
            >>> u = v.normalize()
            >>> u.norm()
            1.0
        """
        n = self.norm()
        if n == 0.0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        return self / n

    def isnan(self) -> np.ndarray:
        return np.any(np.isnan(self._data), axis=-1)

    def isinf(self) -> np.ndarray:
        return np.any(np.isinf(self._data), axis=-1)

    def isfinite(self) -> np.ndarray:
        return np.all(np.isfinite(self._data), axis=-1)

    def isclose(self, other: QuatVector, rtol: float = 1e-05, atol: float = 1e-08) -> np.ndarray:
        if len(self) != len(other):
            raise ValueError("Size mismatch")
        return np.isclose(self._data, other._data, rtol=rtol, atol=atol).all(axis=-1)

    # -- serialization --------------------------------------------------------
    def to_json(self) -> str:
        return _json.dumps({"type": "QuatVector", "data": self._data.tolist()})

    def to_bytes(self) -> bytes:
        data = self._data.astype(np.float64)
        shape = np.array(data.shape, dtype=np.int32)
        return _struct.pack('<ii', 1, len(shape)) + shape.tobytes() + data.tobytes()

    @classmethod
    def from_json(cls, s: str) -> "QuatVector":
        d = _json.loads(s)
        return cls(np.array(d["data"], dtype=float))

    @classmethod
    def from_bytes(cls, b: bytes) -> "QuatVector":
        type_id, ndim = _struct.unpack_from('<ii', b, 0)
        offset = 8
        shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
        offset += ndim * 4
        size = int(np.prod(shape))
        data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64).reshape(shape)
        return cls(data)

    # -- matrix representation -----------------------------------------------
    def to_complex_matrix(self) -> np.ndarray:
        n = len(self)
        M = np.zeros((2*n, 2), dtype=complex)
        for i in range(n):
            M[2*i:2*i+2, :] = Quaternion(self._data[i]).to_complex_matrix()
        return M

    def to_real_matrix_left(self) -> np.ndarray:
        """Return left-multiplication real matrix of shape (4n, 4).

        Example:
            >>> v = QuatVector.ones(2)
            >>> M = v.to_real_matrix_left()
            >>> M.shape
            (8, 4)
        """
        n = len(self)
        M = np.zeros((4*n, 4))
        for i in range(n):
            M[4*i:4*i+4, :] = Quaternion(self._data[i]).to_real_matrix_left()
        return M

    @classmethod
    def from_real_matrix_left(cls, M: np.ndarray) -> QuatVector:
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0] % 4 or M.shape[1] != 4:
            raise ValueError(f"Expected (4n,4), got {M.shape}")
        n = M.shape[0] // 4
        data = np.empty((n, 4))
        for i in range(n):
            block = M[4*i:4*i+4, :]
            data[i] = Quaternion.from_real_matrix_left(block)._data
        return cls(data)


# ---------------------------------------------------------------------------
class QuatMatrix:
    __slots__ = ('_data', '_m', '_n')

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

    # -- constructors --------------------------------------------------------
    @classmethod
    def zeros(cls, m: int, n: int) -> QuatMatrix:
        return cls(np.zeros((m, n, 4)))

    @classmethod
    def eye(cls, n: int) -> QuatMatrix:
        data = np.zeros((n, n, 4))
        idx = np.arange(n)
        data[idx, idx, 0] = 1.
        return cls(data)

    # -- properties ----------------------------------------------------------
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

    def __setitem__(self, idx: int | tuple, value: Quaternion | QuatVector) -> None:
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

    @property
    def data(self) -> np.ndarray:
        return self._data.copy()

    def to_array(self) -> np.ndarray:
        return self._data.copy()

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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

    # -- component arrays (m, n) ---------------------------------------------
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

    # -- display -------------------------------------------------------------
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

    # -- arithmetic ----------------------------------------------------------
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

    def __neg__(self) -> QuatMatrix:
        return QuatMatrix(-self._data)

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

    def __rmul__(self, other: Real | Complex | Quaternion) -> QuatMatrix:
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return QuatMatrix(self._data * s)
        if isinstance(other, Quaternion):
            return QuatMatrix(_hamilton(other._data, self._data))
        return NotImplemented

    # -- matrix operations ---------------------------------------------------
    def transpose(self) -> QuatMatrix:
        """Quaternion matrix transpose.

        Example:
            >>> A = QuatMatrix.zeros(2, 3)
            >>> A.T.shape
            (3, 2)
        """
        return QuatMatrix(np.transpose(self._data, (1, 0, 2)))

    @property
    def T(self) -> QuatMatrix:
        """Alias for transpose()."""
        return self.transpose()

    def conjugate(self) -> QuatMatrix:
        """Entrywise quaternion conjugate."""
        return QuatMatrix(self._data * _CONJ)

    def adjoint(self) -> QuatMatrix:
        """Quaternion conjugate transpose (adjoint).

        Example:
            >>> A = QuatMatrix.zeros(2, 3)
            >>> A.H.shape
            (3, 2)
        """
        t = np.transpose(self._data, (1, 0, 2))
        return QuatMatrix(t * _CONJ)

    def norm(self) -> float:
        return float(np.sqrt(self.norm_squared()))

    def norm_squared(self) -> float:
        return float((self._data * self._data).sum())

    @property
    def H(self) -> QuatMatrix:
        """Alias for adjoint()."""
        return self.adjoint()

    def isnan(self) -> np.ndarray:
        return np.any(np.isnan(self._data), axis=-1)

    def isinf(self) -> np.ndarray:
        return np.any(np.isinf(self._data), axis=-1)

    def isfinite(self) -> np.ndarray:
        return np.all(np.isfinite(self._data), axis=-1)

    def isclose(self, other: QuatMatrix, rtol: float = 1e-05, atol: float = 1e-08) -> np.ndarray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return np.isclose(self._data, other._data, rtol=rtol, atol=atol).all(axis=-1)

    # -- serialization --------------------------------------------------------
    def to_json(self) -> str:
        return _json.dumps({"type": "QuatMatrix", "data": self._data.tolist()})

    def to_bytes(self) -> bytes:
        data = self._data.astype(np.float64)
        shape = np.array(data.shape, dtype=np.int32)
        return _struct.pack('<ii', 2, len(shape)) + shape.tobytes() + data.tobytes()

    @classmethod
    def from_json(cls, s: str) -> "QuatMatrix":
        d = _json.loads(s)
        return cls(np.array(d["data"], dtype=float))

    @classmethod
    def from_bytes(cls, b: bytes) -> "QuatMatrix":
        type_id, ndim = _struct.unpack_from('<ii', b, 0)
        offset = 8
        shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
        offset += ndim * 4
        size = int(np.prod(shape))
        data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64).reshape(shape)
        return cls(data)

    # -- matrix representations ----------------------------------------------
    def to_complex_matrix(self) -> np.ndarray:
        M = np.zeros((2*self._m, 2*self._n), dtype=complex)
        for i in range(self._m):
            for j in range(self._n):
                M[2*i:2*i+2, 2*j:2*j+2] = \
                    Quaternion(self._data[i, j]).to_complex_matrix()
        return M

    @classmethod
    def from_complex_matrix(cls, M: np.ndarray) -> QuatMatrix:
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0] % 2 or M.shape[1] % 2:
            raise ValueError(f"Expected (2m,2n), got {M.shape}")
        m, n = M.shape[0] // 2, M.shape[1] // 2
        result = np.zeros((m, n, 4))
        for i in range(m):
            for j in range(n):
                block = M[2*i:2*i+2, 2*j:2*j+2]
                result[i, j] = Quaternion.from_complex_matrix(block)._data
        return cls(result)

    def to_real_matrix_left(self) -> np.ndarray:
        """Return left-multiplication real matrix of shape (4m, 4n).

        Example:
            >>> A = QuatMatrix.eye(2)
            >>> M = A.to_real_matrix_left()
            >>> M.shape
            (8, 8)
        """
        return np.einsum('rck,mnk->mrnc', _REAL_LEFT, self._data, optimize=True).reshape(
            4 * self._m, 4 * self._n)

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


# ---------------------------------------------------------------------------
class QuatTensor:
    __slots__ = ('_data', '_p', '_q', '_r')

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

    # -- constructors --------------------------------------------------------
    @classmethod
    def zeros(cls, p: int, q: int, r: int) -> QuatTensor:
        return cls(np.zeros((p, q, r, 4)))

    # -- properties ----------------------------------------------------------
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

    def __setitem__(self, idx: int | tuple, value: Quaternion | QuatMatrix | QuatVector) -> None:
        if isinstance(idx, tuple) and len(idx) == 3:
            self._data[idx[0], idx[1], idx[2]] = value._data
        elif isinstance(idx, int):
            self._data[idx] = value._data
        elif isinstance(idx, tuple) and len(idx) == 2:
            self._data[idx[0], idx[1]] = value._data
        else:
            raise TypeError(f"Invalid index {idx}")

    @property
    def data(self) -> np.ndarray:
        return self._data.copy()

    def to_array(self) -> np.ndarray:
        return self._data.copy()

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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

    # -- component tensors (n, H, W) -----------------------------------------
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

    def __repr__(self) -> str:
        return f"QuatTensor({self._p}x{self._q}x{self._r})"

    # -- arithmetic ----------------------------------------------------------
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

    def __neg__(self) -> QuatTensor:
        return QuatTensor(-self._data)

    def __mul__(self, scalar: Real | Complex | Quaternion) -> QuatTensor:
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data * r)
        if isinstance(scalar, Quaternion):
            return QuatTensor(_hamilton(self._data, scalar._data))
        return NotImplemented

    def __rmul__(self, scalar: Real | Complex | Quaternion) -> QuatTensor:
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data * r)
        if isinstance(scalar, Quaternion):
            return QuatTensor(_hamilton(scalar._data, self._data))
        return NotImplemented

    def __truediv__(self, scalar: Real | Complex) -> QuatTensor:
        if isinstance(scalar, (Real, Complex)):
            s = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data / s)
        return NotImplemented

    # -- mode-n products -----------------------------------------------------
    def mode_1_product(self, A: QuatMatrix) -> QuatTensor:
        """Contract first mode with quaternion matrix A.

        Example:
            >>> X = QuatTensor.zeros(3, 4, 5)
            >>> A = QuatMatrix.zeros(2, 3)
            >>> Y = X.mode_1_product(A)
            >>> Y.shape
            (2, 4, 5)
        """
        if not isinstance(A, QuatMatrix):
            raise TypeError("A must be QuatMatrix")
        if A.shape[1] != self._p:
            raise ValueError(
                f"A cols must match first mode {self._p}")
        result = _hamilton(
            A._data[:, :, None, None, :],
            self._data[None, :, :, :, :]
        ).sum(axis=1)
        return QuatTensor(result)

    def mode_2_product(self, A: QuatMatrix) -> QuatTensor:
        if not isinstance(A, QuatMatrix):
            raise TypeError("A must be QuatMatrix")
        if A.shape[1] != self._q:
            raise ValueError(
                f"A cols must match second mode {self._q}")
        result = _hamilton(
            A._data[None, :, :, None, :],
            self._data[:, None, :, :, :]
        ).sum(axis=2)
        return QuatTensor(result)

    def mode_3_product(self, A: QuatMatrix) -> QuatTensor:
        if not isinstance(A, QuatMatrix):
            raise TypeError("A must be QuatMatrix")
        if A.shape[1] != self._r:
            raise ValueError(
                f"A cols must match third mode {self._r}")
        result = _hamilton(
            A._data[None, None, :, :, :],
            self._data[:, :, None, :, :]
        ).sum(axis=3)
        return QuatTensor(result)

    # -- algebraic -----------------------------------------------------------
    def inner(self, other: QuatTensor) -> Quaternion:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return Quaternion(
            _hamilton(self._data * _CONJ, other._data)
            .sum(axis=(0, 1, 2)))

    def norm(self) -> float:
        return float(np.sqrt(self.norm_squared()))

    def norm_squared(self) -> float:
        conj_sum = _hamilton(
            self._data * _CONJ, self._data).sum(axis=(0, 1, 2))
        return float(conj_sum[0])

    def isnan(self) -> np.ndarray:
        return np.any(np.isnan(self._data), axis=-1)

    def isinf(self) -> np.ndarray:
        return np.any(np.isinf(self._data), axis=-1)

    def isfinite(self) -> np.ndarray:
        return np.all(np.isfinite(self._data), axis=-1)

    def isclose(self, other: QuatTensor, rtol: float = 1e-05, atol: float = 1e-08) -> np.ndarray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return np.isclose(self._data, other._data, rtol=rtol, atol=atol).all(axis=-1)

    # -- serialization --------------------------------------------------------
    def to_json(self) -> str:
        return _json.dumps({"type": "QuatTensor", "data": self._data.tolist()})

    def to_bytes(self) -> bytes:
        data = self._data.astype(np.float64)
        shape = np.array(data.shape, dtype=np.int32)
        return _struct.pack('<ii', 3, len(shape)) + shape.tobytes() + data.tobytes()

    @classmethod
    def from_json(cls, s: str) -> "QuatTensor":
        d = _json.loads(s)
        return cls(np.array(d["data"], dtype=float))

    @classmethod
    def from_bytes(cls, b: bytes) -> "QuatTensor":
        type_id, ndim = _struct.unpack_from('<ii', b, 0)
        offset = 8
        shape = np.frombuffer(b[offset:offset + ndim * 4], dtype=np.int32)
        offset += ndim * 4
        size = int(np.prod(shape))
        data = np.frombuffer(b[offset:offset + size * 8], dtype=np.float64).reshape(shape)
        return cls(data)

    # -- unfolding -----------------------------------------------------------
    def unfold(self, mode: int) -> QuatMatrix:
        """Unfold tensor along ``mode`` into a QuatMatrix.

        Example:
            >>> X = QuatTensor.zeros(3, 4, 5)
            >>> X.unfold(1).shape
            (3, 20)
            >>> X.unfold(2).shape
            (4, 15)
            >>> X.unfold(3).shape
            (5, 12)
        """
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
# Convenience function & module-level basis constants
# ---------------------------------------------------------------------------
def dict_to_quat_matrix(X_dict: dict) -> QuatMatrix:
    """Convert a single-sample quaternion dict to QuatMatrix.

    X_dict: {'real': (H,W), 'i': (H,W), 'j': (H,W), 'k': (H,W)}
    Returns: QuatMatrix of shape (H, W) — one Quaternion per pixel.

    Example:
        >>> import numpy as np
        >>> X = {'real': np.zeros((2,3)), 'i': np.zeros((2,3)),
        ...      'j': np.zeros((2,3)), 'k': np.zeros((2,3))}
        >>> M = dict_to_quat_matrix(X)
        >>> M.shape
        (2, 3)
    """
    data = np.stack(
        [X_dict['real'], X_dict['i'], X_dict['j'], X_dict['k']],
        axis=-1)  # (H, W, 4)
    return QuatMatrix(data)


def dict_to_quat_tensor(X_dict: dict) -> QuatTensor:
    """Convert a batched quaternion dict to QuatTensor.

    Equivalent to stacking per-sample dict_to_quat_matrix results.

    X_dict: {'real': (n,H,W), 'i': (n,H,W), 'j': (n,H,W), 'k': (n,H,W)}
    Returns: QuatTensor of shape (n, H, W) — mode1=sample, mode2=H, mode3=W
    """
    data = np.stack(
        [X_dict['real'], X_dict['i'], X_dict['j'], X_dict['k']],
        axis=-1)  # (n, H, W, 4)
    return QuatTensor(data)


def labels_to_quat_vector(y: np.ndarray, binary: bool = False) -> QuatVector:
    """Convert integer labels to a pure-real QuatVector.

    y: (n,) int array
    binary: if True, maps 0→-1, 1→+1

    Example:
        >>> import numpy as np
        >>> y = np.array([0, 1, 2])
        >>> v = labels_to_quat_vector(y)
        >>> v.shape
        (3,)
        >>> labels_to_quat_vector(y, binary=True).real
        array([-1.,  1.,  2.])
    """
    data = np.zeros((len(y), 4))
    vals = np.asarray(y, dtype=float)
    if binary:
        vals = np.where(vals == 0, -1., 1.)
    data[:, 0] = vals
    return QuatVector(data)
