#!/usr/bin/env python3
"""Quaternion Algebra Library - Quaternion, QuatVector, QuatMatrix, QuatTensor."""
import numpy as np
from typing import Tuple, List
from numbers import Real, Complex

# ---------------------------------------------------------------------------
# Precomputed constants & vectorized Hamilton product
# ---------------------------------------------------------------------------
_CONJ  = np.array([1., -1., -1., -1.])
_REAL  = np.array([1.,  0.,  0.,  0.])
_ZERO4 = np.array([0.,  0.,  0.,  0.])

# Weight tensor for real-matrix construction: W[r,c,k] gives coefficient
# of quaternion component k at block position (r,c) in the 4x4 real matrix.
_RW = np.zeros((4, 4, 4))
_RW[0, 0, 0] = 1;   _RW[0, 1, 1] = -1;  _RW[0, 2, 2] = -1;  _RW[0, 3, 3] = -1
_RW[1, 0, 1] = 1;   _RW[1, 1, 0] = 1;   _RW[1, 2, 3] = -1;  _RW[1, 3, 2] = 1
_RW[2, 0, 2] = 1;   _RW[2, 1, 3] = 1;   _RW[2, 2, 0] = 1;   _RW[2, 3, 1] = -1
_RW[3, 0, 3] = 1;   _RW[3, 1, 2] = -1;  _RW[3, 2, 1] = 1;   _RW[3, 3, 0] = 1


def _hamilton(p, q):
    """Vectorized Hamilton product. p, q: broadcastable arrays, last dim == 4."""
    a1, b1, c1, d1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    a2, b2, c2, d2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    shp = np.broadcast_shapes(p.shape[:-1], q.shape[:-1]) + (4,)
    out = np.empty(shp)
    out[..., 0] = a1*a2 - b1*b2 - c1*c2 - d1*d2
    out[..., 1] = a1*b2 + b1*a2 + c1*d2 - d1*c2
    out[..., 2] = a1*c2 - b1*d2 + c1*a2 + d1*b2
    out[..., 3] = a1*d2 + b1*c2 - c1*b2 + d1*a2
    return out


# ---------------------------------------------------------------------------
class Quaternion:
    __slots__ = ('_data',)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            val = args[0]
            if isinstance(val, Quaternion):
                self._data = val._data.copy()
            elif isinstance(val, np.ndarray):
                arr = np.asarray(val, dtype=float).ravel()
                if arr.size == 1:
                    self._data = np.array([arr[0], 0., 0., 0.])
                elif arr.size == 4:
                    self._data = arr
                else:
                    raise ValueError(f"Expected 1 or 4 elements, got {arr.size}")
            elif isinstance(val, (list, tuple)):
                arr = np.asarray(val, dtype=float).ravel()
                if arr.size == 1:
                    self._data = np.array([arr[0], 0., 0., 0.])
                elif arr.size == 4:
                    self._data = arr
                else:
                    raise ValueError(f"Expected 1 or 4 elements, got {arr.size}")
            elif isinstance(val, Real):
                self._data = np.array([float(val), 0., 0., 0.])
            elif isinstance(val, Complex):
                self._data = np.array([val.real, val.imag, 0., 0.])
            else:
                raise TypeError(f"Cannot construct Quaternion from {type(val)}")
        elif len(args) == 4 and not kwargs:
            self._data = np.array(args, dtype=float)
        elif len(args) == 2 and not kwargs:
            self._data = np.array([args[0], args[1], 0., 0.], dtype=float)
        elif len(args) == 3 and not kwargs:
            self._data = np.array([args[0], args[1], args[2], 0.], dtype=float)
        elif not args:
            scalar = kwargs.get('scalar', kwargs.get('real', 0.0))
            vector = kwargs.get('vector', kwargs.get('imag', (0.0, 0.0, 0.0)))
            self._data = np.array(
                [float(scalar), float(vector[0]), float(vector[1]), float(vector[2])])
        else:
            raise TypeError(f"Invalid arguments: args={args}, kwargs={kwargs}")

    # -- constructors --------------------------------------------------------
    @classmethod
    def zero(cls):
        return cls(0., 0., 0., 0.)

    @classmethod
    def one(cls):
        return cls(1., 1., 1., 1.)

    @classmethod
    def from_axis_angle(cls, axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        half = angle / 2.
        s = np.sin(half)
        return cls(np.cos(half), s*axis[0], s*axis[1], s*axis[2])

    # -- accessors -----------------------------------------------------------
    @property
    def r(self):
        return self._data[0].item()

    @property
    def i(self):
        return self._data[1].item()

    @property
    def j(self):
        return self._data[2].item()

    @property
    def k(self):
        return self._data[3].item()

    @property
    def scalar(self):
        return self._data[0].item()

    @property
    def vector(self):
        return self._data[1:4].copy()

    @property
    def real(self):
        return self._data[0].item()

    @property
    def imag(self):
        return self._data[1:4].copy()

    @property
    def components(self):
        d = self._data
        return (d[0].item(), d[1].item(), d[2].item(), d[3].item())

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return self._data[idx].item()

    def __iter__(self):
        return iter(self.components)

    # -- display -------------------------------------------------------------
    def __repr__(self):
        a, b, c, d = self.components
        return f"Quaternion({a}, {b}, {c}, {d})"

    def __str__(self):
        a, b, c, d = self.components
        parts = []
        if a != 0.0 or (b == 0.0 and c == 0.0 and d == 0.0):
            parts.append(f"{a}")
        if b != 0.0:
            sign = "+" if (b > 0 and parts) else ""
            parts.append(f"{sign}{b}i")
        if c != 0.0:
            sign = "+" if (c > 0 and parts) else ""
            parts.append(f"{sign}{c}j")
        if d != 0.0:
            sign = "+" if (d > 0 and parts) else ""
            parts.append(f"{sign}{d}k")
        return "".join(parts).replace("+-", "-")

    # -- equality / hash -----------------------------------------------------
    def __eq__(self, other):
        if isinstance(other, Quaternion):
            return np.allclose(self._data, other._data)
        if isinstance(other, (Real, Complex)):
            return self == Quaternion(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(round(v, 12) for v in self.components))

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self._data + other._data)
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data[0] + r, self._data[1],
                              self._data[2], self._data[3])
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self._data - other._data)
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data[0] - r, self._data[1],
                              self._data[2], self._data[3])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(r - self._data[0], -self._data[1],
                              -self._data[2], -self._data[3])
        return NotImplemented

    def __neg__(self):
        return Quaternion(-self._data)

    def __pos__(self):
        return Quaternion(self._data.copy())

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(_hamilton(self._data, other._data))
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data * s)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data * s)
        if isinstance(other, Quaternion):
            return Quaternion(_hamilton(other._data, self._data))
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            return self * other.inverse()
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data / s)
        return NotImplemented

    # -- algebraic -----------------------------------------------------------
    def conjugate(self):
        return Quaternion(self._data * _CONJ)

    def norm(self):
        return float(np.linalg.norm(self._data))

    def norm_squared(self):
        return float(np.dot(self._data, self._data))

    def normalize(self):
        n = self.norm()
        if n == 0.0:
            raise ZeroDivisionError("Cannot normalize zero quaternion")
        return Quaternion(self._data / n)

    def inverse(self):
        n2 = self.norm_squared()
        if n2 == 0.0:
            raise ZeroDivisionError("Zero quaternion has no inverse")
        return Quaternion(self._data * _CONJ / n2)

    def exp(self):
        a, b, c, d = self._data
        v_norm = np.sqrt(b*b + c*c + d*d)
        ea = np.exp(a)
        if v_norm < 1e-15:
            return Quaternion(ea, 0., 0., 0.)
        s = np.sin(v_norm) / v_norm
        return Quaternion(ea * np.cos(v_norm), ea * s * b,
                          ea * s * c, ea * s * d)

    def log(self):
        a, b, c, d = self._data
        n = np.linalg.norm(self._data)
        v_norm = np.sqrt(b*b + c*c + d*d)
        if n < 1e-15:
            raise ValueError("Cannot compute log of zero quaternion")
        if v_norm < 1e-15:
            return Quaternion(np.log(n), 0., 0., 0.)
        theta = np.arctan2(v_norm, a)
        factor = theta / v_norm
        return Quaternion(np.log(n), factor*b, factor*c, factor*d)

    def pow(self, t):
        return (t * self.log()).exp()

    def dot(self, other):
        return float(np.dot(self._data, other._data))

    def commutator(self, other):
        return self * other - other * self

    # -- rotation ------------------------------------------------------------
    def rotate_vector(self, v):
        qv = Quaternion(0., float(v[0]), float(v[1]), float(v[2]))
        qc = self.conjugate()
        return (self * qv * qc).vector

    # -- matrix representations ----------------------------------------------
    def to_complex_matrix(self):
        a, b, c, d = self._data
        return np.array([[a + 1j*b, c + 1j*d],
                         [-c + 1j*d, a - 1j*b]])

    @classmethod
    def from_complex_matrix(cls, M):
        M = np.asarray(M)
        if M.shape != (2, 2):
            raise ValueError(f"Expected (2,2) matrix, got {M.shape}")
        a = (M[0, 0].real + M[1, 1].real) / 2.
        b = (M[0, 0].imag - M[1, 1].imag) / 2.
        c = (M[0, 1].real - M[1, 0].real) / 2.
        d = (M[0, 1].imag + M[1, 0].imag) / 2.
        return cls(a.real, b.real, c.real, d.real)

    def to_real_matrix(self):
        a, b, c, d = self._data
        return np.array([[a, -b, -c, -d],
                         [b,  a, -d,  c],
                         [c,  d,  a, -b],
                         [d, -c,  b,  a]])

    def to_real_matrix_right(self):
        a, b, c, d = self._data
        return np.array([[a, -b, -c, -d],
                         [b,  a,  d, -c],
                         [c, -d,  a,  b],
                         [d,  c, -b,  a]])

    @classmethod
    def from_real_matrix(cls, M):
        M = np.asarray(M)
        if M.shape != (4, 4):
            raise ValueError(f"Expected (4,4) matrix, got {M.shape}")
        return cls(M[0, 0], M[1, 0], M[2, 0], M[3, 0])

    def to_array(self):
        return self._data.copy()

    # -- type conversions ----------------------------------------------------
    def __float__(self):
        if np.allclose(self._data[1:], 0.):
            return float(self._data[0])
        raise ValueError(f"Cannot convert non-real quaternion {self} to float")

    def __int__(self):
        return int(self.__float__())

    def __complex__(self):
        if np.allclose(self._data[2:4], 0.):
            return complex(self._data[0], self._data[1])
        raise ValueError(f"Cannot convert {self} to complex (c,d nonzero)")

    def __bool__(self):
        return not np.allclose(self._data, 0.)

    def __abs__(self):
        return self.norm()


# ---------------------------------------------------------------------------
class QuatVector:
    __slots__ = ('_data',)

    def __init__(self, data):
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
    def zeros(cls, n):
        return cls(np.zeros((n, 4)))

    @classmethod
    def ones(cls, n):
        data = np.zeros((n, 4))
        data[:, 0] = 1.
        return cls(data)

    # -- properties ----------------------------------------------------------
    @property
    def shape(self):
        return (self._data.shape[0],)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Quaternion(self._data[idx])
        return QuatVector(self._data[idx])

    def __setitem__(self, idx, value):
        self._data[idx] = value._data

    def __iter__(self):
        return (Quaternion(row) for row in self._data)

    def to_array(self):
        return self._data.copy()

    def quaternions(self):
        return [Quaternion(row) for row in self._data]

    # -- display -------------------------------------------------------------
    def __repr__(self):
        inner = ", ".join(str(Quaternion(r)) for r in self._data[:6])
        if len(self) > 6:
            inner += f", ... ({len(self)} total)"
        return f"QuatVector([{inner}])"

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, QuatVector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError(f"Size mismatch: {len(self)} vs {len(other)}")
        return QuatVector(self._data + other._data)

    def __sub__(self, other):
        if not isinstance(other, QuatVector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError(f"Size mismatch: {len(self)} vs {len(other)}")
        return QuatVector(self._data - other._data)

    def __neg__(self):
        return QuatVector(-self._data)

    def __mul__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatVector(self._data * r)
        if isinstance(scalar, Quaternion):
            return QuatVector(_hamilton(self._data, scalar._data))
        return NotImplemented

    def __rmul__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatVector(self._data * r)
        if isinstance(scalar, Quaternion):
            return QuatVector(_hamilton(scalar._data, self._data))
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            s = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatVector(self._data / s)
        return NotImplemented

    # -- algebraic -----------------------------------------------------------
    def inner(self, other):
        if len(self) != len(other):
            raise ValueError("Size mismatch")
        return Quaternion(
            _hamilton(self._data * _CONJ, other._data).sum(axis=0))

    def dot(self, other):
        return self.inner(other)

    def norm(self):
        return float(np.sqrt(self.norm_squared()))

    def norm_squared(self):
        conj_hprod = _hamilton(self._data * _CONJ, self._data).sum(axis=0)
        return float(conj_hprod[0])

    def normalize(self):
        n = self.norm()
        if n == 0.0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        return self / n

    # -- matrix representation -----------------------------------------------
    def to_complex_matrix(self):
        n = len(self)
        M = np.zeros((2*n, 2), dtype=complex)
        for i in range(n):
            M[2*i:2*i+2, :] = Quaternion(self._data[i]).to_complex_matrix()
        return M

    def to_real_matrix(self):
        n = len(self)
        M = np.zeros((4*n, 4))
        for i in range(n):
            M[4*i:4*i+4, :] = Quaternion(self._data[i]).to_real_matrix()
        return M

    @classmethod
    def from_real_matrix(cls, M):
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0] % 4 or M.shape[1] != 4:
            raise ValueError(f"Expected (4n,4), got {M.shape}")
        n = M.shape[0] // 4
        data = np.empty((n, 4))
        for i in range(n):
            block = M[4*i:4*i+4, :]
            data[i] = Quaternion.from_real_matrix(block)._data
        return cls(data)


# ---------------------------------------------------------------------------
class QuatMatrix:
    __slots__ = ('_data', '_m', '_n')

    def __init__(self, data):
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
    def zeros(cls, m, n):
        return cls(np.zeros((m, n, 4)))

    @classmethod
    def eye(cls, n):
        data = np.zeros((n, n, 4))
        idx = np.arange(n)
        data[idx, idx, 0] = 1.
        return cls(data)

    # -- properties ----------------------------------------------------------
    @property
    def shape(self):
        return (self._m, self._n)

    def __len__(self):
        return self._m

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, j = idx
            return Quaternion(self._data[i, j])
        elif isinstance(idx, int):
            return QuatVector(self._data[idx])
        else:
            raise TypeError(f"Invalid index {idx}")

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            i, j = idx
            self._data[i, j] = value._data
        elif isinstance(idx, int):
            self._data[idx] = value._data
        else:
            raise TypeError(f"Invalid index {idx}")

    def row(self, i):
        return QuatVector(self._data[i])

    def col(self, j):
        return QuatVector(self._data[:, j])

    def to_array(self):
        return self._data.copy()

    # -- display -------------------------------------------------------------
    def __repr__(self):
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
    def __add__(self, other):
        if not isinstance(other, QuatMatrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return QuatMatrix(self._data + other._data)

    def __sub__(self, other):
        if not isinstance(other, QuatMatrix):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return QuatMatrix(self._data - other._data)

    def __neg__(self):
        return QuatMatrix(-self._data)

    def __mul__(self, other):
        if isinstance(other, QuatVector):
            if self._n != len(other):
                raise ValueError("Shape mismatch")
            return QuatVector(
                _hamilton(self._data, other._data[None, :, :]).sum(axis=1))
        if isinstance(other, QuatMatrix):
            if self._n != other._m:
                raise ValueError("Shape mismatch")
            result = np.zeros((self._m, other._n, 4))
            for l in range(self._n):
                result += _hamilton(
                    self._data[:, l, None, :],
                    other._data[None, l, :, :])
            return QuatMatrix(result)
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return QuatMatrix(self._data * s)
        if isinstance(other, Quaternion):
            return QuatMatrix(_hamilton(self._data, other._data))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return QuatMatrix(self._data * s)
        if isinstance(other, Quaternion):
            return QuatMatrix(_hamilton(other._data, self._data))
        return NotImplemented

    # -- matrix operations ---------------------------------------------------
    def transpose(self):
        return QuatMatrix(np.transpose(self._data, (1, 0, 2)))

    @property
    def T(self):
        return self.transpose()

    def conjugate(self):
        return QuatMatrix(self._data * _CONJ)

    def adjoint(self):
        t = np.transpose(self._data, (1, 0, 2))
        return QuatMatrix(t * _CONJ)

    @property
    def H(self):
        return self.adjoint()

    # -- matrix representations ----------------------------------------------
    def to_complex_matrix(self):
        M = np.zeros((2*self._m, 2*self._n), dtype=complex)
        for i in range(self._m):
            for j in range(self._n):
                M[2*i:2*i+2, 2*j:2*j+2] = \
                    Quaternion(self._data[i, j]).to_complex_matrix()
        return M

    @classmethod
    def from_complex_matrix(cls, M):
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

    def to_real_matrix(self):
        return np.einsum('rck,mnk->mrnc', _RW, self._data, optimize=True).reshape(
            4 * self._m, 4 * self._n)

    @classmethod
    def from_real_matrix(cls, M):
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

    def __init__(self, data):
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
    def zeros(cls, p, q, r):
        return cls(np.zeros((p, q, r, 4)))

    # -- properties ----------------------------------------------------------
    @property
    def shape(self):
        return (self._p, self._q, self._r)

    @property
    def ndim(self):
        return 3

    def __len__(self):
        return self._p

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 3:
            return Quaternion(self._data[idx[0], idx[1], idx[2]])
        elif isinstance(idx, int):
            return QuatMatrix(self._data[idx])
        elif isinstance(idx, tuple) and len(idx) == 2:
            return QuatVector(self._data[idx[0], idx[1]])
        else:
            raise TypeError(f"Invalid index {idx}")

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple) and len(idx) == 3:
            self._data[idx[0], idx[1], idx[2]] = value._data
        elif isinstance(idx, int):
            self._data[idx] = value._data
        elif isinstance(idx, tuple) and len(idx) == 2:
            self._data[idx[0], idx[1]] = value._data
        else:
            raise TypeError(f"Invalid index {idx}")

    def to_array(self):
        return self._data.copy()

    def __repr__(self):
        return f"QuatTensor({self._p}x{self._q}x{self._r})"

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, QuatTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return QuatTensor(self._data + other._data)

    def __sub__(self, other):
        if not isinstance(other, QuatTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return QuatTensor(self._data - other._data)

    def __neg__(self):
        return QuatTensor(-self._data)

    def __mul__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data * r)
        if isinstance(scalar, Quaternion):
            return QuatTensor(_hamilton(self._data, scalar._data))
        return NotImplemented

    def __rmul__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data * r)
        if isinstance(scalar, Quaternion):
            return QuatTensor(_hamilton(scalar._data, self._data))
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            s = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data / s)
        return NotImplemented

    # -- mode-n products -----------------------------------------------------
    def mode_1_product(self, A):
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

    def mode_2_product(self, A):
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

    def mode_3_product(self, A):
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
    def inner(self, other):
        if self.shape != other.shape:
            raise ValueError("Shape mismatch")
        return Quaternion(
            _hamilton(self._data * _CONJ, other._data)
            .sum(axis=(0, 1, 2)))

    def norm(self):
        return float(np.sqrt(self.norm_squared()))

    def norm_squared(self):
        conj_sum = _hamilton(
            self._data * _CONJ, self._data).sum(axis=(0, 1, 2))
        return float(conj_sum[0])

    # -- unfolding -----------------------------------------------------------
    def unfold(self, mode):
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

    def to_complex_matrix(self, mode=1):
        return self.unfold(mode).to_complex_matrix()

    def to_real_matrix(self, mode=1):
        return self.unfold(mode).to_real_matrix()


# ---------------------------------------------------------------------------
# Convenience function & module-level basis constants
# ---------------------------------------------------------------------------
def quat(*args):
    return Quaternion(*args)


def dict_to_quat_matrix(X_dict):
    """将四元数字典的单个样本转换为 QuatMatrix.

    X_dict: {'real': (H,W), 'i': (H,W), 'j': (H,W), 'k': (H,W)}
    Returns: QuatMatrix of shape (H, W) —— 每个像素位置一个 Quaternion 元素
    """
    data = np.stack(
        [X_dict['real'], X_dict['i'], X_dict['j'], X_dict['k']],
        axis=-1)  # (H, W, 4)
    return QuatMatrix(data)


def dict_to_quat_tensor(X_dict):
    """将四元数字典（batch）转换为 QuatTensor.

    等价于: 逐样本 dict_to_quat_matrix → QuatTensor stack

    X_dict: {'real': (n,H,W), 'i': (n,H,W), 'j': (n,H,W), 'k': (n,H,W)}
    Returns: QuatTensor of shape (n, H, W) —— mode1=样本, mode2=H, mode3=W
    """
    data = np.stack(
        [X_dict['real'], X_dict['i'], X_dict['j'], X_dict['k']],
        axis=-1)  # (n, H, W, 4)
    return QuatTensor(data)


_I    = Quaternion(0, 1, 0, 0)
_J    = Quaternion(0, 0, 1, 0)
_K    = Quaternion(0, 0, 0, 1)
_ZERO = Quaternion(0, 0, 0, 0)
_R    = Quaternion(1, 0, 0, 0)
_ONE  = Quaternion(1, 1, 1, 1)
