#!/usr/bin/env python3
"""Quaternion Algebra Library - Quaternion, QuatVector, QuatMatrix, QuatTensor."""
import numpy as np
from typing import Tuple, List
from numbers import Real, Complex

class Quaternion:
    __slots__ = ('_data',)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            val = args[0]
            if isinstance(val, Quaternion):
                self._data = val._data.copy()
            elif isinstance(val, (np.ndarray, list, tuple)):
                arr = np.asarray(val, dtype=float).ravel()
                if arr.size == 1:
                    self._data = np.array([arr[0], 0.0, 0.0, 0.0])
                elif arr.size == 4:
                    self._data = arr.astype(float)
                else:
                    raise ValueError(f"Expected 1 or 4 elements, got {arr.size}")
            elif isinstance(val, Real):
                self._data = np.array([float(val), 0.0, 0.0, 0.0])
            elif isinstance(val, Complex):
                self._data = np.array([val.real, val.imag, 0.0, 0.0])
            else:
                raise TypeError(f"Cannot construct Quaternion from {type(val)}")
        elif len(args) == 4 and not kwargs:
            self._data = np.array([float(args[0]), float(args[1]),
                                   float(args[2]), float(args[3])])
        elif len(args) == 2 and not kwargs:
            self._data = np.array([float(args[0]), float(args[1]), 0.0, 0.0])
        elif len(args) == 3 and not kwargs:
            self._data = np.array([float(args[0]), float(args[1]),
                                   float(args[2]), 0.0])
        elif not args:
            scalar = kwargs.get('scalar', kwargs.get('real', 0.0))
            vector = kwargs.get('vector', kwargs.get('imag', (0.0, 0.0, 0.0)))
            self._data = np.array([float(scalar),
                                   float(vector[0]), float(vector[1]), float(vector[2])])
        else:
            raise TypeError(f"Invalid arguments: args={args}, kwargs={kwargs}")

    @classmethod
    def zero(cls): return cls(0.0, 0.0, 0.0, 0.0)
    @classmethod
    def one(cls): return cls(1.0, 1.0, 1.0, 1.0)
    @classmethod
    def from_axis_angle(cls, axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        half = angle / 2.0
        s = np.sin(half)
        return cls(np.cos(half), float(s * axis[0]), float(s * axis[1]), float(s * axis[2]))

    @property
    def r(self): return float(self._data[0])
    @property
    def i(self): return float(self._data[1])
    @property
    def j(self): return float(self._data[2])
    @property
    def k(self): return float(self._data[3])
    @property
    def scalar(self): return float(self._data[0])
    @property
    def vector(self): return self._data[1:4].copy()
    @property
    def real(self): return float(self._data[0])
    @property
    def imag(self): return self._data[1:4].copy()
    @property
    def components(self):
        d = self._data
        return (float(d[0]), float(d[1]), float(d[2]), float(d[3]))
    def __len__(self): return 4
    def __getitem__(self, idx): return float(self._data[idx])
    def __iter__(self): return iter(self.components)

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

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            return np.allclose(self._data, other._data)
        if isinstance(other, (Real, Complex)):
            return self == Quaternion(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(round(v, 12) for v in self.components))

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self._data + other._data)
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data[0] + r, self._data[1], self._data[2], self._data[3])
        return NotImplemented

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self._data - other._data)
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data[0] - r, self._data[1], self._data[2], self._data[3])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(r - self._data[0], -self._data[1], -self._data[2], -self._data[3])
        return NotImplemented

    def __neg__(self): return Quaternion(-self._data)
    def __pos__(self): return Quaternion(self._data.copy())

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            a1, b1, c1, d1 = self._data
            a2, b2, c2, d2 = other._data
            return Quaternion(
                a1*a2 - b1*b2 - c1*c2 - d1*d2,
                a1*b2 + b1*a2 + c1*d2 - d1*c2,
                a1*c2 - b1*d2 + c1*a2 + d1*b2,
                a1*d2 + b1*c2 - c1*b2 + d1*a2,
            )
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data * s)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data * s)
        if isinstance(other, Quaternion):
            return other.__mul__(self)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            return self * other.inverse()
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data / s)
        return NotImplemented

    def conjugate(self):
        return Quaternion(self._data[0], -self._data[1], -self._data[2], -self._data[3])

    def norm(self): return float(np.sqrt(np.sum(self._data**2)))
    def norm_squared(self): return float(np.sum(self._data**2))

    def normalize(self):
        n = self.norm()
        if n == 0.0: raise ZeroDivisionError("Cannot normalize zero quaternion")
        return Quaternion(self._data / n)

    def inverse(self):
        n2 = self.norm_squared()
        if n2 == 0.0: raise ZeroDivisionError("Zero quaternion has no inverse")
        return Quaternion(self._data[0]/n2, -self._data[1]/n2, -self._data[2]/n2, -self._data[3]/n2)

    def exp(self):
        a, b, c, d = self.components
        v_norm = np.sqrt(b*b + c*c + d*d)
        ea = np.exp(a)
        if v_norm < 1e-15: return Quaternion(ea, 0.0, 0.0, 0.0)
        s = np.sin(v_norm) / v_norm
        return Quaternion(ea * np.cos(v_norm), ea * s * b, ea * s * c, ea * s * d)

    def log(self):
        a, b, c, d = self.components
        n = self.norm()
        v_norm = np.sqrt(b*b + c*c + d*d)
        if n < 1e-15: raise ValueError("Cannot compute log of zero quaternion")
        if v_norm < 1e-15: return Quaternion(np.log(n), 0.0, 0.0, 0.0)
        theta = np.arctan2(v_norm, a)
        factor = theta / v_norm
        return Quaternion(np.log(n), factor * b, factor * c, factor * d)

    def pow(self, t): return (t * self.log()).exp()
    def dot(self, other): return float(np.dot(self._data, other._data))
    def commutator(self, other): return self * other - other * self

    def rotate_vector(self, v):
        qv = Quaternion(0.0, float(v[0]), float(v[1]), float(v[2]))
        result = self * qv * self.conjugate()
        return result.vector

    def to_complex_matrix(self):
        a, b, c, d = self.components
        return np.array([[a + 1j*b, c + 1j*d], [-c + 1j*d, a - 1j*b]])

    @classmethod
    def from_complex_matrix(cls, M):
        M = np.asarray(M)
        if M.shape != (2, 2): raise ValueError(f"Expected (2,2) matrix, got {M.shape}")
        a = (M[0,0].real + M[1,1].real) / 2.0
        b = (M[0,0].imag - M[1,1].imag) / 2.0
        c = (M[0,1].real - M[1,0].real) / 2.0
        d = (M[0,1].imag + M[1,0].imag) / 2.0
        return cls(a.real, b.real, c.real, d.real)

    def to_real_matrix(self):
        a, b, c, d = self.components
        return np.array([[a,-b,-c,-d],[b,a,-d,c],[c,d,a,-b],[d,-c,b,a]])

    def to_real_matrix_right(self):
        a, b, c, d = self.components
        return np.array([[a,-b,-c,-d],[b,a,d,-c],[c,-d,a,b],[d,c,-b,a]])

    @classmethod
    def from_real_matrix(cls, M):
        M = np.asarray(M)
        if M.shape != (4,4): raise ValueError(f"Expected (4,4) matrix, got {M.shape}")
        return cls(float(M[0,0]), float(M[1,0]), float(M[2,0]), float(M[3,0]))

    def to_array(self): return self._data.copy()

    def __float__(self):
        if np.allclose(self._data[1:], 0.0): return float(self._data[0])
        raise ValueError(f"Cannot convert non-real quaternion {self} to float")
    def __int__(self): return int(self.__float__())
    def __complex__(self):
        if np.allclose(self._data[2:4], 0.0): return complex(self._data[0], self._data[1])
        raise ValueError(f"Cannot convert {self} to complex (c,d nonzero)")
    def __bool__(self): return not np.allclose(self._data, 0.0)
    def __abs__(self): return self.norm()


class QuatVector:
    __slots__ = ('_data',)

    def __init__(self, data):
        if isinstance(data, QuatVector):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1 and data.size % 4 == 0:
                self._data = data.reshape(-1, 4).astype(float)
            elif data.ndim == 2 and data.shape[1] == 4:
                self._data = data.astype(float)
            else: raise ValueError(f"Invalid array shape {data.shape}")
        elif isinstance(data, (list, tuple)):
            if len(data) == 0: self._data = np.empty((0, 4))
            elif isinstance(data[0], Quaternion):
                self._data = np.array([q._data for q in data])
            elif isinstance(data[0], (list, tuple, np.ndarray)):
                self._data = np.array(data, dtype=float)
                if self._data.ndim == 1: self._data = self._data.reshape(-1, 4)
            else: raise TypeError(f"Unsupported element type {type(data[0])}")
        else: raise TypeError(f"Cannot construct QuatVector from {type(data)}")

    @classmethod
    def zeros(cls, n): return cls(np.zeros((n, 4)))
    @classmethod
    def ones(cls, n):
        data = np.zeros((n, 4)); data[:,0] = 1.0; return cls(data)

    @property
    def shape(self): return (self._data.shape[0],)
    def __len__(self): return self._data.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, int): return Quaternion(self._data[idx])
        return QuatVector(self._data[idx])

    def __setitem__(self, idx, value): self._data[idx] = value._data

    def __iter__(self): return (Quaternion(row) for row in self._data)
    def to_array(self): return self._data.copy()
    def quaternions(self): return [Quaternion(row) for row in self._data]

    def __repr__(self):
        inner = ", ".join(str(Quaternion(r)) for r in self._data[:6])
        if len(self) > 6: inner += f", ... ({len(self)} total)"
        return f"QuatVector([{inner}])"

    def __add__(self, other):
        if not isinstance(other, QuatVector): return NotImplemented
        if len(self) != len(other): raise ValueError(f"Size mismatch: {len(self)} vs {len(other)}")
        return QuatVector(self._data + other._data)

    def __sub__(self, other):
        if not isinstance(other, QuatVector): return NotImplemented
        if len(self) != len(other): raise ValueError(f"Size mismatch: {len(self)} vs {len(other)}")
        return QuatVector(self._data - other._data)

    def __neg__(self): return QuatVector(-self._data)

    def __mul__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatVector(self._data * r)
        if isinstance(scalar, Quaternion):
            result = np.zeros_like(self._data)
            for i in range(len(self)): result[i] = (Quaternion(self._data[i]) * scalar)._data
            return QuatVector(result)
        return NotImplemented

    def __rmul__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatVector(self._data * r)
        if isinstance(scalar, Quaternion):
            result = np.zeros_like(self._data)
            for i in range(len(self)): result[i] = (scalar * Quaternion(self._data[i]))._data
            return QuatVector(result)
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            s = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatVector(self._data / s)
        return NotImplemented

    def inner(self, other):
        if len(self) != len(other): raise ValueError(f"Size mismatch")
        result = Quaternion.zero()
        for i in range(len(self)):
            result = result + Quaternion(self._data[i]).conjugate() * Quaternion(other._data[i])
        return result

    def dot(self, other): return self.inner(other)

    def norm(self): return float(np.sqrt(self.inner(self).r))

    def norm_squared(self): return float(self.inner(self).r)

    def normalize(self):
        n = self.norm()
        if n == 0.0: raise ZeroDivisionError("Cannot normalize zero vector")
        return self / n

    def to_complex_matrix(self):
        n = len(self); M = np.zeros((2*n, 2), dtype=complex)
        for i in range(n): M[2*i:2*i+2,:] = Quaternion(self._data[i]).to_complex_matrix()
        return M


class QuatMatrix:
    __slots__ = ('_data', '_m', '_n')

    def __init__(self, data):
        if isinstance(data, QuatMatrix):
            self._data = data._data.copy(); self._m = data._m; self._n = data._n
        elif isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[2] == 4:
                self._data = data.astype(float); self._m, self._n = data.shape[0], data.shape[1]
            elif data.ndim == 2 and data.shape[1] == 4:
                self._data = data.reshape(1, data.shape[0], 4).astype(float); self._m, self._n = 1, data.shape[0]
            else: raise ValueError(f"Expected (m,n,4), got {data.shape}")
        elif isinstance(data, (list, tuple)):
            if len(data) == 0: self._data = np.empty((0,0,4)); self._m, self._n = 0, 0
            else:
                rows = []
                for row in data:
                    r = []
                    for elem in row:
                        if isinstance(elem, Quaternion): r.append(elem._data)
                        elif isinstance(elem, (list, tuple, np.ndarray)): r.append(np.asarray(elem, dtype=float).ravel())
                        else: raise TypeError(f"Unsupported element type {type(elem)}")
                    rows.append(r)
                self._data = np.array(rows, dtype=float); self._m, self._n = self._data.shape[0], self._data.shape[1]
        else: raise TypeError(f"Cannot construct QuatMatrix from {type(data)}")

    @classmethod
    def zeros(cls, m, n): return cls(np.zeros((m, n, 4)))
    @classmethod
    def eye(cls, n):
        data = np.zeros((n, n, 4))
        for i in range(n): data[i,i,0] = 1.0
        return cls(data)

    @property
    def shape(self): return (self._m, self._n)
    def __len__(self): return self._m

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, j = idx; return Quaternion(self._data[i, j])
        elif isinstance(idx, int): return QuatVector(self._data[idx])
        else: raise TypeError(f"Invalid index {idx}")

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple): i, j = idx; self._data[i, j] = value._data
        elif isinstance(idx, int): self._data[idx] = value._data
        else: raise TypeError(f"Invalid index {idx}")

    def row(self, i): return QuatVector(self._data[i])
    def col(self, j): return QuatVector(self._data[:, j])
    def to_array(self): return self._data.copy()

    def __repr__(self):
        lines = []
        for i in range(min(self._m, 8)):
            row_str = "  ".join(str(Quaternion(self._data[i, j])) for j in range(min(self._n, 6)))
            if self._n > 6: row_str += f" ... ({self._n} cols)"
            lines.append(f"  [{row_str}]")
        if self._m > 8: lines.append(f"  ... ({self._m} rows total)")
        return f"QuatMatrix({self._m}x{self._n}) [\n" + "\n".join(lines) + "\n]"

    def __add__(self, other):
        if not isinstance(other, QuatMatrix): return NotImplemented
        if self.shape != other.shape: raise ValueError(f"Shape mismatch")
        return QuatMatrix(self._data + other._data)

    def __sub__(self, other):
        if not isinstance(other, QuatMatrix): return NotImplemented
        if self.shape != other.shape: raise ValueError(f"Shape mismatch")
        return QuatMatrix(self._data - other._data)

    def __neg__(self): return QuatMatrix(-self._data)

    def __mul__(self, other):
        if isinstance(other, QuatVector):
            if self._n != len(other): raise ValueError(f"Shape mismatch")
            result = np.zeros((self._m, 4))
            for i in range(self._m):
                acc = Quaternion.zero()
                for k in range(self._n): acc = acc + Quaternion(self._data[i,k]) * Quaternion(other._data[k])
                result[i] = acc._data
            return QuatVector(result)
        if isinstance(other, QuatMatrix):
            if self._n != other._m: raise ValueError(f"Shape mismatch")
            result = np.zeros((self._m, other._n, 4))
            for i in range(self._m):
                for j in range(other._n):
                    acc = Quaternion.zero()
                    for k in range(self._n): acc = acc + Quaternion(self._data[i,k]) * Quaternion(other._data[k,j])
                    result[i,j] = acc._data
            return QuatMatrix(result)
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return QuatMatrix(self._data * s)
        if isinstance(other, Quaternion):
            result_data = np.zeros_like(self._data)
            for i in range(self._m):
                for j in range(self._n): result_data[i,j] = (Quaternion(self._data[i,j]) * other)._data
            return QuatMatrix(result_data)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return QuatMatrix(self._data * s)
        if isinstance(other, Quaternion):
            result_data = np.zeros_like(self._data)
            for i in range(self._m):
                for j in range(self._n): result_data[i,j] = (other * Quaternion(self._data[i,j]))._data
            return QuatMatrix(result_data)
        return NotImplemented

    def transpose(self): return QuatMatrix(np.transpose(self._data, (1,0,2)))
    @property
    def T(self): return self.transpose()

    def conjugate(self):
        result = self._data.copy(); result[:,:,1:] *= -1.0; return QuatMatrix(result)

    def adjoint(self):
        t = np.transpose(self._data, (1,0,2)); t[:,:,1:] *= -1.0; return QuatMatrix(t)

    @property
    def H(self): return self.adjoint()

    def to_complex_matrix(self):
        M = np.zeros((2*self._m, 2*self._n), dtype=complex)
        for i in range(self._m):
            for j in range(self._n): M[2*i:2*i+2, 2*j:2*j+2] = Quaternion(self._data[i,j]).to_complex_matrix()
        return M

    @classmethod
    def from_complex_matrix(cls, M):
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0]%2 or M.shape[1]%2: raise ValueError(f"Expected (2m,2n), got {M.shape}")
        m, n = M.shape[0]//2, M.shape[1]//2
        result = np.zeros((m,n,4))
        for i in range(m):
            for j in range(n):
                block = M[2*i:2*i+2, 2*j:2*j+2]; result[i,j] = Quaternion.from_complex_matrix(block)._data
        return cls(result)

    def to_real_matrix(self):
        M = np.zeros((4*self._m, 4*self._n))
        for i in range(self._m):
            for j in range(self._n): M[4*i:4*i+4, 4*j:4*j+4] = Quaternion(self._data[i,j]).to_real_matrix()
        return M

    @classmethod
    def from_real_matrix(cls, M):
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0]%4 or M.shape[1]%4: raise ValueError(f"Expected (4m,4n), got {M.shape}")
        m, n = M.shape[0]//4, M.shape[1]//4
        result = np.zeros((m,n,4))
        for i in range(m):
            for j in range(n):
                block = M[4*i:4*i+4, 4*j:4*j+4]; result[i,j] = Quaternion.from_real_matrix(block)._data
        return cls(result)


class QuatTensor:
    __slots__ = ('_data', '_p', '_q', '_r')

    def __init__(self, data):
        if isinstance(data, QuatTensor):
            self._data = data._data.copy(); self._p, self._q, self._r = data._p, data._q, data._r
        elif isinstance(data, np.ndarray):
            if data.ndim == 4 and data.shape[3] == 4:
                self._data = data.astype(float); self._p, self._q, self._r = data.shape[0], data.shape[1], data.shape[2]
            elif data.ndim == 3 and data.shape[2] == 4:
                self._data = data.astype(float); self._p, self._q, self._r = data.shape[0], 1, data.shape[1]
            elif data.ndim == 2 and data.shape[1] == 4:
                self._data = data.reshape(1,1,data.shape[0],4).astype(float); self._p, self._q, self._r = 1,1,data.shape[0]
            else: raise ValueError(f"Expected (p,q,r,4), got {data.shape}")
        elif isinstance(data, (list, tuple)):
            if len(data) == 0: self._data = np.empty((0,0,0,4)); self._p, self._q, self._r = 0,0,0
            else:
                raw = []
                for page in data:
                    pg = []
                    for row in page:
                        rw = []
                        for elem in row:
                            if isinstance(elem, Quaternion): rw.append(elem._data)
                            elif isinstance(elem, (list,tuple,np.ndarray)): rw.append(np.asarray(elem,dtype=float).ravel())
                            else: raise TypeError(f"Unsupported element type {type(elem)}")
                        pg.append(rw)
                    raw.append(pg)
                self._data = np.array(raw, dtype=float)
                self._p, self._q, self._r = self._data.shape[0], self._data.shape[1], self._data.shape[2]
        else: raise TypeError(f"Cannot construct QuatTensor from {type(data)}")

    @classmethod
    def zeros(cls, p, q, r): return cls(np.zeros((p, q, r, 4)))

    @property
    def shape(self): return (self._p, self._q, self._r)
    @property
    def ndim(self): return 3
    def __len__(self): return self._p

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 3: return Quaternion(self._data[idx[0], idx[1], idx[2]])
        elif isinstance(idx, int): return QuatMatrix(self._data[idx])
        elif isinstance(idx, tuple) and len(idx) == 2: return QuatVector(self._data[idx[0], idx[1]])
        else: raise TypeError(f"Invalid index {idx}")

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple) and len(idx) == 3: self._data[idx[0], idx[1], idx[2]] = value._data
        elif isinstance(idx, int): self._data[idx] = value._data
        elif isinstance(idx, tuple) and len(idx) == 2: self._data[idx[0], idx[1]] = value._data
        else: raise TypeError(f"Invalid index {idx}")

    def to_array(self): return self._data.copy()
    def __repr__(self): return f"QuatTensor({self._p}x{self._q}x{self._r})"

    def __add__(self, other):
        if not isinstance(other, QuatTensor): return NotImplemented
        if self.shape != other.shape: raise ValueError(f"Shape mismatch")
        return QuatTensor(self._data + other._data)

    def __sub__(self, other):
        if not isinstance(other, QuatTensor): return NotImplemented
        if self.shape != other.shape: raise ValueError(f"Shape mismatch")
        return QuatTensor(self._data - other._data)

    def __neg__(self): return QuatTensor(-self._data)

    def __mul__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data * r)
        if isinstance(scalar, Quaternion):
            result = np.zeros_like(self._data)
            for ijk in np.ndindex(self._p, self._q, self._r): result[ijk] = (Quaternion(self._data[ijk]) * scalar)._data
            return QuatTensor(result)
        return NotImplemented

    def __rmul__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            r = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data * r)
        if isinstance(scalar, Quaternion):
            result = np.zeros_like(self._data)
            for ijk in np.ndindex(self._p, self._q, self._r): result[ijk] = (scalar * Quaternion(self._data[ijk]))._data
            return QuatTensor(result)
        return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (Real, Complex)):
            s = float(scalar.real if isinstance(scalar, Complex) else scalar)
            return QuatTensor(self._data / s)
        return NotImplemented

    def mode_1_product(self, A):
        if not isinstance(A, QuatMatrix): raise TypeError("A must be QuatMatrix")
        if A.shape[1] != self._p: raise ValueError(f"A cols must match first mode {self._p}")
        new_p = A.shape[0]; result = np.zeros((new_p, self._q, self._r, 4))
        for j in range(self._q):
            for k in range(self._r):
                col = QuatVector(self._data[:, j, k, :]); result[:, j, k, :] = (A * col)._data
        return QuatTensor(result)

    def mode_2_product(self, A):
        if not isinstance(A, QuatMatrix): raise TypeError("A must be QuatMatrix")
        if A.shape[1] != self._q: raise ValueError(f"A cols must match second mode {self._q}")
        new_q = A.shape[0]; result = np.zeros((self._p, new_q, self._r, 4))
        for i in range(self._p):
            for k in range(self._r):
                row = QuatVector(self._data[i, :, k, :]); result[i, :, k, :] = (A * row)._data
        return QuatTensor(result)

    def mode_3_product(self, A):
        if not isinstance(A, QuatMatrix): raise TypeError("A must be QuatMatrix")
        if A.shape[1] != self._r: raise ValueError(f"A cols must match third mode {self._r}")
        new_r = A.shape[0]; result = np.zeros((self._p, self._q, new_r, 4))
        for i in range(self._p):
            for j in range(self._q):
                tube = QuatVector(self._data[i, j, :, :]); result[i, j, :, :] = (A * tube)._data
        return QuatTensor(result)

    def inner(self, other):
        if self.shape != other.shape: raise ValueError(f"Shape mismatch")
        acc = Quaternion.zero()
        for ijk in np.ndindex(self._p, self._q, self._r):
            acc = acc + Quaternion(self._data[ijk]).conjugate() * Quaternion(other._data[ijk])
        return acc

    def norm(self): return float(np.sqrt(self.inner(self).r))
    def norm_squared(self): return float(self.inner(self).r)

    def unfold(self, mode):
        if mode == 1: return QuatMatrix(self._data.transpose(0,1,2,3).reshape(self._p, self._q*self._r, 4))
        elif mode == 2: return QuatMatrix(self._data.transpose(1,0,2,3).reshape(self._q, self._p*self._r, 4))
        elif mode == 3: return QuatMatrix(self._data.transpose(2,0,1,3).reshape(self._r, self._p*self._q, 4))
        else: raise ValueError(f"mode must be 1,2,3, got {mode}")

    def to_complex_matrix(self, mode=1): return self.unfold(mode).to_complex_matrix()
    def to_real_matrix(self, mode=1): return self.unfold(mode).to_real_matrix()


def quat(*args): return Quaternion(*args)

# Module-level basis constants
_I = Quaternion(0, 1, 0, 0)
_J = Quaternion(0, 0, 1, 0)
_K = Quaternion(0, 0, 0, 1)
_ZERO = Quaternion(0, 0, 0, 0)
_R = Quaternion(1, 0, 0, 0)
_ONE = Quaternion(1, 1, 1, 1)
