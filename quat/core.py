"""Quaternion class — single quaternion value type."""
from __future__ import annotations

import numpy as np
from typing import Tuple, List, Iterator
from numbers import Real, Complex
from quat.algebra import _hamilton, _CONJ


class Quaternion:
    __slots__ = ('_data',)

    def __init__(self, *args, **kwargs) -> None:
        """
        Construct a Quaternion.

        Example
        -------
        >>> q = Quaternion(1, 2, 3, 4)       # (r, i, j, k)
        >>> q = Quaternion(5.0)               # real scalar
        >>> q = Quaternion(1+2j)              # from complex
        >>> q = Quaternion(q2)                # copy
        >>> q = Quaternion(np.array([1,2,3,4]))  # from ndarray
        >>> q = Quaternion(scalar=1, vector=(2,3,4))  # keyword args
        """
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
    def zero(cls) -> Quaternion:
        return cls(0., 0., 0., 0.)

    @classmethod
    def one_q(cls) -> Quaternion:
        return cls(1., 1., 1., 1.)

    @classmethod
    def from_axis_angle(cls, axis, angle) -> Quaternion:
        """
        Construct a unit quaternion from an axis-angle rotation.

        Example
        -------
        >>> q = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        >>> q  # 90° rotation about z-axis
        """
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        half = angle / 2.
        s = np.sin(half)
        return cls(np.cos(half), s*axis[0], s*axis[1], s*axis[2])

    # -- accessors -----------------------------------------------------------
    @property
    def r(self) -> float:
        return self._data[0].item()

    @property
    def i(self) -> float:
        return self._data[1].item()

    @property
    def j(self) -> float:
        return self._data[2].item()

    @property
    def k(self) -> float:
        return self._data[3].item()

    @property
    def real(self) -> float:
        return self._data[0].item()

    @property
    def imag(self) -> np.ndarray:
        return self._data[1:4].copy()

    @property
    def components(self) -> Tuple[float, float, float, float]:
        d = self._data
        return (d[0].item(), d[1].item(), d[2].item(), d[3].item())

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx) -> float:
        return self._data[idx].item()

    def __iter__(self) -> Iterator[float]:
        return iter(self.components)

    # -- display -------------------------------------------------------------
    def __repr__(self) -> str:
        a, b, c, d = self.components
        return f"Quaternion({a}, {b}, {c}, {d})"

    def __str__(self) -> str:
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
    def __eq__(self, other) -> bool:
        if isinstance(other, Quaternion):
            return np.allclose(self._data, other._data)
        if isinstance(other, (Real, Complex)):
            return self == Quaternion(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(round(v, 12) for v in self.components))

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, other) -> Quaternion:
        if isinstance(other, Quaternion):
            return Quaternion(self._data + other._data)
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data[0] + r, self._data[1],
                              self._data[2], self._data[3])
        return NotImplemented

    def __radd__(self, other) -> Quaternion:
        return self.__add__(other)

    def __sub__(self, other) -> Quaternion:
        if isinstance(other, Quaternion):
            return Quaternion(self._data - other._data)
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data[0] - r, self._data[1],
                              self._data[2], self._data[3])
        return NotImplemented

    def __rsub__(self, other) -> Quaternion:
        if isinstance(other, (Real, Complex)):
            r = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(r - self._data[0], -self._data[1],
                              -self._data[2], -self._data[3])
        return NotImplemented

    def __neg__(self) -> Quaternion:
        return Quaternion(-self._data)

    def __pos__(self) -> Quaternion:
        return Quaternion(self._data.copy())

    def __mul__(self, other) -> Quaternion:
        if isinstance(other, Quaternion):
            return Quaternion(_hamilton(self._data, other._data))
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data * s)
        return NotImplemented

    def __rmul__(self, other) -> Quaternion:
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data * s)
        if isinstance(other, Quaternion):
            return Quaternion(_hamilton(other._data, self._data))
        return NotImplemented

    def __truediv__(self, other) -> Quaternion:
        if isinstance(other, Quaternion):
            return self * other.inverse()
        if isinstance(other, (Real, Complex)):
            s = float(other.real if isinstance(other, Complex) else other)
            return Quaternion(self._data / s)
        return NotImplemented

    # -- algebraic -----------------------------------------------------------
    def conjugate(self) -> Quaternion:
        """
        Return the quaternion conjugate (r, -i, -j, -k).

        Example
        -------
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.conjugate()
        Quaternion(1.0, -2.0, -3.0, -4.0)
        """
        return Quaternion(self._data * _CONJ)

    def norm(self) -> float:
        """
        Return the Euclidean norm.

        Example
        -------
        >>> q = Quaternion(3, 4, 0, 0)
        >>> q.norm()
        5.0
        """
        return float(np.linalg.norm(self._data))

    def norm_squared(self) -> float:
        return float(np.dot(self._data, self._data))

    def normalize(self) -> Quaternion:
        """
        Return a unit-norm copy of this quaternion.

        Example
        -------
        >>> q = Quaternion(3, 4, 0, 0)
        >>> q.normalize().norm()
        1.0
        """
        n = self.norm()
        if n == 0.0:
            raise ZeroDivisionError("Cannot normalize zero quaternion")
        return Quaternion(self._data / n)

    def inverse(self) -> Quaternion:
        """
        Return the multiplicative inverse q⁻¹ = q* / ‖q‖².

        Example
        -------
        >>> q = Quaternion(1, 0, 0, 0)
        >>> q * q.inverse()
        Quaternion(1.0, 0.0, 0.0, 0.0)
        """
        n2 = self.norm_squared()
        if n2 == 0.0:
            raise ZeroDivisionError("Zero quaternion has no inverse")
        return Quaternion(self._data * _CONJ / n2)

    def exp(self) -> Quaternion:
        """
        Quaternion exponential.

        Example
        -------
        >>> q = Quaternion(0, 0.3, 0.4, 0)
        >>> roundtrip = q.exp().log()
        >>> abs(roundtrip.r - q.r) < 1e-10
        True
        """
        a, b, c, d = self._data
        v_norm = np.sqrt(b*b + c*c + d*d)
        ea = np.exp(a)
        if v_norm < 1e-15:
            return Quaternion(ea, 0., 0., 0.)
        s = np.sin(v_norm) / v_norm
        return Quaternion(ea * np.cos(v_norm), ea * s * b,
                          ea * s * c, ea * s * d)

    def log(self) -> Quaternion:
        """
        Quaternion logarithm.

        Example
        -------
        >>> q = Quaternion(0, 0.3, 0.4, 0)
        >>> q.log()
        """
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

    def pow(self, t) -> Quaternion:
        """
        Raise quaternion to a real power: exp(t * log(q)).

        Example
        -------
        >>> q = Quaternion(2, 0, 0, 0)
        >>> q.pow(3)
        Quaternion(8.0, 0.0, 0.0, 0.0)
        """
        return (t * self.log()).exp()

    def re_inner(self, other: Quaternion) -> float:
        """
        Real inner product (dot product of coefficient vectors).

        Example
        -------
        >>> a = Quaternion(1, 2, 3, 4)
        >>> b = Quaternion(4, 3, 2, 1)
        >>> a.re_inner(b)
        20.0
        """
        return float(np.dot(self._data, other._data))

    def commutator(self, other: Quaternion) -> Quaternion:
        """
        Commutator [p, q] = p*q - q*p.

        Example
        -------
        >>> i = Quaternion(0, 1, 0, 0)
        >>> j = Quaternion(0, 0, 1, 0)
        >>> i.commutator(j)
        Quaternion(0.0, 0.0, 0.0, 2.0)
        """
        return self * other - other * self

    def isnan(self) -> bool:
        return bool(np.any(np.isnan(self._data)))

    def isinf(self) -> bool:
        return bool(np.any(np.isinf(self._data)))

    def isfinite(self) -> bool:
        return bool(np.all(np.isfinite(self._data)))

    def isclose(self, other: Quaternion, rtol=1e-05, atol=1e-08) -> bool:
        return bool(np.allclose(self._data, other._data, rtol=rtol, atol=atol))

    # -- rotation ------------------------------------------------------------
    def rotate_vector(self, v) -> np.ndarray:
        """
        Rotate a 3D vector by this quaternion: imag(q * (0, v) * q*).

        Example
        -------
        >>> q = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        >>> v = (1.0, 0.0, 0.0)
        >>> vr = q.rotate_vector(v)
        >>> vr  # rotated to y-axis
        array([0., 1., 0.])
        """
        qv = Quaternion(0., float(v[0]), float(v[1]), float(v[2]))
        qc = self.conjugate()
        return (self * qv * qc).imag

    # -- matrix representations ----------------------------------------------
    def to_complex_matrix(self) -> np.ndarray:
        """
        2x2 complex matrix representation.

        Example
        -------
        >>> q = Quaternion(1, 2, 3, 4)
        >>> M = q.to_complex_matrix()
        >>> M.shape
        (2, 2)
        """
        a, b, c, d = self._data
        return np.array([[a + 1j*b, c + 1j*d],
                         [-c + 1j*d, a - 1j*b]])

    @classmethod
    def from_complex_matrix(cls, M) -> Quaternion:
        M = np.asarray(M)
        if M.shape != (2, 2):
            raise ValueError(f"Expected (2,2) matrix, got {M.shape}")
        a = (M[0, 0].real + M[1, 1].real) / 2.
        b = (M[0, 0].imag - M[1, 1].imag) / 2.
        c = (M[0, 1].real - M[1, 0].real) / 2.
        d = (M[0, 1].imag + M[1, 0].imag) / 2.
        return cls(a.real, b.real, c.real, d.real)

    def to_real_matrix_left(self) -> np.ndarray:
        """
        4x4 real left-multiplication matrix.

        Example
        -------
        >>> q = Quaternion(1, 2, 3, 4)
        >>> M = q.to_real_matrix_left()
        >>> M.shape
        (4, 4)
        """
        a, b, c, d = self._data
        return np.array([[a, -b, -c, -d],
                         [b,  a, -d,  c],
                         [c,  d,  a, -b],
                         [d, -c,  b,  a]])

    def to_real_matrix_right(self) -> np.ndarray:
        """
        4x4 real right-multiplication matrix.

        Example
        -------
        >>> q = Quaternion(1, 2, 3, 4)
        >>> M = q.to_real_matrix_right()
        >>> M.shape
        (4, 4)
        """
        a, b, c, d = self._data
        return np.array([[a, -b, -c, -d],
                         [b,  a,  d, -c],
                         [c, -d,  a,  b],
                         [d,  c, -b,  a]])

    @classmethod
    def from_real_matrix_left(cls, M) -> Quaternion:
        M = np.asarray(M)
        if M.shape != (4, 4):
            raise ValueError(f"Expected (4,4) matrix, got {M.shape}")
        return cls(M[0, 0], M[1, 0], M[2, 0], M[3, 0])

    def to_array(self) -> np.ndarray:
        return self._data.copy()

    # -- type conversions ----------------------------------------------------
    def __float__(self) -> float:
        if np.allclose(self._data[1:], 0.):
            return float(self._data[0])
        raise ValueError(f"Cannot convert non-real quaternion {self} to float")

    def __int__(self) -> int:
        return int(self.__float__())

    def __complex__(self) -> complex:
        if np.allclose(self._data[2:4], 0.):
            return complex(self._data[0], self._data[1])
        raise ValueError(f"Cannot convert {self} to complex (c,d nonzero)")

    def __bool__(self) -> bool:
        return not np.allclose(self._data, 0.)

    def __abs__(self) -> float:
        return self.norm()


def quat(*args) -> Quaternion:
    """Convenience constructor for Quaternion."""
    return Quaternion(*args)


_I    = Quaternion(0, 1, 0, 0)
_J    = Quaternion(0, 0, 1, 0)
_K    = Quaternion(0, 0, 0, 1)
_ZERO = Quaternion(0, 0, 0, 0)
_R    = Quaternion(1, 0, 0, 0)
_ONE_Q = Quaternion(1, 1, 1, 1)
