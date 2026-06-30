# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Quaternion class — single quaternion value type."""
from __future__ import annotations

import json as _json
import struct as _struct
import numpy as np
from typing import Tuple, List, Iterator
from numbers import Real, Complex
from quat.algebra import _hamilton, _CONJ


def _quat_to_rotmat(q_data: np.ndarray) -> np.ndarray:
    """Convert quaternion data (4,) to a 3x3 rotation matrix."""
    w, x, y, z = q_data
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,         2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,         1 - 2*x*x - 2*y*y],
    ])


def _euler_from_rotmat(R: np.ndarray, seq: str) -> np.ndarray:
    """Extract intrinsic Euler angles from rotation matrix.

    Supports all 6 Tait-Bryan (e.g. 'zyx') and 6 proper Euler (e.g. 'zxz') sequences.
    """
    i = 'xyz'.index(seq[0])
    j = 'xyz'.index(seq[1])
    k = 'xyz'.index(seq[2])
    if i == k:
        return _euler_proper(R, i, j, k)
    return _euler_tait_bryan(R, i, j, k)


def _euler_tait_bryan(R: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    parity = 1 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1
    s2 = parity * R[i, k]
    theta_j = float(np.arcsin(np.clip(s2, -1., 1.)))
    cos_j = float(np.cos(theta_j))
    if abs(cos_j) > 1e-10:
        theta_i = float(np.arctan2(-parity * R[j, k], R[k, k]))
        theta_k = float(np.arctan2(-parity * R[i, j], R[i, i]))
    else:
        theta_i = float(np.arctan2(parity * R[k, j], R[j, j]))
        theta_k = 0.
    return np.array([theta_i, theta_j, theta_k])


def _euler_proper(R: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    even = (i, j, k) in ((0, 1, 0), (1, 2, 1), (2, 0, 2))
    lft = 3 - i - j
    c2 = R[i, i]
    s2 = np.hypot(R[i, j], R[i, lft])
    theta_j = float(np.arctan2(s2, c2))
    if s2 > 1e-10:
        if even:
            theta_i = float(np.arctan2(R[j, i], -R[lft, i]))
            theta_k = float(np.arctan2(R[i, j],  R[i, lft]))
        else:
            theta_i = float(np.arctan2(R[j, i],  R[lft, i]))
            theta_k = float(np.arctan2(R[i, j], -R[i, lft]))
    else:
        theta_i = float(np.arctan2(-R[j, lft], R[j, j]))
        theta_k = 0.
    return np.array([theta_i, theta_j, theta_k])


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

    def to_axis_angle(self) -> Tuple[np.ndarray, float]:
        """Convert unit quaternion to axis-angle representation.

        Returns (axis, angle) where axis is a unit 3-vector and angle is
        in [0, pi] radians.

        Example:
            >>> q = Quaternion.from_axis_angle((0, 0, 1), 1.2)
            >>> axis, angle = q.to_axis_angle()
            >>> abs(np.linalg.norm(axis) - 1) < 1e-10
            True
            >>> abs(angle - 1.2) < 1e-10
            True
        """
        a, b, c, d = self._data
        v_norm_sq = b*b + c*c + d*d
        if v_norm_sq < 1e-30:
            return np.array([1., 0., 0.]), 0.0
        v_norm = np.sqrt(v_norm_sq)
        angle = 2. * np.arctan2(v_norm, a)
        s = 1. / v_norm if v_norm > 0 else 0.
        return np.array([b * s, c * s, d * s]), float(angle)

    @property
    def angle(self) -> float:
        """Rotation angle extracted from unit quaternion (radians, range [0,π])."""
        return self.to_axis_angle()[1]

    @property
    def axis(self) -> np.ndarray:
        """Unit rotation axis extracted from unit quaternion (3-vector)."""
        return self.to_axis_angle()[0]

    # -- Euler angles ---------------------------------------------------------
    @classmethod
    def from_euler(cls, angles, seq: str = 'zyx', intrinsic: bool = True) -> Quaternion:
        """Construct a quaternion from Euler angles.

        Args:
            angles: (roll, pitch, yaw) or (phi, theta, psi) in radians.
            seq: Rotation sequence, e.g. ``'zyx'``, ``'xyz'``, ``'zxz'``.
            intrinsic: If True (default), rotate about moving axes;
                       if False, rotate about fixed axes.

        Returns:
            Unit quaternion representing the composed rotation.

        Example:
            >>> q = Quaternion.from_euler((0, 0, np.pi/2))
            >>> q  # yaw 90° about z
            Quaternion(0.707..., 0.0, 0.0, 0.707...)
        """
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (3,):
            raise ValueError(f"Expected 3 angles, got {angles.shape}")
        if len(seq) != 3:
            raise ValueError(f"seq must be 3 chars, got {seq!r}")
        if not all(c in 'xyz' for c in seq):
            raise ValueError(f"seq must contain only 'x','y','z', got {seq!r}")
        q = Quaternion(1, 0, 0, 0)
        order = seq if intrinsic else seq[::-1]
        for axis, angle in zip(order, angles if intrinsic else angles[::-1]):
            axis_vec = {'x': np.array([1., 0., 0.]),
                        'y': np.array([0., 1., 0.]),
                        'z': np.array([0., 0., 1.])}[axis]
            q = q * cls.from_axis_angle(axis_vec, float(angle))
        return q

    def to_euler(self, seq: str = 'zyx', intrinsic: bool = True) -> np.ndarray:
        """Extract Euler angles from a unit quaternion.

        Uses the rotation sequence *seq* with the specified convention.

        Args:
            seq: Rotation sequence, e.g. ``'zyx'``.
            intrinsic: Same convention as *from_euler*.

        Returns:
            ndarray of 3 angles in radians.

        Example:
            >>> q = Quaternion.from_euler((0.1, 0.2, 0.3))
            >>> angles = q.to_euler()
            >>> np.allclose(angles, [0.1, 0.2, 0.3])
            True
        """
        if intrinsic:
            R = _quat_to_rotmat(self._data)
            return _euler_from_rotmat(R, seq)
        else:
            rev_seq = seq[::-1]
            R = _quat_to_rotmat(self._data)
            return _euler_from_rotmat(R, rev_seq)[::-1]

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
    def w(self) -> float:
        return self._data[0].item()

    @property
    def real(self) -> float:
        return self._data[0].item()

    @property
    def imag(self) -> np.ndarray:
        return self._data[1:4]

    @property
    def components(self) -> Tuple[float, float, float, float]:
        d = self._data
        return (d[0].item(), d[1].item(), d[2].item(), d[3].item())

    @property
    def shape(self) -> Tuple[()]:
        return ()

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
            return bool(np.array_equal(self._data, other._data))
        if isinstance(other, (Real, Complex)):
            rhs = Quaternion(other)
            return bool(np.array_equal(self._data, rhs._data))
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(float(v) for v in self.components))

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

    @property
    def normalized(self) -> Quaternion:
        """Unit-norm copy (property alias for ``normalize()``)."""
        return self.normalize()

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
        a, b, c, d = self._data
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        u_dot_v = b*x + c*y + d*z
        u2 = b*b + c*c + d*d
        scale = a*a - u2
        cx = c*z - d*y
        cy = d*x - b*z
        cz = b*y - c*x
        return np.array([
            2*u_dot_v*b + scale*x + 2*a*cx,
            2*u_dot_v*c + scale*y + 2*a*cy,
            2*u_dot_v*d + scale*z + 2*a*cz,
        ])

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

    @property
    def data(self) -> np.ndarray:
        """Public accessor for the underlying (4,) ndarray (returns a copy)."""
        return self._data.copy()

    def to_array(self) -> np.ndarray:
        return self._data.copy()

    def to_numpy(self, copy: bool = True, dtype=None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)

    # -- serialization --------------------------------------------------------
    def to_json(self) -> str:
        """Serialize to JSON string.

        Example:
            >>> Quaternion(1,2,3,4).to_json()
            '{"type": "Quaternion", "data": [1.0, 2.0, 3.0, 4.0]}'
        """
        return _json.dumps({"type": "Quaternion", "data": self._data.tolist()})

    def to_bytes(self) -> bytes:
        """Serialize to compact binary format."""
        data = self._data.astype(np.float64)
        return _struct.pack('<i', 0) + data.tobytes()

    @classmethod
    def from_json(cls, s: str) -> "Quaternion":
        """Deserialize from JSON string."""
        d = _json.loads(s)
        return cls(np.array(d["data"], dtype=float))

    @classmethod
    def from_bytes(cls, b: bytes) -> "Quaternion":
        """Deserialize from binary bytes."""
        type_id = _struct.unpack_from('<i', b, 0)[0]
        data = np.frombuffer(b[4:], dtype=np.float64)
        return cls(data)

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

    def minimal(self) -> Quaternion:
        """Return canonical representation with scalar part w >= 0.

        For unit quaternions, *q* and *-q* represent the same rotation.
        This method picks the representative with positive scalar part,
        resolving the sign ambiguity.

        Example:
            >>> q = Quaternion(-1, 0, 0, 0)
            >>> q.minimal().w
            1.0
        """
        w = self._data[0]
        if w > 0:
            return Quaternion(self._data)
        if w < 0:
            return Quaternion(-self._data)
        # w == 0: pick first non-zero imaginary component positive
        for i in range(1, 4):
            if abs(self._data[i]) > 1e-15:
                if self._data[i] < 0:
                    return Quaternion(-self._data)
                break
        return Quaternion(self._data)

    def __abs__(self) -> float:
        return self.norm()

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False and dtype is None:
            return self._data
        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """NumPy ufunc dispatch — preserve Quaternion type for supported ops."""
        if method != '__call__':
            return NotImplemented
        if kwargs.get('out') is not None:
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
        if ufunc is np.positive:
            return Quaternion(a._data.copy())
        if ufunc is np.absolute:
            return float(abs(a))
        if ufunc is np.conjugate:
            return a.conjugate()
        return NotImplemented


def quat(*args) -> Quaternion:
    """Convenience constructor for Quaternion."""
    return Quaternion(*args)


_I    = Quaternion(0, 1, 0, 0)
_J    = Quaternion(0, 0, 1, 0)
_K    = Quaternion(0, 0, 0, 1)
_ZERO = Quaternion(0, 0, 0, 0)
_R    = Quaternion(1, 0, 0, 0)
_ONE_Q = Quaternion(1, 1, 1, 1)
