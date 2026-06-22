# =========================================================================
#  LLM-generated code.  See README.md for full disclosure.
# =========================================================================

"""Tests for quat.serialization — JSON, binary, scipy interop."""
import unittest
import json
import numpy as np
from tests.conftest import QuatTestCase


class TestJSONSerialization(QuatTestCase):
    def test_quaternion_roundtrip(self):
        from quat.serialization import to_json, from_json
        from quat.core import Quaternion
        q = Quaternion(1.5, -2.3, 3.7, -4.1)
        s = to_json(q)
        q2 = from_json(s)
        self.assertEqual(q, q2)

    def test_quatvector_roundtrip(self):
        from quat.serialization import to_json, from_json
        from quat.collections import QuatVector
        v = QuatVector(np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]))
        s = to_json(v)
        v2 = from_json(s)
        self.assertTrue(np.allclose(v.to_array(), v2.to_array()))

    def test_quatmatrix_roundtrip(self):
        from quat.serialization import to_json, from_json
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.random.randn(3, 4, 4))
        s = to_json(M)
        M2 = from_json(s)
        self.assertTrue(np.allclose(M.to_array(), M2.to_array()))

    def test_quattensor_roundtrip(self):
        from quat.serialization import to_json, from_json
        from quat.collections import QuatTensor
        T = QuatTensor(np.random.randn(2, 3, 4, 4))
        s = to_json(T)
        T2 = from_json(s)
        self.assertTrue(np.allclose(T.to_array(), T2.to_array()))

    def test_json_format(self):
        from quat.serialization import to_json
        from quat.core import Quaternion
        q = Quaternion(1, 2, 3, 4)
        s = to_json(q)
        d = json.loads(s)
        self.assertIn("type", d)
        self.assertIn("data", d)


class TestBinarySerialization(QuatTestCase):
    def test_quaternion_roundtrip(self):
        from quat.serialization import to_bytes, from_bytes
        from quat.core import Quaternion
        q = Quaternion(1.5, -2.3, 3.7, -4.1)
        b = to_bytes(q)
        q2 = from_bytes(b)
        self.assertEqual(q, q2)

    def test_matrix_roundtrip(self):
        from quat.serialization import to_bytes, from_bytes
        from quat.collections import QuatMatrix
        M = QuatMatrix(np.random.randn(3, 4, 4))
        b = to_bytes(M)
        M2 = from_bytes(b)
        self.assertTrue(np.allclose(M.to_array(), M2.to_array()))


class TestScipyRotationInterop(QuatTestCase):
    def test_quaternion_to_scipy(self):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("scipy not available")
        from quat.serialization import to_scipy_rotation
        from quat.core import Quaternion
        q = Quaternion.from_axis_angle((0, 0, 1), np.pi / 2)
        rot = to_scipy_rotation(q)
        self.assertIsInstance(rot, Rotation)

    def test_roundtrip_scipy(self):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("scipy not available")
        from quat.serialization import to_scipy_rotation, from_scipy_rotation
        from quat.core import Quaternion
        q = Quaternion.from_axis_angle((1, 0, 0), np.pi / 4)
        rot = to_scipy_rotation(q)
        q2 = from_scipy_rotation(rot)
        self.assertAlmostEqual(q.normalize().r, q2.normalize().r, places=10)
