# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import numpy as np

from solvcon.track import kalmanfilter
from solvcon.track.kalmanfilter import (
    quat_identity,
    quat_multiply,
    quat_to_dcm,
    rotvec_to_quat,
)


ZERO3 = np.zeros(3, dtype='float64')
EYE3 = np.eye(3, dtype='float64')


class QuaternionMathTC(unittest.TestCase):

    def test_skew_matches_cross_product(self):
        v = np.array([0.3, -1.2, 2.5], dtype='float64')
        u = np.array([-0.7, 0.4, 1.1], dtype='float64')
        np.testing.assert_allclose(
            kalmanfilter._skew(v) @ u, np.cross(v, u), atol=1e-15,
        )

    def test_skew_antisymmetric(self):
        m = kalmanfilter._skew(np.array([1.0, 2.0, 3.0], dtype='float64'))
        np.testing.assert_array_equal(m.T, -m)

    def test_identity_value(self):
        np.testing.assert_array_equal(
            quat_identity(), [0.0, 0.0, 0.0, 1.0],
        )

    def test_multiply_identity_is_neutral(self):
        q = rotvec_to_quat([0.3, -0.5, 0.8])
        np.testing.assert_allclose(
            quat_multiply(q, quat_identity()), q, atol=1e-15,
        )
        np.testing.assert_allclose(
            quat_multiply(quat_identity(), q), q, atol=1e-15,
        )

    def test_multiply_breckenridge_basis_products(self):
        i = [1.0, 0.0, 0.0, 0.0]
        j = [0.0, 1.0, 0.0, 0.0]
        np.testing.assert_allclose(
            quat_multiply(i, j), [0.0, 0.0, -1.0, 0.0], atol=1e-15,
        )
        np.testing.assert_allclose(
            quat_multiply(j, i), [0.0, 0.0, 1.0, 0.0], atol=1e-15,
        )

    def test_multiply_preserves_unit_norm(self):
        q1 = rotvec_to_quat([0.2, 0.1, -0.4])
        q2 = rotvec_to_quat([-0.3, 0.7, 0.5])
        norm = float(np.linalg.norm(quat_multiply(q1, q2)))
        self.assertAlmostEqual(norm, 1.0, places=12)

    def test_multiply_dcm_homomorphism(self):
        q1 = rotvec_to_quat([0.2, 0.1, -0.4])
        q2 = rotvec_to_quat([-0.3, 0.7, 0.5])
        np.testing.assert_allclose(
            quat_to_dcm(quat_multiply(q1, q2)),
            quat_to_dcm(q1) @ quat_to_dcm(q2),
            atol=1e-12,
        )

    def test_dcm_of_identity(self):
        np.testing.assert_array_equal(
            quat_to_dcm(quat_identity()), EYE3,
        )

    def test_dcm_orthonormal_proper(self):
        r = quat_to_dcm(rotvec_to_quat([0.4, -0.9, 1.3]))
        np.testing.assert_allclose(r @ r.T, EYE3, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=12)

    def test_dcm_quarter_turn_convention(self):
        q = rotvec_to_quat([0.0, 0.0, np.pi / 2.0])
        np.testing.assert_allclose(
            quat_to_dcm(q) @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-15,
        )

    def test_rotvec_zero_gives_identity(self):
        np.testing.assert_array_equal(
            rotvec_to_quat(ZERO3), quat_identity(),
        )

    def test_rotvec_small_angle_series(self):
        theta = np.array([1.0e-13, 0.0, 0.0], dtype='float64')
        q = rotvec_to_quat(theta)
        np.testing.assert_allclose(
            q, [5.0e-14, 0.0, 0.0, 1.0], rtol=1e-6, atol=0.0,
        )

    def test_rotvec_half_turn_about_z(self):
        q = rotvec_to_quat([0.0, 0.0, np.pi])
        np.testing.assert_allclose(
            q, [0.0, 0.0, 1.0, 0.0], atol=1e-15,
        )

    def test_rotvec_dcm_matches_rodrigues(self):
        for theta in ([0.5, 0.0, 0.0], [0.1, -0.7, 0.4], [-1.2, 0.3, 2.1]):
            theta = np.asarray(theta, dtype='float64')
            angle = float(np.linalg.norm(theta))
            axis = theta / angle
            expected = (
                np.cos(angle) * EYE3
                + (1.0 - np.cos(angle)) * np.outer(axis, axis)
                - np.sin(angle) * kalmanfilter._skew(axis)
            )
            np.testing.assert_allclose(
                quat_to_dcm(rotvec_to_quat(theta)), expected, atol=1e-12,
            )


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
