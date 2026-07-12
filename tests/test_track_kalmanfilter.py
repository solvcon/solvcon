# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import pathlib
import unittest

import numpy as np

from solvcon.track import kalmanfilter
from solvcon.track.kalmanfilter import (
    EARTH_RATE,
    WGS84_GM,
    WGS84_J2,
    WGS84_SEMI_MAJOR,
    DLC_IMU_DCM_CON_IMU,
    InertialKalmanFilter,
    quat_identity,
    quat_multiply,
    quat_to_dcm,
    rotvec_to_quat,
)


DT = 0.02
ZERO3 = np.zeros(3, dtype='float64')
EYE3 = np.eye(3, dtype='float64')


def _rest_state():
    state = np.zeros(10, dtype='float64')
    state[0] = WGS84_SEMI_MAJOR
    state[9] = 1.0
    return state


def _two_body_gravity(position):
    position = np.asarray(position, dtype='float64')
    r = float(np.linalg.norm(position))
    return -WGS84_GM * position / r ** 3


class _StubEvent:

    def __init__(self, data):
        self.data = data


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


class DlcImuConstantTC(unittest.TestCase):

    def test_dcm_con_imu_orthonormal(self):
        r = DLC_IMU_DCM_CON_IMU
        np.testing.assert_allclose(r @ r.T, EYE3, atol=5e-4)

    def test_dcm_con_imu_proper_rotation(self):
        det = float(np.linalg.det(DLC_IMU_DCM_CON_IMU))
        self.assertAlmostEqual(det, 1.0, delta=1e-3)


class GravityModelTC(unittest.TestCase):

    def test_two_body_equator_value(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, use_j2=False)
        g = kf.gravity([WGS84_SEMI_MAJOR, 0.0, 0.0])
        expected = -WGS84_GM / WGS84_SEMI_MAJOR ** 2
        np.testing.assert_allclose(g, [expected, 0.0, 0.0], rtol=1e-14)

    def test_two_body_general_position(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, use_j2=False)
        p = np.array([3.0e6, -4.0e6, 2.5e6], dtype='float64')
        np.testing.assert_allclose(
            kf.gravity(p), _two_body_gravity(p), rtol=1e-14,
        )

    def test_j2_equator_scale(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, use_j2=True)
        g = kf.gravity([WGS84_SEMI_MAJOR, 0.0, 0.0])
        expected_x = (
            -WGS84_GM / WGS84_SEMI_MAJOR ** 2 * (1.0 + 1.5 * WGS84_J2)
        )
        np.testing.assert_allclose(
            g, [expected_x, 0.0, 0.0], rtol=1e-12, atol=1e-15,
        )

    def test_j2_pole_scale(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, use_j2=True)
        g = kf.gravity([0.0, 0.0, WGS84_SEMI_MAJOR])
        expected_z = (
            -WGS84_GM / WGS84_SEMI_MAJOR ** 2 * (1.0 - 3.0 * WGS84_J2)
        )
        np.testing.assert_allclose(
            g, [0.0, 0.0, expected_z], rtol=1e-12, atol=1e-15,
        )

    def test_zero_position_returns_zero(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT)
        np.testing.assert_array_equal(kf.gravity(ZERO3), ZERO3)


class FilterConstructionTC(unittest.TestCase):

    def test_rejects_wrong_state_shape(self):
        with self.assertRaisesRegex(
            ValueError, "initial_state must have shape",
        ):
            InertialKalmanFilter(np.zeros(9, dtype='float64'), dt=DT)

    def test_rejects_zero_quaternion(self):
        with self.assertRaisesRegex(ValueError, "zero norm"):
            InertialKalmanFilter(np.zeros(10, dtype='float64'), dt=DT)

    def test_rejects_zero_dt(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            InertialKalmanFilter(_rest_state(), dt=0.0)

    def test_rejects_negative_dt(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            InertialKalmanFilter(_rest_state(), dt=-DT)

    def test_normalizes_initial_quaternion(self):
        state = _rest_state()
        state[6:10] = [0.0, 3.0, 0.0, 4.0]
        kf = InertialKalmanFilter(state, dt=DT)
        np.testing.assert_allclose(
            kf.quaternion, [0.0, 0.6, 0.0, 0.8], rtol=1e-15,
        )

    def test_does_not_mutate_input_state(self):
        state = _rest_state()
        state[6:10] = [0.0, 3.0, 0.0, 4.0]
        before = state.copy()
        InertialKalmanFilter(state, dt=DT)
        np.testing.assert_array_equal(state, before)

    def test_state_roundtrip(self):
        state = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
            dtype='float64',
        )
        kf = InertialKalmanFilter(state, dt=DT)
        np.testing.assert_array_equal(kf.position, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(kf.velocity, [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(kf.quaternion, [0.0, 0.0, 0.0, 1.0])

    def test_lever_arm_defaults_to_zero(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT)
        np.testing.assert_array_equal(kf.lever_arm, ZERO3)

    def test_lever_arm_stored_as_copy(self):
        arm = np.array([0.1, 0.2, 0.3], dtype='float64')
        kf = InertialKalmanFilter(_rest_state(), dt=DT, lever_arm=arm)
        arm[0] = 99.0
        np.testing.assert_array_equal(kf.lever_arm, [0.1, 0.2, 0.3])

    def test_covariance_shape(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT)
        self.assertEqual(kf.covariance.shape, (10, 10))


class FilterPredictTC(unittest.TestCase):

    def test_stationary_body_stays_put(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, earth_rate=0.0)
        p0 = kf.position
        dv = -kf.gravity(p0) * DT
        for _ in range(50):
            kf.predict(dv, ZERO3)
        np.testing.assert_allclose(kf.position, p0, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(kf.velocity, ZERO3, atol=1e-9)
        np.testing.assert_allclose(
            kf.quaternion, quat_identity(), atol=1e-15,
        )

    def test_free_fall_single_step(self):
        kf = InertialKalmanFilter(
            _rest_state(), dt=DT, earth_rate=0.0, use_j2=False,
        )
        p0 = kf.position
        g0 = _two_body_gravity(p0)
        kf.predict(ZERO3, ZERO3)
        np.testing.assert_allclose(kf.velocity, g0 * DT, rtol=1e-12)
        np.testing.assert_allclose(
            kf.position - p0, 0.5 * g0 * DT ** 2, rtol=1e-5, atol=1e-12,
        )

    def test_free_fall_two_steps(self):
        kf = InertialKalmanFilter(
            _rest_state(), dt=DT, earth_rate=0.0, use_j2=False,
        )
        p0 = kf.position
        g0 = _two_body_gravity(p0)
        v1 = g0 * DT
        p1 = p0 + 0.5 * g0 * DT ** 2
        g1 = _two_body_gravity(p1)
        v2 = v1 + g1 * DT
        p2 = p1 + v1 * DT + 0.5 * g1 * DT ** 2
        kf.predict(ZERO3, ZERO3)
        kf.predict(ZERO3, ZERO3)
        np.testing.assert_allclose(kf.velocity, v2, rtol=1e-10)
        np.testing.assert_allclose(
            kf.position - p0, p2 - p0, rtol=1e-5, atol=1e-12,
        )

    def test_commanded_acceleration_single_step(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, earth_rate=0.0)
        p0 = kf.position
        a_cmd = np.array([0.0, 2.5, 0.0], dtype='float64')
        dv = (a_cmd - kf.gravity(p0)) * DT
        kf.predict(dv, ZERO3)
        np.testing.assert_allclose(
            kf.velocity, a_cmd * DT, rtol=1e-9, atol=1e-15,
        )
        np.testing.assert_allclose(
            kf.position - p0, 0.5 * a_cmd * DT ** 2, rtol=1e-9, atol=1e-9,
        )

    def test_pure_spin_integrates_attitude(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, earth_rate=0.0)
        dtheta = np.array([0.0, 0.0, 0.008], dtype='float64')
        nstep = 25
        for _ in range(nstep):
            kf.predict(ZERO3, dtheta)
        total = dtheta * nstep
        angle = float(np.linalg.norm(total))
        axis = total / angle
        expected_dcm = (
            np.cos(angle) * EYE3
            + (1.0 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * kalmanfilter._skew(axis)
        )
        np.testing.assert_allclose(
            quat_to_dcm(kf.quaternion), expected_dcm, atol=1e-12,
        )

    def test_quaternion_norm_preserved(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, earth_rate=0.0)
        dtheta = np.array([0.01, -0.02, 0.015], dtype='float64')
        for _ in range(200):
            kf.predict(ZERO3, dtheta)
        norm = float(np.linalg.norm(kf.quaternion))
        self.assertAlmostEqual(norm, 1.0, places=12)

    def test_earth_rest_equilibrium(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT)
        p0 = kf.position
        g0 = kf.gravity(p0)
        centrifugal = np.array([
            EARTH_RATE ** 2 * p0[0], EARTH_RATE ** 2 * p0[1], 0.0,
        ], dtype='float64')
        dv = (-g0 - centrifugal) * DT
        dtheta = np.array([0.0, 0.0, EARTH_RATE * DT], dtype='float64')
        for _ in range(20):
            kf.predict(dv, dtheta)
        np.testing.assert_allclose(kf.velocity, ZERO3, atol=1e-7)
        np.testing.assert_allclose(
            kf.quaternion, quat_identity(), atol=1e-12,
        )
        np.testing.assert_allclose(kf.position, p0, rtol=0.0, atol=1e-2)

    def test_coriolis_deflects_moving_body(self):
        state = _rest_state()
        vel = 100.0
        state[3:6] = [0.0, vel, 0.0]
        kf = InertialKalmanFilter(state, dt=DT, include_centrifugal=False)
        dv = -kf.gravity(kf.position) * DT
        kf.predict(dv, ZERO3)
        expected_vx = 2.0 * EARTH_RATE * vel * DT
        np.testing.assert_allclose(
            kf.velocity, [expected_vx, vel, 0.0], rtol=1e-12, atol=1e-15,
        )

    def test_centrifugal_toggle_changes_velocity(self):
        kf_on = InertialKalmanFilter(_rest_state(), dt=DT)
        kf_off = InertialKalmanFilter(
            _rest_state(), dt=DT, include_centrifugal=False,
        )
        dv = -kf_on.gravity(kf_on.position) * DT
        kf_on.predict(dv, ZERO3)
        kf_off.predict(dv, ZERO3)
        diff = kf_on.velocity - kf_off.velocity
        expected = [EARTH_RATE ** 2 * WGS84_SEMI_MAJOR * DT, 0.0, 0.0]
        np.testing.assert_allclose(diff, expected, rtol=1e-12, atol=1e-15)

    def test_lever_arm_zero_equals_disabled(self):
        kf_zero = InertialKalmanFilter(
            _rest_state(), dt=DT, earth_rate=0.0, lever_arm=ZERO3,
        )
        kf_none = InertialKalmanFilter(
            _rest_state(), dt=DT, earth_rate=0.0, lever_arm=None,
        )
        dv = np.array([0.1, -0.2, 0.3], dtype='float64')
        dtheta = np.array([0.01, 0.02, -0.03], dtype='float64')
        for _ in range(3):
            kf_zero.predict(dv, dtheta)
            kf_none.predict(dv, dtheta)
        np.testing.assert_array_equal(kf_zero.state, kf_none.state)

    def test_lever_arm_correction(self):
        arm = np.array([0.1, -0.2, 0.5], dtype='float64')
        kf_arm = InertialKalmanFilter(
            _rest_state(), dt=DT, earth_rate=0.0, lever_arm=arm,
        )
        kf_ref = InertialKalmanFilter(
            _rest_state(), dt=DT, earth_rate=0.0, lever_arm=None,
        )

        kf_arm.predict(ZERO3, ZERO3)
        kf_ref.predict(ZERO3, ZERO3)
        np.testing.assert_array_equal(kf_arm.state, kf_ref.state)

        dtheta = np.array([0.02, 0.04, -0.06], dtype='float64')
        kf_arm.predict(ZERO3, dtheta)
        kf_ref.predict(ZERO3, dtheta)
        omega = dtheta / DT
        correction = (
            np.cross(omega, np.cross(omega, arm)) * DT
            + np.cross(omega, arm)
        )
        np.testing.assert_allclose(
            kf_arm.velocity - kf_ref.velocity, -correction,
            rtol=1e-12, atol=1e-15,
        )
        np.testing.assert_array_equal(kf_arm.quaternion, kf_ref.quaternion)

    def test_predict_returns_current_state(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, earth_rate=0.0)
        ret = kf.predict(ZERO3, ZERO3)
        np.testing.assert_array_equal(ret, kf.state)

    def test_covariance_trace_increases(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, earth_rate=0.0)
        trace0 = float(np.trace(kf.covariance))
        kf.predict(ZERO3, ZERO3)
        self.assertGreater(float(np.trace(kf.covariance)), trace0)

    def test_accepts_python_lists(self):
        kf = InertialKalmanFilter(_rest_state(), dt=DT, earth_rate=0.0)
        ret = kf.predict([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(ret.shape, (10,))


class FigurePathTC(unittest.TestCase):

    def test_file_name_prefix(self):
        self.assertEqual(
            kalmanfilter._figure_path("/tmp/run1"),
            pathlib.Path("/tmp/run1_kf_predict.png"),
        )

    def test_directory_prefix(self):
        self.assertEqual(
            kalmanfilter._figure_path("/tmp/out/"),
            pathlib.Path("/tmp/out/_kf_predict.png"),
        )

    def test_relative_prefix(self):
        self.assertEqual(
            kalmanfilter._figure_path("results"),
            pathlib.Path("results_kf_predict.png"),
        )


class EventHelperTC(unittest.TestCase):

    def test_initial_state_from_event(self):
        data = {}
        for i in (1, 2, 3):
            data[f"truth_pos_CON_ECEF_ECEF_M[{i}]"] = float(i)
            data[f"truth_vel_CON_ECEF_ECEF_MpS[{i}]"] = float(3 + i)
        for i in (1, 2, 3, 4):
            data[f"truth_quat_CON2ECEF[{i}]"] = float(6 + i)
        state = kalmanfilter._initial_state_from_event(_StubEvent(data))
        expected = np.arange(1.0, 11.0, dtype='float64')
        np.testing.assert_array_equal(state, expected)

    def test_imu_increments_rotate_into_con(self):
        dv_imu = np.array([1.0, 2.0, 3.0], dtype='float64')
        da_imu = np.array([0.1, 0.2, 0.3], dtype='float64')
        data = {}
        for i in (1, 2, 3):
            data[f"DATA_DELTA_VEL[{i}]"] = dv_imu[i - 1]
            data[f"DATA_DELTA_ANGLE[{i}]"] = da_imu[i - 1]
        dv, da = kalmanfilter._imu_increments(_StubEvent(data))
        np.testing.assert_array_equal(dv, DLC_IMU_DCM_CON_IMU @ dv_imu)
        np.testing.assert_array_equal(da, DLC_IMU_DCM_CON_IMU @ da_imu)

    def test_gt_pos_vel(self):
        data = {}
        for i in (1, 2, 3):
            data[f"truth_pos_CON_ECEF_ECEF_M[{i}]"] = float(i)
            data[f"truth_vel_CON_ECEF_ECEF_MpS[{i}]"] = float(3 + i)
        p, v = kalmanfilter._gt_pos_vel(_StubEvent(data))
        np.testing.assert_array_equal(p, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(v, [4.0, 5.0, 6.0])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
