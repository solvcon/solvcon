# Copyright (c) 2026, Stephen Xie <zonghanxie@proton.me>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Kalman filter for strapdown inertial navigation in the Earth-centered
Earth-fixed (ECEF) frame.

This module is tailored to the Blue Origin Deorbit, Descent, and Landing
Tipping Point (BODDL-TP) NASA dataset.

The propagated state vector is

    x = [p_x, p_y, p_z, v_x, v_y, v_z, q_v0, q_v1, q_v2, q_s]   (shape 10,)

and the covariance is the 10x10 covariance of this same state.
"""

import pathlib

import numpy as np

from .. import SimpleArrayFloat64
from .. import KalmanFilterFp64


__all__ = [
    "EARTH_RATE",
    "WGS84_GM",
    "WGS84_SEMI_MAJOR",
    "WGS84_J2",
    "DLC_IMU_DCM_CON_IMU",
    "DLC_IMU_LEVER_ARM_CON",
    "InertialKalmanFilter",
]


EARTH_RATE = 7.2921150e-5
"""Earth sidereal rotation rate (rad/s)."""

WGS84_GM = 3.986004418e14
"""WGS-84 Earth gravitational constant (m^3/s^2)."""

WGS84_SEMI_MAJOR = 6378137.0
"""WGS-84 equatorial radius (m)."""

WGS84_J2 = 1.0826267e-3
"""WGS-84 second zonal harmonic (dimensionless)."""

DLC_IMU_DCM_CON_IMU = np.array([
    [-0.2477, -0.1673,  0.9543],
    [-0.0478,  0.9859,  0.1604],
    [-0.9677, -0.0059, -0.2522],
], dtype='float64')
"""DCM that rotates a vector in the DLC IMU frame into the CON frame."""

DLC_IMU_LEVER_ARM_CON = np.array(
    [-0.08035, 0.28390, -1.42333], dtype='float64',
)
"""Position of the DLC IMU in the CON frame, relative to CON (m)."""

STATE_DIM = 10
CONTROL_DIM = 10


def _skew(v):
    """
    Build the 3x3 skew-symmetric cross-product matrix of a 3-vector.

    :param v: Three-element vector.
    :type v: numpy.ndarray
    :return: Matrix ``[v]x`` such that ``[v]x @ u == numpy.cross(v, u)``.
    :rtype: numpy.ndarray
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype='float64')


def quat_identity():
    """
    Return the Breckenridge identity quaternion (scalar stored last).

    :return: Array ``[0, 0, 0, 1]``.
    :rtype: numpy.ndarray
    """
    return np.array([0.0, 0.0, 0.0, 1.0], dtype='float64')


def quat_multiply(q1, q2):
    """
    Multiply two Breckenridge quaternions ``q1 (x) q2`` with scalar last.

    Because Breckenridge sets ``ijk = +1``, the cross-product term of the
    vector component carries the opposite sign to the Hamilton form.

    :param q1: Left quaternion ``[v0, v1, v2, s]``.
    :type q1: numpy.ndarray
    :param q2: Right quaternion ``[v0, v1, v2, s]``.
    :type q2: numpy.ndarray
    :return: Composed quaternion ``[v0, v1, v2, s]``.
    :rtype: numpy.ndarray
    """
    v1 = np.asarray(q1[0:3], dtype='float64')
    s1 = float(q1[3])
    v2 = np.asarray(q2[0:3], dtype='float64')
    s2 = float(q2[3])

    s = s1 * s2 - float(np.dot(v1, v2))
    v = s1 * v2 + s2 * v1 - np.cross(v1, v2)
    return np.array([v[0], v[1], v[2], s], dtype='float64')


def quat_to_dcm(q):
    """
    Convert a Breckenridge quaternion into the DCM rotating a vector from
    the body frame to the reference (ECEF) frame.

    :param q: Unit quaternion ``[v0, v1, v2, s]``.
    :type q: numpy.ndarray
    :return: 3x3 body-to-reference rotation matrix.
    :rtype: numpy.ndarray
    """
    v = np.asarray(q[0:3], dtype='float64')
    s = float(q[3])
    vv = float(np.dot(v, v))
    eye3 = np.eye(3, dtype='float64')
    return (s * s - vv) * eye3 + 2.0 * np.outer(v, v) - 2.0 * s * _skew(v)


def rotvec_to_quat(theta):
    """
    Convert a rotation vector (axis times angle) into a Breckenridge
    quaternion with scalar last.

    :param theta: Rotation vector in radians, shape ``(3,)``.
    :type theta: numpy.ndarray
    :return: Unit quaternion ``[v0, v1, v2, s]``.
    :rtype: numpy.ndarray
    """
    theta = np.asarray(theta, dtype='float64')
    angle = float(np.linalg.norm(theta))
    if angle < 1.0e-12:
        half = 0.5 * theta
        scalar = 1.0 - 0.125 * float(np.dot(theta, theta))
        return np.array(
            [half[0], half[1], half[2], scalar], dtype='float64',
        )
    half = 0.5 * angle
    axis = theta / angle
    sin_half = np.sin(half)
    return np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        np.cos(half),
    ], dtype='float64')


class InertialKalmanFilter:
    """
    Strapdown Kalman-filter predictor in ECEF, wrapping ``KalmanFilterFp64``.

    Each call to :meth:`predict` consumes one IMU sample (delta-velocity and
    delta-angle in the body frame), composes the corresponding control input
    ``u``, and delegates the linear propagation to the C++
    :class:`modmesh.KalmanFilterFp64` instance.  ``dt`` is fixed at
    construction time -- the BODDL-TP DLC IMU has a constant sampling
    interval -- so the C++ filter and the ``F``/``B`` matrices are built
    exactly once.

    :ivar dt: IMU sampling interval (s) baked into ``F``.
    :vartype dt: float
    :ivar earth_rate: Earth rotation rate (rad/s).
    :vartype earth_rate: float
    :ivar use_j2: Whether to include the J2 term in the gravity model.
    :vartype use_j2: bool
    :ivar include_centrifugal: Whether to include the centrifugal term.
    :vartype include_centrifugal: bool
    :ivar filter: Underlying C++ Kalman filter.
    :vartype filter: modmesh.KalmanFilterFp64
    """

    def __init__(
        self,
        initial_state,
        dt,
        earth_rate=EARTH_RATE,
        process_noise=1.0e-3,
        measurement_noise=1.0,
        jitter=1.0e-9,
        use_j2=True,
        include_centrifugal=True,
        lever_arm=None,
    ):
        """
        Build the filter from an initial state.

        :param initial_state: Initial 10-element state ``[p, v, q]`` where
            ``q`` is the Breckenridge body-to-ECEF quaternion with scalar
            last.  Must be non-zero in its quaternion slice.
        :type initial_state: numpy.ndarray
        :param dt: IMU sampling interval (s).  Must be strictly positive.
            Baked into ``F`` so the C++ filter only needs to be
            constructed once.
        :type dt: float
        :param earth_rate: Scalar Earth rotation rate (rad/s).
        :type earth_rate: float
        :param process_noise: Process noise standard deviation forwarded to
            ``KalmanFilterFp64`` (``Q = process_noise**2 * I``).
        :type process_noise: float
        :param measurement_noise: Measurement noise standard deviation; the
            predict-only workflow does not use it but the C++ constructor
            requires a value.
        :type measurement_noise: float
        :param jitter: Numerical jitter added to the innovation covariance
            by the C++ filter during updates.
        :type jitter: float
        :param use_j2: Include the J2 zonal term in the gravity model.
        :type use_j2: bool
        :param include_centrifugal: Include centrifugal acceleration.
        :type include_centrifugal: bool
        :param lever_arm: Position of the IMU sensor in the body/CON frame
            relative to the CON reference point (m).  When non-zero, the
            ``predict`` step subtracts the tangential (``alpha x l``) and
            centripetal (``omega x (omega x l)``) lever-arm accelerations
            from the supplied IMU delta-velocity, recovering the
            equivalent specific force at CON.  ``None`` disables the
            correction (default).
        :type lever_arm: numpy.ndarray or None
        """
        state = np.asarray(initial_state, dtype='float64').copy()
        if state.shape != (STATE_DIM,):
            raise ValueError(
                f"initial_state must have shape ({STATE_DIM},), "
                f"got {state.shape}"
            )
        qnorm = float(np.linalg.norm(state[6:10]))
        if qnorm == 0.0:
            raise ValueError("initial quaternion has zero norm")
        state[6:10] /= qnorm

        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive")
        self.dt = dt
        self.earth_rate = float(earth_rate)
        self.use_j2 = bool(use_j2)
        self.include_centrifugal = bool(include_centrifugal)
        if lever_arm is None:
            self.lever_arm = np.zeros(3, dtype='float64')
        else:
            self.lever_arm = np.asarray(
                lever_arm, dtype='float64',
            ).reshape(3).copy()

        self._prev_omega_body = None

        # Bake the dt-dependent linear pieces of the strapdown dynamics
        # into F now that dt is fixed
        f_matrix = np.eye(STATE_DIM, dtype='float64')
        f_matrix[0:3, 3:6] = dt * np.eye(3, dtype='float64')
        omega_ie_skew = np.array([
            [0.0, -self.earth_rate, 0.0],
            [self.earth_rate, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype='float64')
        f_matrix[3:6, 3:6] += -2.0 * omega_ie_skew * dt
        if self.include_centrifugal:
            omega_sq = self.earth_rate * self.earth_rate
            cf = np.array([omega_sq, omega_sq, 0.0], dtype='float64')
            f_matrix[3:6, 0:3] += np.diag(cf) * dt
        self._f_matrix = f_matrix
        self._b_matrix = np.eye(STATE_DIM, CONTROL_DIM, dtype='float64')

        h_sa = SimpleArrayFloat64([1, STATE_DIM])

        self.filter = KalmanFilterFp64(
            x=SimpleArrayFloat64(array=state),
            f=SimpleArrayFloat64(array=self._f_matrix),
            b=SimpleArrayFloat64(array=self._b_matrix),
            h=h_sa,
            process_noise=float(process_noise),
            measurement_noise=float(measurement_noise),
            jitter=float(jitter),
        )

    @property
    def state(self):
        """
        Return a copy of the current 10-element mean state vector.

        :return: Array ``[p (3), v (3), q (4)]``.
        :rtype: numpy.ndarray
        """
        return self.filter.state.ndarray.copy()

    @property
    def covariance(self):
        """
        Return a copy of the current 10x10 state covariance matrix.

        :return: Covariance matrix.
        :rtype: numpy.ndarray
        """
        return self.filter.covariance.ndarray.copy()

    @property
    def position(self):
        """Position vector in ECEF (m)."""
        return self.state[0:3]

    @property
    def velocity(self):
        """Velocity vector in ECEF (m/s)."""
        return self.state[3:6]

    @property
    def quaternion(self):
        """Body-to-ECEF quaternion ``[v0, v1, v2, s]``."""
        return self.state[6:10]

    def gravity(self, position):
        """
        Evaluate ECEF gravitational acceleration at ``position``.

        The model is two-body gravity plus an optional J2 zonal-harmonic
        term.  The centrifugal acceleration is not included here because
        that term is handled by the linear part of ``F``.

        :param position: ECEF position vector (m).
        :type position: numpy.ndarray
        :return: Gravitational acceleration in ECEF (m/s^2).
        :rtype: numpy.ndarray
        """
        position = np.asarray(position, dtype='float64')
        r = float(np.linalg.norm(position))
        if r <= 0.0:
            return np.zeros(3, dtype='float64')

        x, y, z = position
        base = -WGS84_GM / (r ** 3)
        if not self.use_j2:
            return base * position

        zr2 = (z / r) ** 2
        ar2 = (WGS84_SEMI_MAJOR / r) ** 2
        c = 1.5 * WGS84_J2 * ar2
        return np.array([
            base * x * (1.0 - c * (5.0 * zr2 - 1.0)),
            base * y * (1.0 - c * (5.0 * zr2 - 1.0)),
            base * z * (1.0 - c * (5.0 * zr2 - 3.0)),
        ], dtype='float64')

    def predict(self, delta_vel_body, delta_angle_body):
        """
        Run one predict step using IMU delta-velocity and delta-angle.

        :param delta_vel_body: Integrated specific force over ``dt`` in the
            body frame (m/s), shape ``(3,)``.
        :type delta_vel_body: numpy.ndarray
        :param delta_angle_body: Integrated angular rate over ``dt`` in the
            body frame (rad), shape ``(3,)``.
        :type delta_angle_body: numpy.ndarray
        :return: Updated 10-element state vector (copy).
        :rtype: numpy.ndarray
        """
        dt = self.dt
        delta_vel_body = np.asarray(
            delta_vel_body, dtype='float64',
        ).reshape(3)
        delta_angle_body = np.asarray(
            delta_angle_body, dtype='float64',
        ).reshape(3)

        # Lever-arm correction
        omega = delta_angle_body / dt
        if np.any(self.lever_arm):
            if self._prev_omega_body is None:
                delta_omega = np.zeros(3, dtype='float64')
            else:
                delta_omega = omega - self._prev_omega_body
            centripetal = (
                np.cross(omega, np.cross(omega, self.lever_arm)) * dt
            )
            tangential = np.cross(delta_omega, self.lever_arm)
            delta_vel_body = delta_vel_body - centripetal - tangential
        self._prev_omega_body = omega

        x_k = self.state
        p_k = x_k[0:3]
        q_k = x_k[6:10]

        dcm_be = quat_to_dcm(q_k)
        dcm_eb = dcm_be.T
        omega_ie = np.array([0.0, 0.0, self.earth_rate], dtype='float64')

        dv_ecef = dcm_be @ delta_vel_body
        g = self.gravity(p_k)
        dv_grav = dv_ecef + g * dt

        delta_theta_eb_body = delta_angle_body - dcm_eb @ omega_ie * dt

        dq = rotvec_to_quat(-delta_theta_eb_body)
        q_new = quat_multiply(q_k, dq)
        q_new /= np.linalg.norm(q_new)

        # ``v*dt``, Coriolis, and the linear centrifugal coupling are
        # already applied by ``F x_k``; ``u`` only carries the residual
        # state-dependent pieces.
        u = np.zeros(CONTROL_DIM, dtype='float64')
        u[0:3] = 0.5 * dt * dv_grav
        u[3:6] = dv_grav
        u[6:10] = q_new - q_k

        self.filter.predict(SimpleArrayFloat64(array=u))
        return self.state


def _initial_state_from_event(gt_event):
    """
    Build a 10-element state vector from a ground-truth event.

    :param gt_event: Ground-truth event reference.
    :type gt_event: modmesh.track.dataset.EventReference
    :return: Initial state ``[p (3), v (3), q (4)]``.
    :rtype: numpy.ndarray
    """
    data = gt_event.data
    return np.array([
        data["truth_pos_CON_ECEF_ECEF_M[1]"],
        data["truth_pos_CON_ECEF_ECEF_M[2]"],
        data["truth_pos_CON_ECEF_ECEF_M[3]"],
        data["truth_vel_CON_ECEF_ECEF_MpS[1]"],
        data["truth_vel_CON_ECEF_ECEF_MpS[2]"],
        data["truth_vel_CON_ECEF_ECEF_MpS[3]"],
        data["truth_quat_CON2ECEF[1]"],
        data["truth_quat_CON2ECEF[2]"],
        data["truth_quat_CON2ECEF[3]"],
        data["truth_quat_CON2ECEF[4]"],
    ], dtype='float64')


def _imu_increments(imu_event):
    """
    Extract delta-velocity and delta-angle vectors from an IMU event and
    rotate them from the DLC IMU sensor frame into the vehicle CON frame.

    :param imu_event: IMU event reference.
    :type imu_event: modmesh.track.dataset.EventReference
    :return: Tuple ``(delta_vel_con, delta_angle_con)`` expressed in CON.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    data = imu_event.data
    dv_imu = np.array([
        data["DATA_DELTA_VEL[1]"],
        data["DATA_DELTA_VEL[2]"],
        data["DATA_DELTA_VEL[3]"],
    ], dtype='float64')
    da_imu = np.array([
        data["DATA_DELTA_ANGLE[1]"],
        data["DATA_DELTA_ANGLE[2]"],
        data["DATA_DELTA_ANGLE[3]"],
    ], dtype='float64')
    dv = DLC_IMU_DCM_CON_IMU @ dv_imu
    da = DLC_IMU_DCM_CON_IMU @ da_imu
    return dv, da


def _gt_pos_vel(gt_event):
    """
    Extract ground-truth position and velocity vectors from an event.

    :param gt_event: Ground-truth event reference.
    :type gt_event: modmesh.track.dataset.EventReference
    :return: Tuple ``(position, velocity)`` in ECEF.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    data = gt_event.data
    p = np.array([
        data["truth_pos_CON_ECEF_ECEF_M[1]"],
        data["truth_pos_CON_ECEF_ECEF_M[2]"],
        data["truth_pos_CON_ECEF_ECEF_M[3]"],
    ], dtype='float64')
    v = np.array([
        data["truth_vel_CON_ECEF_ECEF_MpS[1]"],
        data["truth_vel_CON_ECEF_ECEF_MpS[2]"],
        data["truth_vel_CON_ECEF_ECEF_MpS[3]"],
    ], dtype='float64')
    return p, v


def _figure_path(prefix):
    """
    Build the PNG output path by appending the fixed suffix
    ``_kf_predict.png`` to the user-supplied path-and-file-name prefix.

    :param prefix: Output directory path plus file-name prefix.
    :type prefix: str
    :return: Path of the PNG file to write.
    :rtype: pathlib.Path
    """
    return pathlib.Path(f"{prefix}_kf_predict.png")


def _predict_flight():
    """
    Load the BODDL-TP dataset, run KF predict, and evaluate errors.

    :return: Tuple ``(pred_t, pred_p, pred_v, gt_t, gt_p, gt_v,
        err_p)``.  Times are seconds since the first ground-truth
        event; positions and velocities are ECEF arrays with one row
        per sample; ``err_p`` holds the per-axis position error on the
        ground-truth timestamps.
    :rtype: tuple[numpy.ndarray, ...]
    """
    import ssl

    from solvcon.track import dataset as dataset_mod

    ssl._create_default_https_context = ssl._create_stdlib_context
    dataset_obj = dataset_mod.NasaDataset(
        "https://techport.nasa.gov/api/file/presignedUrl/380503",
        "DDL-F1_Dataset-20201013.zip",
    )
    dataset_obj.download()
    dataset_obj.extract()
    dataset_obj.load()

    gt_start = next(
        (ev for ev in dataset_obj.events if ev.source == "ground_truth"),
        None,
    )
    if gt_start is None:
        raise RuntimeError("dataset contains no ground-truth events")

    # assuming ``dt`` is fixed across the run
    imu_ts_after_start = [
        ev.timestamp for ev in dataset_obj.events
        if ev.source == "imu" and ev.timestamp >= gt_start.timestamp
    ]
    if len(imu_ts_after_start) < 2:
        raise RuntimeError(
            "need at least two IMU events after gt_start to fix dt"
        )
    dt = (imu_ts_after_start[1] - imu_ts_after_start[0]) * 1.0e-9
    if dt <= 0.0:
        raise RuntimeError("non-positive dt from the first two IMU samples")

    kf = InertialKalmanFilter(
        initial_state=_initial_state_from_event(gt_start),
        dt=dt,
        lever_arm=DLC_IMU_LEVER_ARM_CON,
    )

    t0 = gt_start.timestamp * 1.0e-9
    pred_t = [0.0]
    pred_p = [kf.position.copy()]
    pred_v = [kf.velocity.copy()]
    gt_t = [0.0]
    gt_p0, gt_v0 = _gt_pos_vel(gt_start)
    gt_p = [gt_p0]
    gt_v = [gt_v0]

    for ev in dataset_obj.events:
        if ev.timestamp < gt_start.timestamp:
            continue
        if ev.source == "imu":
            dv, da = _imu_increments(ev)
            kf.predict(dv, da)
            pred_t.append(ev.timestamp * 1.0e-9 - t0)
            pred_p.append(kf.position.copy())
            pred_v.append(kf.velocity.copy())
        elif ev.source == "ground_truth" and ev is not gt_start:
            p, v = _gt_pos_vel(ev)
            gt_t.append(ev.timestamp * 1.0e-9 - t0)
            gt_p.append(p)
            gt_v.append(v)

    pred_t = np.asarray(pred_t, dtype='float64')
    pred_p = np.asarray(pred_p, dtype='float64')
    pred_v = np.asarray(pred_v, dtype='float64')
    gt_t = np.asarray(gt_t, dtype='float64')
    gt_p = np.asarray(gt_p, dtype='float64')
    gt_v = np.asarray(gt_v, dtype='float64')

    # Interpolation so the error is evaluated at the same
    # instants as the ground-truth samples.
    pred_p_on_gt = np.column_stack([
        np.interp(gt_t, pred_t, pred_p[:, i]) for i in range(3)
    ])
    err_p = pred_p_on_gt - gt_p

    return pred_t, pred_p, pred_v, gt_t, gt_p, gt_v, err_p


def main(argv=None):
    """
    Parse command-line arguments, run the KF predict through
    :func:`_predict_flight`, and plot the results.

    :param argv: Command-line arguments; ``None`` uses ``sys.argv``.
    :type argv: list[str] or None
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Kalman-filter predict baseline on the BODDL-TP "
                    "Flight 1 dataset",
    )
    parser.add_argument(
        "--save-fig-prefix",
        default=None,
        help="path plus file-name prefix for saving the results figure; "
             "the fixed suffix '_kf_predict.png' is appended, so "
             "'/tmp/run1' gives /tmp/run1_kf_predict.png; when omitted "
             "the figure is shown in a window instead",
    )
    args = parser.parse_args(argv)

    import matplotlib
    if args.save_fig_prefix is not None:
        # Agg renders without a display, so saving works headlessly.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (pred_t, pred_p, pred_v, gt_t, gt_p, gt_v,
     err_p) = _predict_flight()

    fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex=True)
    axis_names = ("X", "Y", "Z")
    for i, name in enumerate(axis_names):
        axes[i, 0].plot(
            gt_t, gt_p[:, i],
            "k.", markersize=2, label="ground truth",
        )
        axes[i, 0].plot(
            pred_t, pred_p[:, i],
            "b-", linewidth=0.7, label="KF predict",
        )
        axes[i, 0].set_ylabel(f"p_{name} (m)")
        axes[i, 0].grid(True)
        axes[i, 0].legend(loc="best", fontsize=8)

        axes[i, 1].plot(
            gt_t, gt_v[:, i],
            "k.", markersize=2, label="ground truth",
        )
        axes[i, 1].plot(
            pred_t, pred_v[:, i],
            "r-", linewidth=0.7, label="KF predict",
        )
        axes[i, 1].set_ylabel(f"v_{name} (m/s)")
        axes[i, 1].grid(True)
        axes[i, 1].legend(loc="best", fontsize=8)

        axes[i, 2].plot(
            gt_t, err_p[:, i],
            "g-", linewidth=0.7, label=f"err p_{name}",
        )
        axes[i, 2].axhline(0.0, color="k", linewidth=0.5)
        axes[i, 2].set_ylabel(f"err p_{name} (m)")
        axes[i, 2].grid(True)
        axes[i, 2].legend(loc="best", fontsize=8)

    axes[0, 0].set_title("ECEF position")
    axes[0, 1].set_title("ECEF velocity")
    axes[0, 2].set_title("Position error (predict - truth)")
    axes[2, 0].set_xlabel("time since first ground-truth (s)")
    axes[2, 1].set_xlabel("time since first ground-truth (s)")
    axes[2, 2].set_xlabel("time since first ground-truth (s)")
    fig.suptitle("KF predict vs. NASA ground truth")
    fig.tight_layout()

    if args.save_fig_prefix is None:
        plt.show()
    else:
        out = _figure_path(args.save_fig_prefix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"saved figure to {out}")


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
