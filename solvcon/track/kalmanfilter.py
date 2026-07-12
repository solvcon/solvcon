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
Quaternion utilities for strapdown inertial navigation, using the
Breckenridge convention (``i^2 = j^2 = k^2 = -1``, ``ijk = +1``) with
the scalar component stored last: ``q = [q_v0, q_v1, q_v2, q_s]``.
"""

import numpy as np


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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
