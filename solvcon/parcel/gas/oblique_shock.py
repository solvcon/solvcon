# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import numpy as np


__all__ = [
    'ObliqueShockRelation',
]


class ObliqueShockRelation(object):
    r"""
    Calculators of oblique shock relations.

    The constructor must take the ratio of specific heat:

    >>> ObliqueShockRelation()
    Traceback (most recent call last):
        ...
    TypeError: __init__() takes exactly 2 arguments (1 given)
    >>> ob = ObliqueShockRelation(gamma=1.4)

    The ratio of specific heat can be accessed through the :py:attr:`gamma`
    attribute:

    >>> ob.gamma
    1.4

    The object can be used to calculate shock relations.  For example,
    :py:meth:`calc_density_ratio` returns the :math:`\rho_2/\rho_1`:

    >>> round(ob.calc_density_ratio(mach1=3, beta=37.8/180*np.pi), 10)
    2.4204302545

    The solution changes as :py:attr:`gamma` changes:

    >>> ob.gamma = 1.2
    >>> round(ob.calc_density_ratio(mach1=3, beta=37.8/180*np.pi), 10)
    2.7793244902
    """

    def __init__(self, gamma):
        """
        :param gamma: Ratio of specific heat :math:`\gamma`, dimensionless.
        """
        #: Ratio of specific heat :math:`\gamma`, dimensionless.
        self.gamma = gamma
        super(ObliqueShockRelation, self).__init__()

    def calc_density_ratio(self, mach1, beta):
        r"""
        Calculate the ratio of density :math:`\rho` across an oblique shock
        wave of which the angle deflected from the upstream flow is
        :math:`\beta` and the upstream Mach number is :math:`M_1`:

        .. math::

          \frac{\rho_2}{\rho_1} =
            \frac{(\gamma + 1) M_{n1}^2}
                 {(\gamma - 1) M_{n1}^2 + 2}

        where :math:`M_{n1} = M_1\sin\beta`.

        This method accepts scalar:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> round(ob.calc_density_ratio(mach1=3, beta=37.8/180*np.pi), 10)
        2.4204302545

        as well as :py:class:`numpy.ndarray`:

        >>> angle = 37.8/180*np.pi; angle = np.array([angle, angle])
        >>> np.round(ob.calc_density_ratio(mach1=3, beta=angle), 10).tolist()
        [2.4204302545, 2.4204302545]

        :param mach1: Upstream Mach number :math:`M_1`, dimensionless.
        :param beta: Oblique shock angle :math:`\beta` deflected from the
                     upstream flow, in radian.
        """
        # Pull data from self.
        gamma = self.gamma
        # Calculate in the way to minimize buffers if arrays are input.
        mach_n1_sq = mach1 * np.sin(beta)
        mach_n1_sq **= 2
        result = (gamma + 1) * mach_n1_sq
        result /= (gamma - 1) * mach_n1_sq + 2
        return result

    def calc_pressure_ratio(self, mach1, beta):
        r"""
        Calculate the ratio of pressure :math:`p` across an oblique shock wave
        of which the angle deflected from the upstream flow is :math:`\beta`
        and the upstream Mach number is :math:`M_1`:

        .. math::

          \frac{p_2}{p_1} = 1 + \frac{2\gamma}{\gamma+1}(M_{n1}^2 - 1)

        where :math:`M_{n1} = M_1\sin\beta`.

        This method accepts scalar:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> round(ob.calc_pressure_ratio(mach1=3, beta=37.8/180*np.pi), 10)
        3.7777114257

        as well as :py:class:`numpy.ndarray`:

        >>> angle = 37.8/180*np.pi; angle = np.array([angle, angle])
        >>> np.round(ob.calc_pressure_ratio(mach1=3, beta=angle), 10).tolist()
        [3.7777114257, 3.7777114257]

        :param mach1: Upstream Mach number :math:`M_1`, dimensionless.
        :param beta: Oblique shock angle :math:`\beta` deflected from the
                     upstream flow, in radian.
        """
        # Pull data from self.
        gamma = self.gamma
        # Calculate in the way to minimize buffers if arrays are input.
        result = mach1 * np.sin(beta)
        result **= 2
        result -= 1
        result *= 2 * gamma / (gamma + 1)
        result += 1
        return result

    def calc_temperature_ratio(self, mach1, beta):
        r"""
        Calculate the ratio of temperature :math:`T` across an oblique shock wave
        of which the angle deflected from the upstream flow is :math:`\beta`
        and the upstream Mach number is :math:`M_1`:

        .. math::

          \frac{T_2}{T_1} = \frac{p_2}{p_1} \frac{\rho_1}{\rho_2}

        where both :math:`p_2/p_1` and :math:`\rho_1/\rho_2` are functions of
        :math:`\gamma`, :math:`M_1`, and :math:`\beta`.  See also
        :py:meth:`calc_pressure_ratio` and :py:meth:`calc_density_ratio`.

        This method accepts scalar:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> round(ob.calc_temperature_ratio(mach1=3, beta=37.8/180*np.pi), 10)
        1.5607602899

        as well as :py:class:`numpy.ndarray`:

        >>> angle = 37.8/180*np.pi; angle = np.array([angle, angle])
        >>> np.round(ob.calc_temperature_ratio(mach1=3, beta=angle), 10).tolist()
        [1.5607602899, 1.5607602899]

        :param mach1: Upstream Mach number :math:`M_1`, dimensionless.
        :param beta: Oblique shock angle :math:`\beta` deflected from the
                     upstream flow, in radian.
        """
        pratio = self.calc_pressure_ratio(mach1, beta)
        rratio = self.calc_density_ratio(mach1, beta)
        return pratio/rratio

    def calc_dmach(self, mach1, beta=None, theta=None, delta=1):
        r"""
        Calculate the downstream Mach number from the upstream Mach number
        :math:`M_1` and either of the shock or flow deflection angles:

        .. math::

          M_2 = \frac{M_{n2}}{\sin(\beta-\theta)}

        where :math:`M_{n2}` is calculated from :py:meth:`calc_normal_dmach`
        with :math:`M_{n1} = M_1\sin\beta`.

        The method can be invoked with only either :math:`\beta` or
        :math:`\theta`:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> ob.calc_dmach(3, beta=0.2, theta=0.1)
        Traceback (most recent call last):
            ...
        ValueError: got (beta=0.2, theta=0.1), but I need to take either beta or theta
        >>> ob.calc_dmach(3)
        Traceback (most recent call last):
            ...
        ValueError: got (beta=None, theta=None), but I need to take either beta or theta

        This method accepts scalar:

        >>> round(ob.calc_dmach(3, beta=37.8/180*np.pi), 10)
        1.9924827009
        >>> round(ob.calc_dmach(3, theta=20./180*np.pi), 10)
        1.9941316656

        as well as :py:class:`numpy.ndarray`:

        >>> angle = 37.8/180*np.pi; angle = np.array([angle, angle])
        >>> np.round(ob.calc_dmach(3, beta=angle), 10).tolist()
        [1.9924827009, 1.9924827009]
        >>> angle = 20./180*np.pi; angle = np.array([angle, angle])
        >>> np.round(ob.calc_dmach(3, theta=angle), 10).tolist()
        [1.9941316656, 1.9941316656]

        :param mach1: Upstream Mach number :math:`M_1`, dimensionless.
        :keyword beta: Oblique shock angle :math:`\beta` deflected from the
                       upstream flow, in radian.
        :keyword theta: Downstream flow angle :math:`\theta` deflected from the
                        upstream flow, in radian.
        :keyword delta: A switching integer :math:`\delta`.  For :math:`\delta =
                        0`, the function gives the solution of strong shock,
                        while for :math:`\delta = 1`, it gives the solution of
                        weak shock.  This keyword argument is only valid when
                        *theta* is given.  The default value is 1.
        """
        if (beta is None) == (theta is None):
            errmsg = "got (beta=%s, theta=%s), " % (str(beta), str(theta))
            errmsg += "but I need to take either beta or theta"
            raise ValueError(errmsg)
        # Calculate the other angle.
        if beta is None:
            beta = self.calc_shock_angle(mach1, theta, delta=delta)
        if theta is None:
            theta = self.calc_flow_angle(mach1, beta)
        # Calculate and return the downstream Mach number.
        mach_n1 = mach1 * np.sin(beta)
        return self.calc_normal_dmach(mach_n1) / np.sin(beta - theta)

    def calc_normal_dmach(self, mach_n1):
        r"""
        Calculate the downstream Mach number from the given upstream Mach
        number :math:`M_{n1}`, in the direction normal to the shock:

        .. math::

          M_{n2} = \sqrt{\frac{(\gamma-1)M_{n1}^2 + 2}
                              {2\gamma M_{n1}^2 - (\gamma-1)}}

        This method accepts scalar:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> round(ob.calc_normal_dmach(mach_n1=3), 10)
        0.4751909633

        as well as :py:class:`numpy.ndarray`:

        >>> np.round(ob.calc_normal_dmach(mach_n1=np.array([3, 3])), 10).tolist()
        [0.4751909633, 0.4751909633]

        :param mach_n1: Upstream Mach number :math:`M_{n1}` normal to the shock
                        wave, dimensionless.
        """
        # Pull data from self.
        gamma = self.gamma
        # Calculate.
        mach_n1_sq = mach_n1*mach_n1
        result = (gamma - 1) * mach_n1_sq + 2
        result /= 2*gamma*mach_n1_sq - (gamma - 1)
        return np.sqrt(result)

    def calc_flow_angle(self, mach1, beta):
        r"""
        Calculate the downstream flow angle :math:`\theta` deflected from the
        upstream flow by using :py:meth:`calc_flow_tangent`, in radian.
        
        This method accepts scalar:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> angle = 48.25848/180*np.pi
        >>> round(ob.calc_flow_angle(mach1=4, beta=angle)/np.pi*180, 10)
        32.0000000807

        as well as :py:class:`numpy.ndarray`:

        >>> angle = 48.25848/180*np.pi; angle = np.array([angle, angle])
        >>> np.round((ob.calc_flow_angle(mach1=4, beta=angle)/np.pi*180), 10).tolist()
        [32.0000000807, 32.0000000807]

        See Example 4.6 in [Anderson03]_ for the forward analysis.  The above
        is the inverse analysis.

        :param mach1: Upstream Mach number :math:`M_1`, dimensionless.
        :param beta: Oblique shock angle :math:`\beta` deflected from the
                     upstream flow, in radian.
        """
        return np.arctan(self.calc_flow_tangent(mach1, beta))

    def calc_flow_tangent(self, mach1, beta):
        r"""
        Calculate the trigonometric tangent function :math:`\tan\beta` of the
        downstream flow angle :math:`\theta` deflected from the upstream flow
        by using the :math:`\theta`\ -:math:`\beta`\ -:math:`M` relation:

        .. math::

          \tan\theta = 2\cot\beta
                       \frac{M_1^2\sin^2\beta - 1}
                            {M_1^2(\gamma+\cos2\beta) + 2}

        This method accepts scalar:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> angle = 48.25848/180*np.pi
        >>> round(ob.calc_flow_tangent(mach1=4, beta=angle), 10)
        0.6248693539

        as well as :py:class:`numpy.ndarray`:

        >>> angle = 48.25848/180*np.pi; angle = np.array([angle, angle])
        >>> np.round(ob.calc_flow_tangent(mach1=4, beta=angle), 10).tolist()
        [0.6248693539, 0.6248693539]

        See Example 4.6 in [Anderson03]_ for the forward analysis.  The above
        is the inverse analysis.

        :param mach1: Upstream Mach number :math:`M_1`, dimensionless.
        :param beta: Oblique shock angle :math:`\beta` deflected from the
                     upstream flow, in radian.
        """
        # Pull data from self.
        gamma = self.gamma
        # Calculate.
        mach1_sq = mach1 * mach1
        result = np.sin(beta)
        result *= result
        result *= mach1_sq
        result -= 1
        result /= mach1_sq * (gamma + np.cos(beta*2)) + 2
        result /= np.tan(beta)
        result *= 2
        return result

    def calc_shock_angle(self, mach1, theta, delta=1):
        r"""
        Calculate the downstream shock angle :math:`\beta` deflected from the
        upstream flow by using :py:meth:`calc_shock_tangent`, in radian.
        
        This method accepts scalar:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> angle = 32./180*np.pi
        >>> round(ob.calc_shock_angle(mach1=4, theta=angle, delta=1)/np.pi*180, 10)
        48.2584798722

        as well as :py:class:`numpy.ndarray`:

        >>> angle = np.array([angle, angle])
        >>> np.round(ob.calc_shock_angle(mach1=4, theta=angle, delta=1)/np.pi*180,
        ...          10).tolist()
        [48.2584798722, 48.2584798722]

        See Example 4.6 in [Anderson03]_ for the analysis.

        :param mach1: Upstream Mach number :math:`M_1`, dimensionless.
        :param theta: Downstream flow angle :math:`\theta` deflected from the
                      upstream flow, in radian.
        :param delta: A switching integer :math:`\delta`.  For :math:`\delta =
                      0`, the function gives the solution of strong shock,
                      while for :math:`\delta = 1`, it gives the solution of
                      weak shock.  The default value is 1.
        """
        return np.arctan(self.calc_shock_tangent(mach1, theta, delta))

    def calc_shock_tangent(self, mach1, theta, delta):
        r"""
        Calculate the downstream shock angle :math:`\beta` deflected from the
        upstream flow by using the alternative :math:`\beta`\ -:math:`\theta`\
        -:math:`M` relation:

        .. math::

          \tan\beta =
            \frac{M_1^2 - 1
                + 2\lambda\cos\left(\frac{4\pi\delta + \cos^{-1}\chi}{3}\right)}
                 {3\left(1 + \frac{\gamma-1}{2}M_1^2\right)\tan\theta}

        where :math:`\lambda` and :math:`\chi` are obtained internally by
        calling :py:meth:`calc_shock_tangent_aux`.

        This method accepts scalar:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> angle = 32./180*np.pi
        >>> round(ob.calc_shock_tangent(mach1=4, theta=angle, delta=1), 10)
        1.1207391858

        as well as :py:class:`numpy.ndarray`:

        >>> angle = np.array([angle, angle])
        >>> np.round(ob.calc_shock_tangent(mach1=4, theta=angle, delta=1),
        ...          10).tolist()
        [1.1207391858, 1.1207391858]

        See Example 4.6 in [Anderson03]_ for the analysis.

        :param mach1: Upstream Mach number :math:`M_1`, dimensionless.
        :param theta: Downstream flow angle :math:`\theta` deflected from the
                      upstream flow, in radian.
        :param delta: A switching integer :math:`\delta`.  For :math:`\delta =
                      0`, the function gives the solution of strong shock,
                      while for :math:`\delta = 1`, it gives the solution of
                      weak shock.
        """
        # Pull data from self.
        gamma = self.gamma
        # Calculate.
        mach1_sq = mach1 * mach1
        lmbd, chi = self.calc_shock_tangent_aux(mach1, theta)
        result = np.arccos(chi)
        result += 4 * np.pi * delta
        result /= 3
        result = np.cos(result)
        result *= lmbd
        result *= 2
        result += mach1_sq
        result -= 1
        result /= 3 * (1 + (gamma-1)/2*mach1_sq) * np.tan(theta)
        return result

    def calc_shock_tangent_aux(self, mach1, theta):
        r"""
        Calculate the :math:`\lambda` and :math:`\chi` functions used by
        :py:meth:`calc_shock_tangent`:

        .. math::

          \lambda =
            \sqrt{(M_1^2-1)^2
                - 3\left(1+\frac{\gamma-1}{2}M_1^2\right)
                   \left(1+\frac{\gamma+1}{2}M_1^2\right)\tan^2\theta}

        .. math::

          \chi =
            \frac{(M_1^2-1)^3
                - 9\left(1+\frac{\gamma-1}{2}M_1^2\right)
                   \left(1+\frac{\gamma-1}{2}M_1^2+\frac{\gamma+1}{4}M_1^4\right)
                   \tan^2\theta}
                 {\lambda^3}

        This method accepts scalar:

        >>> ob = ObliqueShockRelation(gamma=1.4)
        >>> lmbd, chi = ob.calc_shock_tangent_aux(mach1=4, theta=32./180*np.pi)
        >>> round(lmbd, 10), round(chi, 10)
        (11.2080188412, 0.7428957121)

        as well as :py:class:`numpy.ndarray`:

        >>> angle = 32./180*np.pi; angle = np.array([angle, angle])
        >>> lmbd, chi = ob.calc_shock_tangent_aux(mach1=4, theta=angle)
        >>> np.round(lmbd, 10).tolist()
        [11.2080188412, 11.2080188412]
        >>> np.round(chi, 10).tolist()
        [0.7428957121, 0.7428957121]

        See Example 4.6 in [Anderson03]_ for the analysis.

        :param mach1: Upstream Mach number :math:`M_1`, dimensionless.
        :param theta: Downstream flow angle :math:`\theta` deflected from the
                      upstream flow, in radian.
        """
        # Pull data from self.
        gamma = self.gamma
        # Calculate common values.
        mach1_sq = mach1 * mach1
        tant_sq = np.tan(theta)
        tant_sq *= tant_sq
        # Calculate :math:`\lambda`.
        result = tant_sq * (1 + (gamma+1)/2*mach1_sq)
        result *= 1 + (gamma-1)/2*mach1_sq
        result *= 3
        result = (mach1_sq - 1)**2 - result
        lmbd = np.sqrt(result)
        # Calculate :math:`\chi`.
        result = (gamma-1)/2*mach1_sq
        result += 1
        result *= result + (gamma+1)/4*mach1_sq*mach1_sq
        result *= tant_sq
        result *= 9
        result = (mach1_sq - 1)**3 - result
        result /= lmbd**3
        return lmbd, result
