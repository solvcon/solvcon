# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
NACA airfoil shape
"""

import numpy as np

from solvcon import core

__all__ = [
    'Naca4',
    'Naca4Sampler',
]


class Naca4(object):
    """
    Returns points in [0 1] for the given 4 digit NACA number string.
    Reference from
    https://ntrs.nasa.gov/api/citations/19930091108/downloads/19930091108.pdf
    and https://github.com/dgorissen/naca

    See also https://en.wikipedia.org/wiki/NACA_airfoil
    """

    # FUTURE: This class does not use numpy array batch code because the code
    # should be ported into C++ for speed.

    A0 = +0.2969
    A1 = -0.1260
    A2 = -0.3516
    A3 = +0.2843
    A4O = -0.1015  # non-zero/open trailing edge
    A4C = -0.1036  # zero/close trailing edge

    def __init__(self, number='', camber=None, pos=None, thick=None,
                 open_trailing_edge=False, cosine_spacing=False):
        # Parse the 4-digit NACA number if it is valid.
        if len(number) == 4:
            if camber is None:
                camber = float(number[0]) / 100.0
            if pos is None:
                pos = float(number[1]) / 10.0
            if thick is None:
                thick = float(number[2:]) / 100.0
        # Check for the required arguments (may be parsed from number).
        if camber is None:
            raise ValueError('camber cannot be None')
        if pos is None:
            raise ValueError('pos cannot be None')
        if thick is None:
            raise ValueError('thick cannot be None')
        # Set the attributes.
        self.camber = camber
        self.pos = pos
        self.thick = thick
        self.open_trailing_edge = open_trailing_edge
        self.cosine_spacing = cosine_spacing

    def calc_yt(self, xc):
        r"""
        Calculate the y value for the symmetric profile.

        .. math::

          y_t = 5t(0.2969 \sqrt{x} - 0.1260 x - 0.3516 x^2 + 0.2843 x^3
                 - 0.1015 x^4)

        Zero (close) trailing edge:

        .. math::

          y_t = 5t(0.2969 \sqrt{x} - 0.1260 x - 0.3516 x^2 + 0.2843 x^3
                 - 0.1036x^4)

        :param xc: Location along the x direction
        :return: The y value (positive)
        """
        xh = np.sqrt(xc)
        x2 = xc * xc
        x3 = x2 * xc
        x4 = x3 * xc
        a0 = self.A0
        a1 = self.A1
        a2 = self.A2
        a3 = self.A3
        a4 = self.A4O if self.open_trailing_edge else self.A4C
        t = self.thick
        yt = 5.0 * t * (a0 * xh + a1 * xc + a2 * x2 + a3 * x3 + a4 * x4)
        return yt

    def calc_ul(self, xc):
        r"""
        Calculate the coordinates (xu, yu) and (xl, yl) (u for upper and l for
        lower) at the corresponding location on the center/camber.

        For asymmetric camber (pos and camber are non-zero), the mean camber
        line is calculated (:math:`m` is the maximum camber, :math:`p` is the
        position of the maximum camber):

        .. math:

          y_c = \frac{m}{p^2}(2px - x^2), \quad 0 \le x \le p \\
          y_c = \frac{m}{(1-p)^2}(1 - 2p + 2px - x^2), \quad p \le x \le 1

        The thickness needs to be perpendicular to the camber line, the upper
        and lower profile should be calculated by:

        .. math:

          x_u = x - y_t\sin\theta , \; y_u = y_c + y_t\cos\theta \\
          x_l = x + y_t\sin\theta , \; y_l = y_c - y_t\cos\theta

        The angle :math:`\theta` should be determined by:

        .. math:

          \theta = \arctan\frac{\mathrm{d}y_c}{\mathrm{d}x} \\
          \frac{\mathrm{d}y_c}{\mathrm{d}x} = \frac{2m}{p^2}(p-x),
            \quad 0 \le x \le p \\
          \frac{\mathrm{d}y_c}{\mathrm{d}x} = \frac{2m}{(1-p)^2}(p-x),
            \quad p \le x \le 1

        :param xc: x coordinate at center/camber
        :return: xu, yu, xl, yl
        """
        yt = self.calc_yt(xc)
        pos = self.pos
        if pos == 0:
            return xc, yt, xc, -yt
        else:
            camber = self.camber
            if xc <= pos:
                xc1 = xc
                yc1 = camber / np.power(pos, 2) * xc1 * (2 * pos - xc1)
                dyc1_dx = camber / np.power(pos, 2) * (2 * pos - 2 * xc1)
                yc = yc1
                dyc_dx = dyc1_dx
            else:
                xc2 = xc
                yc2 = camber / np.power(1 - pos, 2) * (1 - 2 * pos + xc2) * (
                        1 - xc2)
                dyc2_dx = camber / np.power(1 - pos, 2) * (2 * pos - 2 * xc2)
                yc = yc2
                dyc_dx = dyc2_dx

            theta = np.arctan(dyc_dx)

            return (xc - yt * np.sin(theta), yc + yt * np.cos(theta),
                    xc + yt * np.sin(theta), yc - yt * np.cos(theta))

    def calc_points(self, npoint):
        """
        :param npoint: Number of sample points on the center/camber
        :return: Arrays for xu, yu, xl, yl
        """
        # Make xc array.
        if self.cosine_spacing:
            _ = np.linspace(0.0, np.pi, npoint + 1, dtype='float64')
            xcarr = 0.5 * (1.0 - np.cos(_))
        else:
            xcarr = np.linspace(0.0, 1.0, npoint + 1, dtype='float64')
        nmid = len(xcarr)
        pp = core.PointPadFp64(ndim=2, nelem=nmid * 2 - 1)
        for i, xc in enumerate(xcarr):
            xu, yu, xl, yl = self.calc_ul(xc)
            pp.set_at(nmid - 1 - i, xu, yu)
            if i > 0:
                pp.set_at(nmid - 1 + i, xl, yl)
        return pp


class Naca4Sampler(object):
    """
    Sample the profile of Naca4 airfoil.
    """

    def __init__(self, world, naca4):
        self.world = world
        self.naca4 = naca4
        self.points = None

    def populate_points(self, npoint, fac, off_x, off_y):
        """
        Populate the points on the airfoil profile.

        :param npoint: Number of points
        :param fac: Scaling factor
        :param off_x: Offset in x
        :param off_y: Offset in y
        :return: None
        """
        self.points = self.naca4.calc_points(npoint)
        self.points.x.ndarray[:] *= fac
        self.points.y.ndarray[:] *= fac
        self.points.x.ndarray[:] += off_x
        self.points.y.ndarray[:] += off_y

    def draw_line(self):
        """
        Draw by connecting points using line segments.

        :return: None
        """
        Point = core.Point3dFp64
        points = self.points
        world = self.world
        for it in range(len(points) - 1):
            p0 = points.get_at(it)
            p1 = points.get_at(it + 1)
            world.add_segment(Point(p0.x, p0.y, 0), Point(p1.x, p1.y, 0))

    def draw_cbc(self, spacing=0.01):
        """
        Draw by connecting points using cubic Bezier curves.

        :param spacing: Spacing for sampling cubic Bezier curve.
        :return: None
        """
        Point = core.Point3dFp64
        points = self.points
        world = self.world

        ptsx = points.x.ndarray
        ptsy = points.x.ndarray
        segments = np.hypot(ptsx[:-1] - ptsx[1:],
                            ptsy[:-1] - ptsy[1:]) // spacing
        nsample = np.where(segments > 2, segments - 1, 2).astype(int)
        for it in range(len(points) - 1):
            p0 = np.array(points[it])
            p3 = np.array(points[it + 1])
            p1 = p0 + (1 / 3) * (p3 - p0)
            p2 = p0 + (2 / 3) * (p3 - p0)
            b = world.add_bezier(p0=Point(p0[0], p0[1], 0),
                                 p1=Point(p1[0], p1[1], 0),
                                 p2=Point(p2[0], p2[1], 0),
                                 p3=Point(p3[0], p3[1], 0))
            b.sample(nsample[it])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
