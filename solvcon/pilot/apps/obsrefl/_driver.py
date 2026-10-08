# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Boundary-condition groups and solver driver for the oblique-shock
reflection.
"""

import math

from .... import core
from .... import mesh as meshmod

__all__ = [
    'ObliqueShock',
    'ObliqueShockRelation',
]


class ObliqueShockRelation(object):
    """Calculate the flow-property jumps across an oblique shock.

    (Not yet validated.)

    ``beta`` is the shock angle and ``theta`` the flow-deflection angle,
    both in radians and measured from the upstream flow direction; the
    formulas follow chapter 4 of Anderson, Modern Compressible Flow (3rd
    ed.).  Ported from the gas parcel of the legacy solvcon code base.

    :ivar gamma: Ratio of specific heats.
    """

    def __init__(self, gamma):
        self.gamma = gamma

    def calc_density_ratio(self, mach1, beta):
        """Return the density ratio rho2/rho1 across a shock of angle
        ``beta`` at upstream Mach number ``mach1``."""
        gamma = self.gamma
        mn1sq = (mach1 * math.sin(beta)) ** 2
        return (gamma + 1) * mn1sq / ((gamma - 1) * mn1sq + 2)

    def calc_pressure_ratio(self, mach1, beta):
        """Return the pressure ratio p2/p1 across a shock of angle ``beta``
        at upstream Mach number ``mach1``."""
        gamma = self.gamma
        mn1sq = (mach1 * math.sin(beta)) ** 2
        return 1 + 2 * gamma / (gamma + 1) * (mn1sq - 1)

    def calc_temperature_ratio(self, mach1, beta):
        """Return the temperature ratio T2/T1 across a shock of angle
        ``beta`` at upstream Mach number ``mach1``."""
        return (self.calc_pressure_ratio(mach1, beta)
                / self.calc_density_ratio(mach1, beta))

    def calc_normal_dmach(self, mach_n1):
        """Return the downstream Mach number normal to the shock from the
        normal upstream Mach number ``mach_n1``."""
        gamma = self.gamma
        mn1sq = mach_n1 * mach_n1
        return math.sqrt(((gamma - 1) * mn1sq + 2)
                         / (2 * gamma * mn1sq - (gamma - 1)))

    def calc_dmach(self, mach1, beta=None, theta=None, delta=1):
        """Return the downstream Mach number from the upstream Mach number
        ``mach1`` and either the shock angle ``beta`` or the deflection
        angle ``theta`` (with ``delta`` selecting the weak, 1, or strong,
        0, shock branch)."""
        if (beta is None) == (theta is None):
            raise ValueError(
                f"got (beta={beta}, theta={theta}), "
                f"but I need to take either beta or theta")
        if beta is None:
            beta = self.calc_shock_angle(mach1, theta, delta=delta)
        if theta is None:
            theta = self.calc_flow_angle(mach1, beta)
        mach_n1 = mach1 * math.sin(beta)
        return self.calc_normal_dmach(mach_n1) / math.sin(beta - theta)

    def calc_flow_angle(self, mach1, beta):
        """Return the deflection angle theta from the upstream Mach number
        ``mach1`` and the shock angle ``beta``."""
        return math.atan(self.calc_flow_tangent(mach1, beta))

    def calc_flow_tangent(self, mach1, beta):
        """Return tan(theta) through the theta-beta-M relation."""
        gamma = self.gamma
        m1sq = mach1 * mach1
        return (2 / math.tan(beta) * (m1sq * math.sin(beta) ** 2 - 1)
                / (m1sq * (gamma + math.cos(2 * beta)) + 2))

    def calc_shock_angle(self, mach1, theta, delta=1):
        """Return the shock angle beta from the upstream Mach number
        ``mach1`` and the deflection angle ``theta`` (with ``delta``
        selecting the weak, 1, or strong, 0, shock branch)."""
        return math.atan(self.calc_shock_tangent(mach1, theta, delta))

    def calc_shock_tangent(self, mach1, theta, delta):
        """Return tan(beta) through the closed-form inversion of the
        theta-beta-M relation."""
        gamma = self.gamma
        m1sq = mach1 * mach1
        lmbd, chi = self.calc_shock_tangent_aux(mach1, theta)
        num = (m1sq - 1 + 2 * lmbd
               * math.cos((4 * math.pi * delta + math.acos(chi)) / 3))
        den = 3 * (1 + (gamma - 1) / 2 * m1sq) * math.tan(theta)
        return num / den

    def calc_shock_tangent_aux(self, mach1, theta):
        """Return the (lambda, chi) auxiliary pair of the closed-form
        theta-beta-M inversion used by :meth:`calc_shock_tangent`."""
        gamma = self.gamma
        m1sq = mach1 * mach1
        tansq = math.tan(theta) ** 2
        disc = ((m1sq - 1) ** 2
                - 3 * (1 + (gamma - 1) / 2 * m1sq)
                * (1 + (gamma + 1) / 2 * m1sq) * tansq)
        if disc <= 0.0:
            raise ValueError(
                f"no attached shock for mach1={mach1:g} and "
                f"theta={math.degrees(theta):g} deg")
        lmbd = math.sqrt(disc)
        chi = ((m1sq - 1) ** 3
               - 9 * (1 + (gamma - 1) / 2 * m1sq)
               * (1 + (gamma - 1) / 2 * m1sq + (gamma + 1) / 4 * m1sq * m1sq)
               * tansq) / lmbd ** 3
        return lmbd, chi


class ObliqueShock(object):
    """Drive the CESE Euler solver over the oblique-shock reflection.
    """

    def __init__(self):
        self.gamma = None
        self.density = None
        self.pressure = None
        self.mach = None
        self.speedofsound = None
        self.velocity = None
        self.relation = None
        self.theta = None
        self.shock_angle = None
        self.density2 = None
        self.pressure2 = None
        self.mach2 = None
        self.velocity2 = None
        self.shock_angle2 = None
        self.density3 = None
        self.pressure3 = None
        self.mach3 = None
        self.velocity3 = None
        self.mesher = None
        self.mesh = None
        # Numerical solver core (EulerCore).
        self.svr = None

    def build_constant(self, gamma=1.4, density=1.0, pressure=1.0, mach=3.0,
                       angle=10.0):
        """Fix the flow states on the two sides of the incident shock.

        The free stream (zone 1) enters horizontally at the given Mach
        number; ``angle`` is the flow deflection across the incident shock
        in degrees.  The oblique-shock relations give the post-shock state
        (zone 2), whose velocity points ``angle`` below horizontal; the
        driver imposes it at the top boundary to anchor the incident
        shock.  The reflected shock turns zone 2 back to horizontal, and
        the same relations give the state behind it (zone 3), which the
        steady solution has to reach; it is kept for validation and
        display, not imposed anywhere.
        """
        self.gamma = gamma
        self.density = density
        self.pressure = pressure
        self.mach = mach
        self.speedofsound = math.sqrt(gamma * pressure / density)
        self.velocity = mach * self.speedofsound
        self.relation = ObliqueShockRelation(gamma=gamma)
        self.theta = theta = math.radians(angle)
        self.shock_angle = beta = self.relation.calc_shock_angle(mach, theta)
        self.density2 = density * self.relation.calc_density_ratio(mach, beta)
        self.pressure2 = (
            pressure * self.relation.calc_pressure_ratio(mach, beta))
        self.mach2 = self.relation.calc_dmach(mach, beta=beta)
        speed2 = self.mach2 * math.sqrt(gamma * self.pressure2 / self.density2)
        self.velocity2 = (speed2 * math.cos(theta),
                          -speed2 * math.sin(theta))
        mach2 = self.mach2
        self.shock_angle2 = beta2 = self.relation.calc_shock_angle(
            mach2, theta)
        self.density3 = (
            self.density2 * self.relation.calc_density_ratio(mach2, beta2))
        self.pressure3 = (
            self.pressure2 * self.relation.calc_pressure_ratio(mach2, beta2))
        self.mach3 = self.relation.calc_dmach(mach2, beta=beta2)
        speed3 = self.mach3 * math.sqrt(gamma * self.pressure3 / self.density3)
        self.velocity3 = (speed3, 0.0)

    def build_numerical(self, cell_type='unstructured', time_increment=2.e-3,
                        sigma0=3.0, taumin=0.0, tauscale=1.0, **mesher_kw):
        """After :meth:`build_constant` is done, build the numerical solver
        :attr:`svr` over the selected mesh flavor.
        """
        if None is self.gamma:
            raise ValueError("constants are not set; call build_constant()")
        self.mesher = meshmod.RectDomainMesher(**mesher_kw)
        self.mesh = self.mesher.make_mesh(cell_type=cell_type)
        # The core prepares the CE geometry (prepare_ce) on construction.
        svr = core.EulerCore(mesh=self.mesh, time_increment=time_increment)
        svr.sigma0 = sigma0
        svr.taumin = taumin
        svr.tauscale = tauscale
        svr.init_solution(gamma=self.gamma, rho=self.density,
                          v=[self.velocity, 0.0], p=self.pressure)
        # The mesher attached one named group per domain edge; read the
        # face lists back from the mesh instead of re-classifying.
        left, top, bottom, right = (
            self.mesh.bc(name).facn.ndarray[:, 0].tolist()
            for name in self.mesher.BOUNDARY_NAMES)
        svr.add_inlet(left, value=[self.density, self.velocity, 0.0,
                                   self.pressure, self.gamma])
        svr.add_inlet(top, value=[self.density2, self.velocity2[0],
                                  self.velocity2[1], self.pressure2,
                                  self.gamma])
        svr.add_slipwall(bottom)
        svr.add_nonrefl(right)
        # Prime the ghost rows from the initial interior state so the first
        # substep does not read zero-filled ghosts.
        svr.bc_soln()
        svr.bc_dsoln()
        self.svr = svr

    def march(self, steps):
        """March the solver the requested number of full CESE steps."""
        self.svr.march(steps=steps)

    def zone_states(self):
        """Return the analytic ``(rho, vx, vy, p)`` of zones 1, 2, and 3.

        Zone 1 is the free stream, zone 2 sits between the incident and the
        reflected shock, and zone 3 sits behind the reflected shock.  The
        steady solution has to hold these values; a display can derive any
        scalar field from them for comparison against the computed field.
        """
        return [(self.density, self.velocity, 0.0, self.pressure),
                (self.density2, self.velocity2[0], self.velocity2[1],
                 self.pressure2),
                (self.density3, self.velocity3[0], self.velocity3[1],
                 self.pressure3)]

    def shock_path(self):
        """Return the analytic shock polyline over the built mesh.

        Three corners: where the incident shock enters at the upper-left
        corner, where it reflects off the bottom wall, and where the
        reflected shock leaves the domain.  Needs :meth:`build_numerical`,
        which sets the mesher whose extents the path is cut to.
        """
        if None is self.mesher:
            raise ValueError("mesh is not built; call build_numerical()")
        msh = self.mesher
        xhit = msh.x0 + (msh.y1 - msh.y0) / math.tan(self.shock_angle)
        if xhit >= msh.x1:
            # The domain is too short for the reflection; the incident
            # shock leaves through the outflow.
            yout = msh.y1 - (msh.x1 - msh.x0) * math.tan(self.shock_angle)
            return [(msh.x0, msh.y1), (msh.x1, yout)]
        # The reflected shock runs at shock_angle2 from the zone-2 flow,
        # which is already theta below horizontal.
        slope = math.tan(self.shock_angle2 - self.theta)
        xend = min(msh.x1, xhit + (msh.y1 - msh.y0) / slope)
        return [(msh.x0, msh.y1), (xhit, msh.y0),
                (xend, msh.y0 + (xend - xhit) * slope)]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
