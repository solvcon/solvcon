# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The shock tube problem with its analytic solution.
"""

import numpy as np

from ...onedim import euler


class ShockTube:
    """
    Shock tube problem.  See the CESE note and Modern Compressible Flow: With
    Historical Perspective, 3/e, 2003, by J D Anderson. ISBN 0-07-242443-5.
    """

    R = 8.31446261815324

    def __init__(self):
        self.gamma = None

        # Zones 1 and 5 are given.
        self.velocity1 = None
        self.pressure1 = None
        self.density1 = None
        self.temperature1 = None
        self.internal_energy1 = None
        self.entropy1 = None
        self.speedofsound1 = None
        self.velocity5 = None
        self.pressure5 = None
        self.density5 = None
        self.temperature5 = None
        self.internal_energy5 = None
        self.entropy5 = None
        self.speedofsound5 = None

        # Zone 4 is calculated by using the normal shock wave.
        self.velocity4 = None
        self.pressure4 = None
        self.density4 = None
        self.temperature4 = None
        self.internal_energy4 = None
        self.entropy4 = None
        self.speedofsound4 = None

        self.velocity_shock = None
        self.velocity_con = None

        # Zone 3 is calculated by using the expansion wave.
        self.velocity3 = None
        self.pressure3 = None
        self.density3 = None
        self.temperature3 = None
        self.internal_energy3 = None
        self.entropy3 = None
        self.speedofsound3 = None

        # Field data of the analytical solution.
        self.coord_field = None
        self.density_field = None
        self.velocity_field = None
        self.pressure_field = None
        self.temperature_field = None
        self.internal_energy_field = None
        self.entropy_field = None

        # Numerical solver (Euler1DSolver).
        self.svr = None

    def build_numerical(self, xmin, xmax, ncoord, time_increment=0.05,
                        xdiaphragm=0.0, keep_edge=False):
        """
        After :py:meth:`build_constant` is done, optionally build the
        numerical solver :py:attr:`svr`.

        :param xmin:
        :param xmax:
        :param ncoord:
        :param time_increment:
        :param xdiaphragm: It should be set to 0.0.
        :return: None
        """
        if None is self.gamma:
            raise ValueError("constants are not set; call build_constant()")

        # Initialize the numerical solver.
        self.svr = euler.Euler1DSolver(xmin, xmax, ncoord,
                                       time_increment=time_increment,
                                       keep_edge=keep_edge)

        # Fill gamma.
        self.svr.gamma.fill(self.gamma)
        # Determine u0 and u2 value at left and right.
        u0_left = self.density1
        u0_right = self.density5
        u2_left = self.svr.calc_u2(self.gamma, self.density1, 0.0,
                                   self.pressure1)
        u2_right = self.svr.calc_u2(self.gamma, self.density5, 0.0,
                                    self.pressure5)
        # Create Boolean selection arrays for left and right.
        slct_left = self.svr.coord < xdiaphragm
        slct_right = np.logical_not(slct_left)
        # u0
        self.svr.so0[slct_left, 0] = u0_left
        self.svr.so0[slct_right, 0] = u0_right
        # u1
        self.svr.so0[:, 1] = 0.0
        # u2
        self.svr.so0[slct_left, 2] = u2_left
        self.svr.so0[slct_right, 2] = u2_right
        # Initialize derivative to zero.
        self.svr.so1.fill(0)
        # Setup the rest in the solver for time-marching.
        self.svr.setup_march()

    def build_constant(self, gamma, pressure1, density1, pressure5, density5):
        # Set the given value in zones 1 (left) and 5 (right).
        self.gamma = gamma
        self.velocity1 = 0.0
        self.pressure1 = pressure1
        self.density1 = density1
        self.temperature1 = pressure1 / (density1 * self.R)
        self.speedofsound1 = self.calc_speedofsound(
            pressure=pressure1, density=density1)
        self.internal_energy1 = self.calc_internal_energy(self.pressure1,
                                                          self.density1)
        self.entropy1 = self.calc_entropy(self.pressure1, self.density1)

        self.velocity5 = 0.0
        self.pressure5 = pressure5
        self.density5 = density5
        self.temperature5 = pressure5 / (density5 * self.R)
        self.speedofsound5 = self.calc_speedofsound(
            pressure=pressure5, density=density5)
        self.internal_energy5 = self.calc_internal_energy(self.pressure5,
                                                          self.density5)
        self.entropy5 = self.calc_entropy(self.pressure5, self.density5)

        # Use the given value to determine the shock strength (between zones 4
        # and 5).
        p45 = self.calc_pressure45()

        # Use the normal shock relationship.
        self.velocity4 = self.calc_velocity4(pressure45=p45)
        self.pressure4 = p45 * self.pressure5
        self.density4 = self.calc_density45(pressure45=p45) * self.density5
        self.temperature4 = self.calc_temperature45(p45) * self.temperature5
        self.speedofsound4 = self.calc_speedofsound(
            pressure=self.pressure4, density=self.density4)
        self.internal_energy4 = self.calc_internal_energy(self.pressure4,
                                                          self.density4)
        self.entropy4 = self.calc_entropy(self.pressure4, self.density4)

        _ = np.sqrt((gamma + 1) / (2 * gamma) * (p45 - 1) + 1)
        self.velocity_shock = self.speedofsound5 * _
        self.velocity_con = self.velocity4

        # Velocity and pressure are the same in zones 3 and 4.
        self.velocity3 = self.velocity4
        self.pressure3 = self.pressure4

        # Use the expansion wave for density in zone 3.
        self.density3 = self.calc_density2(x=0, t=0, velocity2=self.velocity3)
        self.temperature3 = self.calc_temperature2(x=0, t=0,
                                                   velocity2=self.velocity3)
        self.internal_energy3 = self.calc_internal_energy(self.pressure3,
                                                          self.density3)
        self.entropy3 = self.calc_entropy(self.pressure3, self.density3)

        self.speedofsound3 = self.calc_speedofsound(
            pressure=self.pressure3, density=self.density3)

    def calc_pressure45(self, tolerance=1.e-10, maxiter=50):
        """
        Use secant method to calculate the shock strength.

        :param tolerance: Convergence tolerance
        :param maxiter: Maximum iterations of the secant method
        :return: Pressure ratio in zones 4 and 5 ($p_4/p_5$)
        """
        p15 = self.pressure1 / self.pressure5
        a51 = self.speedofsound5 / self.speedofsound1
        gamma = self.gamma

        def _f(p45):
            v = p45 - 1
            nume = (gamma - 1) * a51 * v
            deno = np.sqrt(2 * gamma * (2 * gamma + (gamma + 1) * v))
            v = (1 - nume / deno) ** (-2 * gamma / (gamma - 1))
            return p15 - p45 * v

        p45prev = p15
        residue_prev = _f(p45prev)
        p45 = p45prev / 2
        residue = _f(p45)
        slope = (residue - residue_prev) / (p45 - p45prev)
        count = maxiter
        while np.abs(residue) > tolerance and count > 0:
            p45prev = p45
            p45 -= residue / slope
            residue_prev = residue
            residue = _f(p45)
            slope = (residue - residue_prev) / (p45 - p45prev)
            # DEBUG
            # print("count: %d, p45: %f, residue: %f" % (count, p45, residue))
            count -= 1

        return p45

    def calc_density45(self, pressure45):
        gpn1 = (self.gamma + 1) / (self.gamma - 1)
        rho45 = (1 + gpn1 * pressure45) / (gpn1 + pressure45)
        return rho45

    def calc_temperature45(self, pressure45):
        gpn1 = (self.gamma + 1) / (self.gamma - 1)
        temp45 = pressure45 * ((gpn1 + pressure45) / (1 + gpn1 * pressure45))
        return temp45

    def calc_velocity4(self, pressure45):
        c = self.speedofsound5 / self.gamma
        nume = 2 * self.gamma / (self.gamma + 1)
        deno = pressure45 + (self.gamma - 1) / (self.gamma + 1)
        return c * (pressure45 - 1) * np.sqrt(nume / deno)

    def calc_velocity2(self, x, t):
        return 2 / (self.gamma + 1) * (self.speedofsound1 + x / t)

    def calc_speedofsound2_ratio(self, x, t, velocity2=None):
        if None is velocity2:
            velocity2 = self.calc_velocity2(x, t)
        c = velocity2 / self.speedofsound1
        return 1 - (self.gamma - 1) / 2 * c

    def calc_pressure2(self, x, t):
        c1 = self.calc_speedofsound2_ratio(x, t)
        c2 = 2 * self.gamma / (self.gamma - 1)
        return self.pressure1 * (c1 ** c2)

    def calc_density2(self, x, t, velocity2=None):
        c1 = self.calc_speedofsound2_ratio(x, t, velocity2=velocity2)
        c2 = 2 / (self.gamma - 1)
        return self.density1 * (c1 ** c2)

    def calc_temperature2(self, x, t, velocity2=None):
        return self.temperature1 * (self.calc_speedofsound2_ratio(
                                        x, t, velocity2=velocity2)) ** 2

    def calc_speedofsound(self, pressure, density):
        return np.sqrt(self.gamma * pressure / density)

    def calc_internal_energy(self, pressure, density):
        return pressure / (density * (self.gamma - 1))

    def calc_entropy(self, pressure, density):
        return pressure / (density ** self.gamma)

    def build_field(self, t, coord=None):
        """
        Populate the field data using the analytical solution.

        :param t:
        :param coord: If None, take the coordinate from the numerical solver.
        :return: None
        """
        if coord is None:
            # Set the x-coordinate for numerical solver.
            coord = self.svr.coord[self.svr.xindices]
        # Make a copy; no write back to argument.
        self.coord_field = coord.copy()

        # Determine the zone location and the Boolean selection arrays.
        loc12, loc23, loc34, loc45 = self.calc_locations(t=t)
        slct1 = self.coord_field < loc12
        slct2 = np.logical_and(self.coord_field >= loc12,
                               self.coord_field < loc23)
        slct3 = np.logical_and(self.coord_field >= loc23,
                               self.coord_field < loc34)
        slct4 = np.logical_and(self.coord_field >= loc34,
                               self.coord_field < loc45)
        slct5 = self.coord_field >= loc45

        # Create the field buffers.
        self.density_field = np.zeros_like(self.coord_field, dtype='float64')
        self.velocity_field = np.zeros_like(self.coord_field, dtype='float64')
        self.pressure_field = np.zeros_like(self.coord_field, dtype='float64')
        self.temperature_field = np.zeros_like(self.coord_field,
                                               dtype='float64')
        self.internal_energy_field = np.zeros_like(self.coord_field,
                                                   dtype='float64')
        self.entropy_field = np.zeros_like(self.coord_field, dtype='float64')

        # Set value in the zones whose value is constant.
        def _set_zone(slct, zone_density, zone_velocity, zone_pressure,
                      zone_temperature, zone_int_energy, zone_entropy):
            self.density_field[slct] = zone_density
            self.velocity_field[slct] = zone_velocity
            self.pressure_field[slct] = zone_pressure
            self.temperature_field[slct] = zone_temperature
            self.internal_energy_field[slct] = zone_int_energy
            self.entropy_field[slct] = zone_entropy

        _set_zone(slct1, self.density1, self.velocity1, self.pressure1,
                  self.temperature1, self.internal_energy1, self.entropy1)
        _set_zone(slct3, self.density3, self.velocity3, self.pressure3,
                  self.temperature3, self.internal_energy3, self.entropy3)
        _set_zone(slct4, self.density4, self.velocity4, self.pressure4,
                  self.temperature4, self.internal_energy4, self.entropy4)
        _set_zone(slct5, self.density5, self.velocity5, self.pressure5,
                  self.temperature5, self.internal_energy5, self.entropy5)

        # Expansion wave in zone 2 needs an explicit loop.
        xidx = np.arange(len(coord), dtype='uint64')
        for _idx, _x in zip(xidx[slct2], coord[slct2]):
            self.density_field[_idx] = self.calc_density2(_x, t)
            self.velocity_field[_idx] = self.calc_velocity2(_x, t)
            self.pressure_field[_idx] = self.calc_pressure2(_x, t)
            self.temperature_field[_idx] = self.calc_temperature2(
                _x, t, self.velocity_field[_idx])
            self.internal_energy_field[_idx] = self.calc_internal_energy(
                self.pressure_field[_idx], self.density_field[_idx])
            self.entropy_field[_idx] = self.calc_entropy(
                    self.pressure_field[_idx],
                    self.density_field[_idx])

    def calc_locations(self, t):
        """
        Return array of [x_zone12, x_zone23, x_zone34, x_zone45]
        """
        return np.array([
            # zone12; use speed of sound in zone 1:
            -self.speedofsound1 * t,
            # zone23; the right-most characteristic in zone 2 (expansion wave):
            (self.velocity3 - self.speedofsound3) * t,
            # zone34; contact surface moves at the speed ov v_3 = v_4:
            self.velocity3 * t,
            # zone45; shock wave speed:
            self.velocity_shock * t,
        ], dtype='float64')


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
