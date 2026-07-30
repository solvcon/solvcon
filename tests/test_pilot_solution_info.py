# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import math
import unittest

import numpy as np

import solvcon
from solvcon.multidim.euler import oblique

try:
    from solvcon import pilot
    from solvcon.pilot._euler import _solution_info
except ImportError:
    pilot = None


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class ComputeFieldTC(unittest.TestCase):
    """The derived-field math is independent of Qt, so it runs in CI."""

    # A single cell at rho=1, u=(2, 0), p=1, gamma=1.4; the conserved row is
    # [rho, rho*u, rho*v, E] with E = p/(gamma-1) + 0.5*rho*|u|^2.
    GAMMA = 1.4
    CONS = np.array([[1.0, 2.0, 0.0, 1.0 / 0.4 + 2.0]], dtype='float64')

    def _field(self, name):
        gamma = np.array([self.GAMMA], dtype='float64')
        return _solution_info.SolutionPanel.compute_field(
            name, self.CONS, gamma, ndim=2)[0]

    def test_primitive_fields(self):
        self.assertAlmostEqual(self._field('density'), 1.0)
        self.assertAlmostEqual(self._field('velocity-x'), 2.0)
        self.assertAlmostEqual(self._field('velocity-y'), 0.0)
        self.assertAlmostEqual(self._field('speed'), 2.0)
        self.assertAlmostEqual(self._field('energy'), 1.0 / 0.4 + 2.0)

    def test_pressure_and_mach(self):
        # Pressure inverts the energy relation; Mach divides the speed by the
        # local speed of sound sqrt(gamma p / rho).
        self.assertAlmostEqual(self._field('pressure'), 1.0)
        self.assertAlmostEqual(self._field('mach'), 2.0 / math.sqrt(1.4))

    def test_unknown_field_raises(self):
        with self.assertRaises(ValueError):
            self._field('nonesuch')

    def test_solver_field_excludes_ghost(self):
        # solver_field must slice off the ghost rows so the field spans only
        # the body cells, matching the raw density column.
        shock = oblique.ObliqueShock()
        shock.build_constant()
        shock.build_numerical(cell_type='quad', nx=8, ny=4)
        shock.march(2)
        svr = shock.svr
        density = _solution_info.SolutionPanel.solver_field(svr, 'density')
        self.assertEqual(density.shape[0], svr.ncell)
        np.testing.assert_array_equal(
            density, svr.so0n.ndarray[svr.ngstcell:, 0])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
