# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The body-cell views of the multi-dimensional Euler solution.

The mesh is three triangles fanning around the origin, small enough that
the body slice, the cell volumes, and the residual are all known by hand.
"""

import math
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import solvcon
from solvcon.multidim import euler


class _EulerFieldTB:
    """The three-triangle fan the field checks read."""

    #: Total area of the three triangles.
    AREA = 2.0
    #: Full CESE step; a substep advances half of it.
    DT = 1.e-2

    def setUp(self):
        mh = solvcon.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
        mh.ndcrd[:, :] = [(0, 0), (-1, -1), (1, -1), (0, 1)]
        mh.cltpn.fill(solvcon.StaticMesh.TRIANGLE)
        mh.clnds[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)
        mh.build_interior(do_metric=True)
        mh.build_boundary()
        mh.build_ghost()
        self.mesh = mh
        self.svr = solvcon.EulerCore(mesh=mh, time_increment=self.DT)
        self.svr.init_solution(gamma=1.4, rho=1.0, v=[0.0, 0.0], p=1.0)
        self.field = euler.EulerField(self.svr, mh)

    def _seed(self, name, densities):
        self._seed_column(name, 0, densities)

    def _seed_column(self, name, icol, values):
        arr = getattr(self.svr, name)
        arr.ndarray[arr.nghost:, icol] = values


class EulerFieldTC(_EulerFieldTB, unittest.TestCase):
    """The body-cell views and the reductions over them."""

    def test_views_drop_the_ghost_prefix(self):
        # The tables count their own ghost rows, so a body-cell view spans
        # ncell rows however many ghosts the mesh needed.
        ngst, ncell = self.svr.ngstcell, self.svr.ncell
        self.assertEqual(ngst, self.svr.so0n.nghost)
        rows = self.field.conserved
        self.assertEqual(rows.shape[0], ncell)
        np.testing.assert_array_equal(
            rows, self.svr.so0n.ndarray[ngst:ngst + ncell])

    def test_views_alias_the_solver_memory(self):
        self._seed('so0n', (2.0, 3.0, 4.0))
        np.testing.assert_array_equal(self.field.density, (2.0, 3.0, 4.0))

    def test_geometry_spans_the_body_cells(self):
        ncell = self.svr.ncell
        self.assertEqual(self.field.centroid.shape, (ncell, self.svr.ndim))
        volume = self.field.volume
        self.assertEqual(volume.shape[0], ncell)
        assert_almost_equal(float(volume.sum()), self.AREA)

    def test_overall_mass_integrates_the_density(self):
        # init_solution seeds a uniform density of one, so the mass is the
        # domain area.
        assert_almost_equal(self.field.calc_overall_mass(), self.AREA)
        self._seed('so0n', (2.0, 2.0, 2.0))
        assert_almost_equal(self.field.calc_overall_mass(), 2.0 * self.AREA)

    def test_residual_is_the_rms_change_over_the_half_step(self):
        self._seed('so0c', (1.0, 1.0, 1.0))
        self._seed('so0n', (1.1, 0.9, 1.0))
        expected = math.sqrt((0.1 ** 2 + 0.1 ** 2) / 3.0) / (self.DT / 2.0)
        assert_almost_equal(self.field.calc_residual(), expected)

    def test_residual_vanishes_on_a_frozen_solution(self):
        self._seed('so0c', (1.0, 2.0, 3.0))
        self._seed('so0n', (1.0, 2.0, 3.0))
        self.assertEqual(self.field.calc_residual(), 0.0)

    def test_residual_of_another_conserved_variable(self):
        # Every column of the solution has an older value to difference
        # against, and a run settles them one at a time; density alone can
        # sit still while the momentum still moves.
        self._seed('so0c', (1.0, 1.0, 1.0))
        self._seed('so0n', (1.0, 1.0, 1.0))
        self._seed_column('so0c', 2, (0.0, 0.0, 0.0))
        self._seed_column('so0n', 2, (0.3, -0.3, 0.0))
        self.assertEqual(self.field.calc_residual('density'), 0.0)
        expected = math.sqrt((0.3 ** 2 + 0.3 ** 2) / 3.0) / (self.DT / 2.0)
        assert_almost_equal(self.field.calc_residual('momy'), expected)

    def test_the_conserved_variables_span_the_solution(self):
        # The names are the columns of the solution table, so a 2D run
        # carries the two momentum components between density and energy.
        self.assertEqual(('density', 'momx', 'momy', 'total_energy'),
                         self.field.conserveds)
        self.assertEqual(self.svr.neq, len(self.field.conserveds))
        self.assertEqual(3, self.field.conserved_column('total_energy'))

    def test_residual_of_a_derived_field_is_refused(self):
        # Pressure is derived, not marched, so nothing holds its older
        # value to difference against.
        with self.assertRaises(ValueError):
            self.field.calc_residual('pressure')


class DerivedFieldTC(_EulerFieldTB, unittest.TestCase):
    """The derived fields of a state whose every value is known by hand.

    The three cells all carry rho = 1, u = (2, 0), and p = 1 at gamma = 1.4,
    seeded as the conserved row ``[rho, rho*u, rho*v, E]`` with
    ``E = p/(gamma-1) + 0.5*rho*|u|^2``.
    """

    ENERGY = 1.0 / 0.4 + 2.0

    def setUp(self):
        super().setUp()
        self.field.conserved[:] = (1.0, 2.0, 0.0, self.ENERGY)

    def test_primitive_fields(self):
        assert_almost_equal(self.field.density, (1.0,) * 3)
        assert_almost_equal(self.field.velx, (2.0,) * 3)
        assert_almost_equal(self.field.vely, (0.0,) * 3)
        assert_almost_equal(self.field.speed, (2.0,) * 3)
        assert_almost_equal(self.field.total_energy, (self.ENERGY,) * 3)

    def test_pressure_and_mach(self):
        # Pressure inverts the energy relation; Mach divides the speed by the
        # local speed of sound sqrt(gamma p / rho).
        assert_almost_equal(self.field.pressure, (1.0,) * 3)
        assert_almost_equal(self.field.mach, (2.0 / math.sqrt(1.4),) * 3)

    def test_every_advertised_field_is_reachable_by_name(self):
        # A control or a report holds a name, so each one has to reach the
        # property that derives it.
        for name in euler.EulerField.FIELDS:
            assert_almost_equal(self.field.field(name),
                                getattr(self.field, name))

    def test_unknown_field_raises(self):
        with self.assertRaises(ValueError):
            self.field.field('nonesuch')

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
