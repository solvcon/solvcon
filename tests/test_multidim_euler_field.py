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


class EulerFieldTC(unittest.TestCase):

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
        arr = getattr(self.svr, name)
        arr.ndarray[arr.nghost:, 0] = densities

    def test_views_drop_the_ghost_prefix(self):
        # The tables count their own ghost rows, so a body-cell view spans
        # ncell rows however many ghosts the mesh needed.
        ngst, ncell = self.svr.ngstcell, self.svr.ncell
        self.assertEqual(ngst, self.svr.so0n.nghost)
        rows = self.field.conserved()
        self.assertEqual(rows.shape[0], ncell)
        np.testing.assert_array_equal(
            rows, self.svr.so0n.ndarray[ngst:ngst + ncell])

    def test_views_alias_the_solver_memory(self):
        self._seed('so0n', (2.0, 3.0, 4.0))
        np.testing.assert_array_equal(self.field.density(), (2.0, 3.0, 4.0))

    def test_geometry_spans_the_body_cells(self):
        ncell = self.svr.ncell
        self.assertEqual(self.field.centroid().shape, (ncell, self.svr.ndim))
        volume = self.field.volume()
        self.assertEqual(volume.shape[0], ncell)
        assert_almost_equal(float(volume.sum()), self.AREA)

    def test_total_mass_integrates_the_density(self):
        # init_solution seeds a uniform density of one, so the mass is the
        # domain area.
        assert_almost_equal(self.field.total_mass(), self.AREA)
        self._seed('so0n', (2.0, 2.0, 2.0))
        assert_almost_equal(self.field.total_mass(), 2.0 * self.AREA)

    def test_residual_is_the_rms_change_over_the_half_step(self):
        self._seed('so0c', (1.0, 1.0, 1.0))
        self._seed('so0n', (1.1, 0.9, 1.0))
        expected = math.sqrt((0.1 ** 2 + 0.1 ** 2) / 3.0) / (self.DT / 2.0)
        assert_almost_equal(self.field.residual(), expected)

    def test_residual_vanishes_on_a_frozen_solution(self):
        self._seed('so0c', (1.0, 2.0, 3.0))
        self._seed('so0n', (1.0, 2.0, 3.0))
        self.assertEqual(self.field.residual(), 0.0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
