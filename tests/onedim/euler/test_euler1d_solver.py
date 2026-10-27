# Copyright (c) 2018, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

from solvcon.onedim import euler


class Euler1DSolverTC(unittest.TestCase):

    @staticmethod
    def _build_solver(resolution):
        time_stop = 2 * np.pi
        cfl_max = 1.0
        xmin = 0.0
        xmax = 2 * np.pi
        dx = xmax / resolution * 2
        dt_max = dx * cfl_max
        nstep = int(np.ceil(time_stop / dt_max))
        dt = time_stop / nstep

        svr = euler.Euler1DSolver(
            xmin=xmin, xmax=xmax, ncoord=resolution + 1,
            time_increment=dt)
        svr.setup_march()

        return nstep, svr

    def setUp(self):
        self.resolution = 8
        self.nstep, self.svr = self._build_solver(self.resolution)
        self.xcrd = self.svr.coord.copy()
        self.cycle = 10

    def test_coord(self):
        self.assertEqual(self.resolution + 1, self.svr.ncoord)
        self.assertEqual(self.svr.coord.tolist(), self.xcrd.tolist())

    def test_nvar(self):
        self.assertEqual(3, self.svr.nvar)

    def test_array_getter(self):
        ncoord = self.svr.ncoord
        nvar = self.svr.nvar

        self.assertEqual((ncoord,), self.svr.coord.shape)
        self.assertEqual((ncoord,), self.svr.cfl.shape)
        self.assertEqual((ncoord, nvar), self.svr.so0.shape)
        self.assertEqual((ncoord, nvar), self.svr.so1.shape)

    def test_march_fine_interface(self):
        def _march():
            # first half step.
            self.svr.march_half_so0(odd_plane=False)
            self.svr.treat_boundary_so0()
            self.svr.update_cfl(odd_plane=True)
            self.svr.march_half_so1_alpha2(odd_plane=False)
            self.svr.treat_boundary_so1()
            # second half step.
            self.svr.march_half_so0(odd_plane=True)
            self.svr.update_cfl(odd_plane=False)
            self.svr.march_half_so1_alpha2(odd_plane=True)

        svr2 = self._build_solver(self.resolution)[-1]

        for it in range(self.nstep * self.cycle):
            _march()
            svr2.march_alpha2(steps=1)
            self.assertEqual(self.svr.so0.tolist(), svr2.so0.tolist())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
