# -*- coding: UTF-8 -*-
#
# Copyright (C) 2012 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import unittest

import numpy as np

from solvcon import testing

from . import solver as fake_solver


class TestFakeSolver(unittest.TestCase):
    def setUp(self):
        blk = testing.create_trivial_2d_blk()
        self.svr = fake_solver.FakeSolver(blk, neq=1)
        self.svr.sol.fill(0)
        self.svr.soln.fill(0)
        self.svr.dsol.fill(0)
        self.svr.dsoln.fill(0)

    def test_calcsoln(self):
        svr = self.svr
        # run the solver.
        _ = svr.march(0.0, 0.01, 100)
        # calculate and compare the results in soln (discard ghost cells).
        soln = svr.soln[svr.blk.ngstcell:,:]
        clvol = np.empty_like(soln)
        clvol.fill(0)
        for iistep in range(200):
            clvol[:,0] += svr.blk.clvol*svr.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())

    def test_calcdsoln(self):
        svr = self.svr
        # run the solver.
        _ = svr.march(0.0, 0.01, 100)
        # calculate and compare the results in dsoln (discard ghost cells).
        dsoln = svr.dsoln[svr.blk.ngstcell:,0,:]
        clcnd = np.empty_like(dsoln)
        clcnd.fill(0)
        for iistep in range(200):
            clcnd += svr.blk.clcnd*svr.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

    def placeholder(self):
        pass

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
