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

from unittest import TestCase
from solvcon.anchor import VtkAnchor

from solvcon.hook import BlockHook
class CaseCollect(BlockHook):
    def postmarch(self):
        self._collect_interior('soln', tovar=True)
        self._collect_interior('dsoln', tovar=True)
    def preloop(self):
        self.postmarch()
del BlockHook

class TestBlockCaseRun(TestCase):
    time = 0.0
    time_increment = 1.0
    nsteps = 10

    def _get_case(self, **kw):
        import os
        from solvcon.testing import TestingSolver
        from solvcon.conf import env
        from solvcon.case import BlockCase
        from solvcon.anchor import FillAnchor
        from solvcon.helper import Information
        meshfn = kw.get('meshfn', 'sample.neu')
        kw['meshfn'] = os.path.join(env.datadir, meshfn)
        case = BlockCase(basedir='.', basefn='blockcase', bcmap=None,
            solvertype=TestingSolver, neq=1,
            steps_run=self.nsteps, time_increment=self.time_increment,
            **kw
        )
        case.info = Information()
        case.runhooks.append(FillAnchor,
            keys=('soln', 'dsoln'), value=0.0,
        )
        case.runhooks.append(CaseCollect)
        case.init()
        return case

class TestSequential(TestBlockCaseRun):
    def test_soln(self):
        from numpy import zeros
        from solvcon.domain import Domain
        case = self._get_case(domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        soln = svr.soln[ngstcell:,0]
        # calculate reference
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += svr.clvol[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())

    def test_dsoln(self):
        from numpy import zeros
        from solvcon.domain import Domain
        case = self._get_case(domaintype=Domain)
        svr = case.solver.solverobj
        case.run()
        ngstcell = svr.ngstcell
        # get result.
        dsoln = svr.dsoln[ngstcell:,0,:]
        # calculate reference
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += svr.clcnd[ngstcell:]*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

class TestLocalParallel(TestBlockCaseRun):
    npart = 3

    def test_soln(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from solvcon.domain import Collective
        case = self._get_case(npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        soln = case.execution.var['soln'][:,0]
        # calculate reference
        blk = case.solver.domainobj.blk
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += blk.clvol*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())

    def test_dsoln(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from solvcon.domain import Collective
        case = self._get_case(npart=self.npart, domaintype=Collective)
        case.run()
        # get result.
        dsoln = case.execution.var['dsoln'][:,0,:]
        # calculate reference
        blk = case.solver.domainobj.blk
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += blk.clcnd*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

    def test_ibcthread(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from solvcon.domain import Collective
        case = self._get_case(npart=self.npart, domaintype=Collective,
            ibcthread=True)
        case.run()
        # get result.
        soln = case.execution.var['soln'][:,0]
        # calculate reference
        blk = case.solver.domainobj.blk
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += blk.clvol*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())

class TestPresplitLocalParallel(TestBlockCaseRun):
    npart = 3
    def test_soln(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from solvcon.domain import Collective
        case = self._get_case(npart=self.npart, domaintype=Collective,
            meshfn='sample.dom')
        case.run()
        # get result.
        soln = case.execution.var['soln'][:,0]
        # calculate reference
        blk = case.solver.domainobj.blk
        clvol = zeros(soln.shape, dtype=soln.dtype)
        for iistep in range(self.nsteps*2):
            clvol += blk.clvol*self.time_increment/2
        # compare.
        self.assertTrue((soln==clvol).all())

class TestPresplitLocalParallelNoArrs(TestBlockCaseRun):
    npart = 3
    def _get_case_nocollect(self, **kw):
        import os
        from solvcon.testing import TestingSolver
        from solvcon.conf import env
        from solvcon.case import BlockCase
        from solvcon.anchor import FillAnchor
        from solvcon.helper import Information
        meshfn = kw.get('meshfn', 'sample.neu')
        kw['meshfn'] = os.path.join(env.datadir, meshfn)
        case = BlockCase(basedir='.', basefn='blockcase', bcmap=None,
            solvertype=TestingSolver, neq=1,
            steps_run=self.nsteps, time_increment=self.time_increment,
            **kw
        )
        case.info = Information()
        case.runhooks.append(FillAnchor,
            keys=('soln', 'dsoln'), value=0.0,
        )
        case.init()
        return case
    def test_run(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from solvcon.domain import Collective
        case = self._get_case_nocollect(npart=self.npart,
            domaintype=Collective, meshfn='sample.dom',
            with_arrs=False, with_whole=False)
        case.run()
