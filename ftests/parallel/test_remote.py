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

import os
from unittest import TestCase
from solvcon.anchor import VtkAnchor

from solvcon.solver import BlockSolver
from solvcon.hook import BlockHook

class CaseCollect(BlockHook):
    def postmarch(self):
        self._collect_interior('soln', tovar=True)
        self._collect_interior('dsoln', tovar=True)
    def preloop(self):
        self.postmarch()

class TestingSolver(BlockSolver):
    MESG_FILENAME_DEFAULT = os.devnull

    _interface_init_ = ['cecnd', 'cevol']

    def __init__(self, blk, *args, **kw):
        """
        @keyword neq: number of equations (variables).
        @type neq: int
        """
        from numpy import empty
        super(TestingSolver, self).__init__(blk, *args, **kw)
        # data structure for C/FORTRAN.
        self.blk = blk
        # arrays.
        ndim = self.ndim
        ncell = self.ncell
        ngstcell = self.ngstcell
        ## solutions.
        neq = self.neq
        self.sol = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.soln = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.dsol = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        self.dsoln = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        ## metrics.
        self.cecnd = empty(
            (ngstcell+ncell, self.CLMFC+1, ndim), dtype=self.fpdtype)
        self.cevol = empty((ngstcell+ncell, self.CLMFC+1), dtype=self.fpdtype)

    def create_alg(self):
        from solvcon.parcel.fake._algorithm import FakeAlgorithm
        alg = FakeAlgorithm()
        alg.setup_mesh(self.blk)
        alg.setup_algorithm(self)
        return alg

    ##################################################
    # marching algorithm.
    ##################################################
    MMNAMES = list()
    MMNAMES.append('update')
    def update(self, worker=None):
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    MMNAMES.append('calcsoln')
    def calcsoln(self, worker=None):
        self.create_alg().calc_soln()

    MMNAMES.append('ibcsoln')
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    MMNAMES.append('calccfl')
    def calccfl(self, worker=None):
        self.marchret = -2.0

    MMNAMES.append('calcdsoln')
    def calcdsoln(self, worker=None):
        self.create_alg().calc_dsoln()

    MMNAMES.append('ibcdsoln')
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

class TestBlockCaseRun(TestCase):
    time = 0.0
    time_increment = 1.0
    nsteps = 10

    def _get_case(self, **kw):
        import os
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

class TestRemoteParallel(TestBlockCaseRun):
    npart = 2

    def test_dsoln(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from numpy import zeros
        from solvcon.batch import Localhost
        from solvcon.domain import Distributed
        localpath = os.path.abspath(os.path.dirname(__file__))
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] += ':%s' % localpath
        else:
            os.environ['PYTHONPATH'] = localpath
        case = self._get_case(npart=self.npart, domaintype=Distributed,
            batch=Localhost, rkillfn='')
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

class SampleVtkAnchor(VtkAnchor):
    def process(self, istep):
        from solvcon.visual_vtk import Vop
        self._aggregate()
        usp = Vop.c2p(self.svr.ust)
        cut = Vop.cut(usp, (0,0,0), (0,1,0))
        cut.Update()
        Vop.write_poly(cut, self.vtkfn)
class TestPresplitRemoteParallel(TestBlockCaseRun):
    npart = 3
    def test_dsoln_and_parallel_output(self):
        import sys, os
        import shutil, glob
        from tempfile import mkdtemp
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        try:
            import vtk
        except ImportError:
            raise SkipTest
        from numpy import zeros
        from solvcon.conf import env
        from solvcon.batch import Localhost
        from solvcon.domain import Distributed
        from solvcon.helper import Information
        from solvcon import case, hook, anchor
        # construct case.
        rootdir = os.path.abspath(os.path.dirname(__file__))
        cse = case.BlockCase(basedir='.', basefn='blockcase', bcmap=None,
            rootdir=rootdir, solvertype=TestingSolver, neq=1,
            steps_run=self.nsteps, time_increment=self.time_increment,
            npart=self.npart, domaintype=Distributed,
            batch=Localhost, rkillfn='',
            meshfn=os.path.join(env.datadir, 'sample.dom'))
        cse.info = Information()
        cse.runhooks.append(anchor.FillAnchor,
            keys=('soln', 'dsoln'), value=0.0,
        )
        cse.runhooks.append(CaseCollect)
        tdir = mkdtemp()
        cse.runhooks.append(hook.PMarchSave, anames=[
            ('soln', False, -1),
        ], fpdtype='float32', psteps=1, compressor='gz', altdir=tdir)
        cse.runhooks.append(hook.PVtkHook, anames=[
            ('soln', False, -1),
        ], ankcls=SampleVtkAnchor, psteps=1, altdir=tdir)
        cse.init()
        cse.run()
        # verify output files.
        self.assertEqual(len(glob.glob(os.path.join(tdir, '*.pvtu'))),
            self.nsteps+1)
        self.assertEqual(len(glob.glob(os.path.join(tdir, '*.pvtp'))),
            self.nsteps+1)
        self.assertEqual(len(glob.glob(os.path.join(tdir, '*.vtu'))),
            (self.nsteps+1)*self.npart)
        self.assertEqual(len(glob.glob(os.path.join(tdir, '*.vtp'))),
            (self.nsteps+1)*self.npart)
        shutil.rmtree(tdir)
        # get result.
        dsoln = cse.execution.var['dsoln'][:,0,:]
        # calculate reference
        blk = cse.solver.domainobj.blk
        clcnd = zeros(dsoln.shape, dtype=dsoln.dtype)
        for iistep in range(self.nsteps*2):
            clcnd += blk.clcnd*self.time_increment/2
        # compare.
        self.assertTrue((dsoln==clcnd).all())

class TestTorqueParallel(TestBlockCaseRun):
    npart = 2

    def test_runparallel(self):
        import sys, os
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from solvcon.batch_torque import TaskManager
        if TaskManager._clib_torque == None or \
           'PBS_NODEFILE' not in os.environ: raise SkipTest
        from numpy import zeros
        from solvcon.batch import Torque
        from solvcon.domain import Distributed
        case = self._get_case(npart=self.npart, domaintype=Distributed,
            batch=Torque, rkillfn='')
        case.run()
