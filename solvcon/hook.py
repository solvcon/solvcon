# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Hooks for :py:class:`solvcon.case.MeshCase`.
"""


import numpy as np

from . import rpc
from . import domain
from . import anchor

# import legacy.
from .hook_legacy import (
    Hook, ProgressHook, BlockHook, BlockInfoHook, CollectHook,
    SplitMarker, GroupMarker, VtkSave, SplitSave, MarchSave, PMarchSave,
    PVtkHook)


class MeshHook(object):
    """
    Base type for hooks needing a :py:class:`MeshCase <solvcon.case.MeshCase>`.
    """

    def __init__(self, cse, **kw):
        from . import case # work around cyclic importation.
        assert isinstance(cse, case.MeshCase)
        self.cse = cse
        self.info = cse.info
        self.psteps = kw.pop('psteps', None)
        self.ankcls = kw.pop('ankcls', None)
        # save excessive keywords.
        self.kws = dict(kw)
        super(MeshHook, self).__init__()

    def _makedir(self, dirname, verbose=False):
        """
        Make new directory if it does not exist in prior.

        @param dirname: name of directory to be created.
        @type dirname: str
        @keyword verbose: flag if print out creation message.
        @type verbose: bool
        """
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            if verbose:
                self.info('Created %s' % dirname)

    @staticmethod
    def _deliver_anchor(target, ankcls, ankkw):
        """
        Provide the information to instantiate anchor object for a solver.  The
        target object can be a real solver object or a shadow associated to a
        remote worker object with attached muscle of solver object.

        @param target: the solver or shadow object.
        @type target: solvcon.solver.Solver or solvcon.rpc.Shadow
        @param ankcls: type of the anchor to instantiate.
        @type ankcls: type
        @param ankkw: keywords to instantiate anchor object.
        @type ankkw: dict
        @return: nothing
        """
        if isinstance(target, rpc.Shadow):
            target.drop_anchor(ankcls, ankkw)
        else:
            target.runanchors.append(ankcls, **ankkw)

    def drop_anchor(self, svr):
        """
        Drop the anchor(s) to the solver object.

        @param svr: the solver object on which the anchor(s) is dropped.
        @type svr: solvon.solver.BaseSolver
        @return: nothing
        """
        if self.ankcls:
            self._deliver_anchor(svr, self.ankcls, self.kws)

    def preloop(self):
        """
        Things to do before the time-marching loop.
        """
        pass

    def premarch(self):
        """
        Things to do before the time march for a specific time step.
        """
        pass

    def postmarch(self):
        """
        Things to do after the time march for a specific time step.
        """
        pass

    def postloop(self):
        """
        Things to do after the time-marching loop.
        """
        pass

    @property
    def blk(self):
        return self.cse.solver.domainobj.blk

    def _collect_interior(self, key, tovar=False, inder=False,
        consider_ghost=True):
        """
        @param key: the name of the array to collect in a solver object.
        @type key: str
        @keyword tovar: flag to store collect data to case var dict.
        @type tovar: bool
        @keyword inder: the array is for derived data.
        @type inder: bool
        @keyword consider_ghost: treat the array with the consideration of
            ghost cells.  Default is True.
        @type consider_ghost: bool
        @return: the interior array hold by the solver.
        @rtype: numpy.ndarray
        """
        cse = self.cse
        ncell = self.blk.ncell
        ngstcell = self.blk.ngstcell
        if cse.is_parallel:
            dom = self.cse.solver.domainobj
            # collect arrays from solvers.
            dealer = self.cse.solver.dealer
            arrs = list()
            for iblk in range(dom.nblk):
                dealer[iblk].cmd.pull(key, inder=inder, with_worker=True)
                arr = dealer[iblk].recv()
                arrs.append(arr)
            # create global array.
            shape = [it for it in arrs[0].shape]
            shape[0] = ncell
            arrg = np.empty(shape, dtype=arrs[0].dtype)
            # set global array.
            clmaps = dom.mappers[2]
            for iblk in range(dom.nblk):
                slctg = (clmaps[:,1] == iblk)
                slctl = clmaps[slctg,0]
                if consider_ghost:
                    slctl += dom.shapes[iblk,6]
                arrg[slctg] = arrs[iblk][slctl]
        else:
            if consider_ghost:
                start = ngstcell
            else:
                start = 0
            if inder:
                arrg = cse.solver.solverobj.der[key][start:].copy()
            else:
                arrg = getattr(cse.solver.solverobj, key)[start:].copy()
        if tovar:
            self.cse.execution.var[key] = arrg
        return arrg

    def _spread_interior(self, arrg, key, consider_ghost=True):
        """
        @param arrg: the global array to be spreaded.
        @type arrg: numpy.ndarray
        @param key: the name of the array to collect in a solver object.
        @type key: str
        @keyword consider_ghost: treat the arrays with the consideration of
            ghost cells.  Default is True.
        @type consider_ghost: bool
        @return: the interior array hold by the solver.
        @rtype: numpy.ndarray
        """
        cse = self.cse
        ncell = self.blk.ncell
        ngstcell = self.blk.ngstcell
        if cse.is_parallel:
            dom = self.cse.solver.domainobj
            dealer = self.cse.solver.dealer
            clmaps = dom.mappers[2]
            for iblk in range(len(dom)):
                blk = dom[iblk]
                # create subarray.
                shape = [it for it in arrg.shape]
                if consider_ghost:
                    shape[0] = blk.ngstcell+blk.ncell
                else:
                    shape[0] = blk.ncell
                arr = np.empty(shape, dtype=arrg.dtype)
                # calculate selectors.
                slctg = (clmaps[:,1] == iblk)
                slctl = clmaps[slctg,0]
                if consider_ghost:
                    slctl += blk.ngstcell
                # push data to remote solver.
                arr[slctl] = arrg[slctg]
                dealer[iblk].cmd.push(arr, key, start=blk.ngstcell)
        else:
            if consider_ghost:
                start = ngstcell
            else:
                start = 0
            getattr(cse.solver.solverobj, key)[start:] = arrg[:]
