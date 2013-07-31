# -*- coding: UTF-8 -*-
#
# Copyright (c) 2012, Yung-Yu Chen <yyc@solvcon.net>
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
The Python API for the fake solver.
"""


import warnings

import numpy as np

from solvcon import solver

try: # for readthedocs to work.
    from . import _algorithm
except ImportError:
    warnings.warn("solvcon.parcel.fake._algorithm isn't built",
                  RuntimeWarning)


class FakeSolver(solver.MeshSolver):
    """
    This class represents the Python side of a demonstration-only numerical
    method.  It instantiates a :py:class:`FakeAlgorithm
    <solvcon.parcel.fake._algorithm.FakeAlgorithm>` object.
    Computation-intensive tasks are delegated to the algorithm object.
    """

    def __init__(self, blk, neq=None, **kw):
        """Constructor of :py:class:`FakeSolver`.

        A :py:class:`Block <solvcon.block.Block>` is a prerequisite:

        >>> from solvcon.testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()

        But the constructor also needs to know how many equations (i.e., number
        of variables per cell).  We must provide the *neq* argument:

        >>> _ = FakeSolver(blk)
        Traceback (most recent call last):
            ...
        ValueError: neq must be int (but got None)
        >>> _ = FakeSolver(blk, neq=1.2)
        Traceback (most recent call last):
            ...
        ValueError: neq must be int (but got 1.2)
        >>> svr = FakeSolver(blk, neq=1)

        Each time step is composed by two sub time steps, as the CESE method
        requires:

        >>> svr.substep_run
        2

        The constructor will create four solution arrays (without
        initialization):

        >>> [(type(getattr(svr, key)), getattr(svr, key).shape)
        ...     for key in ('sol', 'soln', 'dsol', 'dsoln')]
        ...     # doctest: +NORMALIZE_WHITESPACE
        [(<type 'numpy.ndarray'>, (6, 1)), (<type 'numpy.ndarray'>, (6, 1)),
        (<type 'numpy.ndarray'>, (6, 1, 2)), (<type 'numpy.ndarray'>, (6, 1,
        2))]
        """
        # meta data.
        if not isinstance(neq, int):
            raise ValueError('neq must be int (but got %s)' % str(neq))
        #: Number of equations, or number of variables on each cell.  Must be
        #: an :py:class:`int`.
        self.neq = neq
        super(FakeSolver, self).__init__(blk, **kw)
        self.substep_run = 2
        # arrays.
        ndim = blk.ndim
        ncell = blk.ncell
        ngstcell = blk.ngstcell
        fpdtype = 'float64'
        neq = self.neq
        #: This is the "present" solution array (:py:class:`numpy.ndarray`) for
        #: the algorithm with two sub time step.
        self.sol = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        #: This is the "next" or "new" solution array
        #: (:py:class:`numpy.ndarray`) for the algorithm with two sub time
        #: step.
        self.soln = np.empty((ngstcell+ncell, neq), dtype=fpdtype)
        #: This is the "present" solution gradient array
        #: (:py:class:`numpy.ndarray`) for the algorithm with two sub time
        #: step.
        self.dsol = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)
        #: This is the "next" or "new" solution gradient array
        #: (:py:class:`numpy.ndarray`) for the algorithm with two sub time
        #: step.
        self.dsoln = np.empty((ngstcell+ncell, neq, ndim), dtype=fpdtype)

    def create_alg(self):
        """
        Create a :py:class:`FakeAlgorithm <._algorithm.FakeAlgorithm>`
        object.

        >>> from solvcon.testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> svr = FakeSolver(blk, neq=1)
        >>> isinstance(svr.create_alg(), _algorithm.FakeAlgorithm)
        True
        """
        alg = _algorithm.FakeAlgorithm()
        alg.setup_mesh(self.blk)
        alg.setup_algorithm(self)
        return alg

    ###########################################################################
    # begin marching algorithm.
    #: See :py:attr:`solvcon.solver.MeshSolver._MMNAMES`.
    _MMNAMES = solver.MeshSolver.new_method_list()

    @_MMNAMES.register
    def update(self, worker=None):
        """
        Update the present solution arrays (:py:attr:`sol` and :py:attr:`dsol`)
        with the contents in the next solution arrays (:py:attr:`dsol` and
        :py:attr:`dsoln`).
 
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> svr = FakeSolver(blk, neq=1)
        >>> # initialize with different solution arrays.
        >>> svr.sol.fill(0)
        >>> svr.soln.fill(2)
        >>> svr.dsol.fill(0)
        >>> svr.dsoln.fill(2)
        >>> (svr.sol != svr.soln).all()
        True
        >>> (svr.dsol != svr.dsoln).all()
        True
        >>> # update and then solution arrays become the same.
        >>> svr.update()
        >>> (svr.sol == svr.soln).all()
        True
        >>> (svr.dsol == svr.dsoln).all()
        True
        """
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    @_MMNAMES.register
    def calcsoln(self, worker=None):
        """
        Advance :py:attr:`sol` to :py:attr:`soln`.  The calculation is
        delegated to :py:meth:`FakeAlgorithm.calc_soln
        <solvcon.parcel.fake._algorithm.FakeAlgorithm.calc_soln>`.

        >>> # build a block before creating a solver.
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> # create a solver.
        >>> svr = FakeSolver(blk, neq=1)
        >>> # initialize the solver.
        >>> svr.sol.fill(0)
        >>> svr.soln.fill(0)
        >>> svr.dsol.fill(0)
        >>> svr.dsoln.fill(0)
        >>> # run the solver.
        >>> ret = svr.march(0.0, 0.01, 100)
        >>> # calculate and compare the results in soln.
        >>> soln = svr.soln[svr.blk.ngstcell:,:]
        >>> clvol = np.empty_like(soln)
        >>> clvol.fill(0)
        >>> for iistep in range(200):
        ...     clvol[:,0] += svr.blk.clvol*svr.time_increment/2
        >>> # compare.
        >>> (soln==clvol).all()
        True
        """
        self.create_alg().calc_soln()

    @_MMNAMES.register
    def ibcsoln(self, worker=None):
        """
        Interchange BC for the :py:attr:`soln` array.  Only used for parallel
        computing.
        """
        if worker: self.exchangeibc('soln', worker=worker)

    @_MMNAMES.register
    def calccfl(self, worker=None):
        """
        Calculate the CFL number.  For :py:class:`FakeSolver`, this method does
        nothing.
        """
        self.marchret = -2.0

    @_MMNAMES.register
    def calcdsoln(self, worker=None):
        """
        Advance :py:attr:`dsol` to :py:attr:`dsoln`.  The calculation is
        delegated to :py:meth:`FakeAlgorithm.calc_dsoln
        <solvcon.parcel.fake._algorithm.FakeAlgorithm.calc_dsoln>`.

        >>> # build a block before creating a solver.
        >>> from solvcon.testing import create_trivial_2d_blk
        >>> blk = create_trivial_2d_blk()
        >>> # create a solver.
        >>> svr = FakeSolver(blk, neq=1)
        >>> # initialize the solver.
        >>> svr.sol.fill(0)
        >>> svr.soln.fill(0)
        >>> svr.dsol.fill(0)
        >>> svr.dsoln.fill(0)
        >>> # run the solver.
        >>> ret = svr.march(0.0, 0.01, 100)
        >>> # calculate and compare the results in dsoln.
        >>> dsoln = svr.dsoln[svr.blk.ngstcell:,0,:]
        >>> clcnd = np.empty_like(dsoln)
        >>> clcnd.fill(0)
        >>> for iistep in range(200):
        ...     clcnd += svr.blk.clcnd*svr.time_increment/2
        >>> # compare.
        >>> (dsoln==clcnd).all()
        True
        """
        self.create_alg().calc_dsoln()

    @_MMNAMES.register
    def ibcdsoln(self, worker=None):
        """
        Interchange BC for the :py:attr:`dsoln` array.  Only used for parallel
        computing.
        """
        if worker: self.exchangeibc('dsoln', worker=worker)
    # end marching algorithm.
    ###########################################################################

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
