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

from solvcon.mesh cimport sc_mesh_t, Mesh
from _algorithm cimport sc_fake_algorithm_t
import numpy as np
cimport numpy as cnp

# initialize NumPy.
cnp.import_array()

cdef extern:
    int sc_fake_calc_soln(sc_mesh_t *msd, sc_fake_algorithm_t *alg)
    int sc_fake_calc_dsoln(sc_mesh_t *msd, sc_fake_algorithm_t *alg)

cdef extern from "stdlib.h":
    void* malloc(size_t size)

cdef class FakeAlgorithm(Mesh):
    """
    An algorithm class that does trivial calculation.
    """
    def __cinit__(self):
        self._alg = <sc_fake_algorithm_t *>malloc(sizeof(sc_fake_algorithm_t))

    def setup_algorithm(self, svr):
        # meta data.
        self._alg.neq = svr.neq
        self._alg.time = svr.time
        self._alg.time_increment = svr.time_increment
        # arrays.
        cdef cnp.ndarray[double, ndim=2, mode="c"] sol = svr.sol
        self._alg.sol = &sol[self._msd.ngstcell,0]
        cdef cnp.ndarray[double, ndim=2, mode="c"] soln = svr.soln
        self._alg.soln = &soln[self._msd.ngstcell,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] dsol = svr.dsol
        self._alg.dsol = &dsol[self._msd.ngstcell,0,0]
        cdef cnp.ndarray[double, ndim=3, mode="c"] dsoln = svr.dsoln
        self._alg.dsoln = &dsoln[self._msd.ngstcell,0,0]

    def calc_soln(self):
        sc_fake_calc_soln(self._msd, self._alg)

    def calc_dsoln(self):
        sc_fake_calc_dsoln(self._msd, self._alg)

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
