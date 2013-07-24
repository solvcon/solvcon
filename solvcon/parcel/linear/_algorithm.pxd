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

cdef public:
    ctypedef struct sc_linear_algorithm_t:
        # equation number.
        int neq
        # temporal information.
        double time, time_increment
        # c-tau scheme.
        int alpha
        double sigma0, taylor, cnbfac, sftfac, taumin, tauscale
        # metric array.
        double *cecnd, *cevol, *sfmrc
        # parameters.
        ## group data.
        int ngroup, gdlen
        double *grpda
        ## scalar parameters.
        int nsca
        double *amsca
        ## vector parameters.
        int nvec
        double *amvec
        # solution array.
        double *sol, *dsol, *solt, *soln, *dsoln
        double *stm, *cfl, *ocfl

from solvcon.mesh cimport Mesh
cdef class LinearAlgorithm(Mesh):
    cdef sc_linear_algorithm_t *_alg

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
