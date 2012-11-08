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
    ctypedef struct sc_lincese_algorithm_t:
        int neq
        double time, time_increment
        # group shape.
        int ngroup, gdlen
        # parameter shape.
        int nsca, nvec
        # scheme.
        int alpha
        double sigma0, taylor, cnbfac, sftfac, taumin, tauscale
        # function pointer.
        void (*jacofunc)(void *exd, int icl, double *fcn, double *jacos)
        # meta array.
        double *grpda
        # metric array.
        double *cecnd, *cevol, *sfmrc
        # solution array.
        double *amsca, *amvec, *sol, *dsol, *solt, *soln, *dsoln
        double *stm, *cfl, *ocfl

from mesh cimport Mesh
cdef class LinceseAlgorithm(Mesh):
    cdef sc_lincese_algorithm_t *_alg

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
