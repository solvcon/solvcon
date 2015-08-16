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

from libc.stdint cimport intptr_t

cdef public:
    ctypedef struct sc_mesh_t:
        int ndim, nnode, nface, ncell, nbound, ngstnode, ngstface, ngstcell
        # geometry.
        double *ndcrd
        double *fccnd
        double *fcnml
        double *fcara
        double *clcnd
        double *clvol
        # meta.
        int *fctpn
        int *cltpn
        int *clgrp
        # connectivity.
        int *fcnds
        int *fccls
        int *clnds
        int *clfcs

    ctypedef struct sc_bound_t:
        int nbound, nvalue
        int *facn
        double *value

    cdef enum sc_mesh_shape_enum:
        FCMND = 4
        CLMND = 8
        CLMFC = 6
        FCREL = 4
        BFREL = 3

cdef class Table:
    cdef readonly intptr_t nghost
    cdef readonly char *_body
    cdef readonly object _nda

cdef class Mesh:
    cdef sc_mesh_t *_msd

cdef class Bound:
    cdef sc_bound_t *_bcd

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
