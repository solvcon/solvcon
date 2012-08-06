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
    ctypedef struct sc_mesh_t:
        int ndim, nnode, nface, ncell, nbound, ngstnode, ngstface, ngstcell
        # geometry.
        double *ndcrd, *fccnd, *fcnml, *fcara, *clcnd, *clvol
        # meta.
        int *fctpn, *cltpn, *clgrp
        # connectivity.
        int *fcnds, *fccls, *clnds, *clfcs

    cdef enum sc_mesh_shape_enum:
        FCMND = 4
        CLMND = 8
        CLMFC = 6
        FCREL = 4
        BFREL = 3

cdef class Mesh:
    cdef sc_mesh_t *_mesh

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
