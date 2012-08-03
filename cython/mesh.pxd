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

cdef extern from "solvcon/mesh.h":
    int FCMND=4
    int CLMND=8
    int CLMFC=6
    int FCREL=4
    int BFREL=3

    ctypedef struct sc_mesh:
        int ndim, nnode, nface, ncell, nbound, ngstnode, ngstface, ngstcell
        # geometry.
        double *ndcrd, *fccnd, *fcnml, *fcara, *clcnd, *clvol
        # meta.
        int *fctpn, *cltpn, *clgrp
        # connectivity.
        int *fcnds, *fccls, *clnds, *clfcs

    void sc_mesh_build_ghost(sc_mesh *msd, int *bndfcs)
    int sc_mesh_calc_metric(sc_mesh *msd, int use_incenter)
    int sc_mesh_extract_faces_from_cells(sc_mesh *msd, int mface,
            int *pnface, int *clfcs, int *fctpn, int *fcnds, int *fccls)
    int sc_mesh_build_rcells(sc_mesh *msd, int *rcells, int *rcellno)
    int sc_mesh_build_csr(sc_mesh *msd, int *rcells, int *adjncy)

cdef class MeshData:
    cdef sc_mesh *_mesh

# vim: set fenc=utf8 ft=pyrex ff=unix ai et sw=4 ts=4 tw=79:
