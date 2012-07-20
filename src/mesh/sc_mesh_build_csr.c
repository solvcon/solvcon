/*
 * Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "solvcon/mesh.h"

int sc_mesh_build_csr(sc_mesh *msd, int *rcells, int *adjncy) {
    // pointers.
    int *prcells, *padjncy;
    // iterators.
    int icl, ifl, ieg;

    // fill.
    prcells = rcells;
    padjncy = adjncy;
    ieg = 0;
    for (icl=0; icl<msd->ncell; icl++) {
        for (ifl=0; ifl<CLMFC; ifl++) {
            if (prcells[ifl] != -1) {
                padjncy[ieg] = prcells[ifl];
                ieg += 1;
            };
        };
        // advance pointers.
        prcells += CLMFC;
    };

    return 0;
};
// vim: set ts=4 et:

