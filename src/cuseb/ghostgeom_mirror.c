/*
 * Copyright (C) 2008-2010 Yung-Yu Chen <yyc@solvcon.net>.
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


#include "cuse.h"

int ghostgeom_mirror(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls;
    double *pfccnd, *pfcnml;
    double *picecnd, *pjcecnd;
    // scalars.
    double len;
	// iterators.
	int ibnd, ifc, icl, jcl;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        picecnd = exd->cecnd + icl*(CLMFC+1)*NDIM;
        pjcecnd = exd->cecnd + jcl*(CLMFC+1)*NDIM;
        // calculate displacement.
        pfccnd = exd->fccnd + ifc * NDIM;
        pfcnml = exd->fcnml + ifc * NDIM;
        len = (pfccnd[0] - picecnd[0]) * pfcnml[0]
            + (pfccnd[1] - picecnd[1]) * pfcnml[1]
#if NDIM == 3
            + (pfccnd[2] - picecnd[2]) * pfcnml[2]
#endif
            ;
        len *= 2.0;
        // set ghost solution point.
        pjcecnd[0] = picecnd[0] + pfcnml[0] * len;
        pjcecnd[1] = picecnd[1] + pfcnml[1] * len;
#if NDIM == 3
        pjcecnd[2] = picecnd[2] + pfcnml[2] * len;
#endif
        // advance boundary face.
        pfacn += 3;
    };
    return 0;
}
// vim: set ts=4 et:
