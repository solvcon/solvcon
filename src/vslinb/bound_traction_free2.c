/*
 * Copyright (C) 2010-2011 Yung-Yu Chen <yyc@solvcon.net>.
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

#include "vslin.h"

int bound_traction_free2_soln(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd, *pvalue;
    double *pisol, *pjsoln;
    // scalars.
    double tdep;
    // arrays.
    double nvec[3], svec[3];
    double rotm[3][3], bondm[6][6], bondminv[6][6];
    double sts[6], trc[3];
    // iterators.
    int ibnd, ifc, icl, jcl, ieq;
    int it, jt;
    nvec[2] = 0.0;
    svec[2] = 0.0;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc * FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        // set through velocity.
        pisol = exd->sol + icl * NEQ;
        pjsoln = exd->soln + jcl * NEQ;
        pjsoln[0] = pisol[0];
        pjsoln[1] = pisol[1];
        pjsoln[2] = pisol[2];
        // XXX: ad hoc code.
        pjsoln[3] =  pisol[3]; // T11
        pjsoln[4] = -pisol[4]; // T22
        pjsoln[5] =  pisol[5]; // T33
        pjsoln[6] = -pisol[6]; // T23
        pjsoln[7] =  pisol[7]; // T13
        pjsoln[8] = -pisol[8]; // T12
        // advance boundary face.
        pfacn += 3;
    };
    return 0;
}
int bound_traction_free2_dsoln(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd;
    double *pidsol, *pjdsoln;
    // arrays.
    double nvec[3], svec[3];
    double rotm[3][3], bondm[6][6], bondminv[6][6];
    double dvel0[3][3], dvel1[3][3], dsts0[6][3], dsts1[6][3];
    // iterators.
    int ibnd, ifc, icl, jcl, ieq;
    int ii, ij, ik;
    nvec[2] = 0.0;
    svec[2] = 0.0;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc * FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pidsol = exd->dsol + icl * NEQ * NDIM;
        pjdsoln = exd->dsoln + jcl * NEQ * NDIM;
        // XXX: ad hoc code.
        for (ieq=0; ieq<3; ieq++) {
            pjdsoln[0] =  pidsol[0]; // unchanged.
            pjdsoln[1] = -pidsol[1]; // unchanged.
            //pjdsoln[1] = 0; // unchanged.
            pidsol += NDIM;
            pjdsoln += NDIM;
        };
        // T11 unchanged.
        pjdsoln[0] =  pidsol[0];
        pjdsoln[1] = -pidsol[1];
        pidsol += NDIM;
        pjdsoln += NDIM;
        // T22 vanishing.
        pjdsoln[0] = -pidsol[0];
        pjdsoln[1] =  pidsol[1];
        pidsol += NDIM;
        pjdsoln += NDIM;
        // T33 unchanged.
        pjdsoln[0] =  pidsol[0];
        pjdsoln[1] = -pidsol[1];
        pidsol += NDIM;
        pjdsoln += NDIM;
        // T23 vanishing.
        pjdsoln[0] = -pidsol[0];
        pjdsoln[1] =  pidsol[1];
        pidsol += NDIM;
        pjdsoln += NDIM;
        // T13 unchanged.
        pjdsoln[0] =  pidsol[0];
        pjdsoln[1] = -pidsol[1];
        pidsol += NDIM;
        pjdsoln += NDIM;
        // T12 vanishing.
        pjdsoln[0] = -pidsol[0];
        pjdsoln[1] =  pidsol[1];
        pidsol += NDIM;
        pjdsoln += NDIM;
        // advance boundary face.
        pfacn += 3;
    };
    return 0;
}
// vim: set ts=4 et:
