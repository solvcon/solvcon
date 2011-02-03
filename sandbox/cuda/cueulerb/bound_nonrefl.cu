/*
 * Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
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

#include "cueuler.h"

int bound_nonrefl_soln(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls;
    double *pisol, *pisoln, *pjsoln;
    // iterators.
    int ibnd, ifc, icl, jcl, ieq;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        // set solutions.
        pisol = exd->sol + icl*NEQ;
        pisoln = exd->soln + icl*NEQ;
        pjsoln = exd->soln + jcl*NEQ;
        for (ieq=0; ieq<NEQ; ieq++) {
            pjsoln[ieq] = pisoln[ieq] + exd->taylor*(pisol[ieq] - pisoln[ieq]);
        };
        // advance boundary face.
        pfacn += BFREL;
    };
    return 0;
}
int bound_nonrefl_dsoln(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pidsol, *pidsoln, *pjdsoln, *pdsol, *pdsoln;
    double *pndcrd, *pfccnd, *pfcnml;
    // scalars.
    double len;
    // arrays.
    double dif[NDIM];
    double vec[NEQ][NDIM];
    double mat[NDIM][NDIM], matinv[NDIM][NDIM];
    // iterators.
    int ibnd, ifc, icl, jcl, ieq;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        pfcnds = exd->fcnds + ifc*(FCMND+1);
        icl = pfccls[0];
        jcl = pfccls[1];
        pidsol = exd->dsol + icl*NEQ*NDIM;
        pidsoln = exd->dsoln + icl*NEQ*NDIM;
        pjdsoln = exd->dsoln + jcl*NEQ*NDIM;
        // coordinate transformation and set transformed vectors.
        pfcnml = exd->fcnml + ifc*NDIM;
        pfccnd = exd->fccnd + ifc*NDIM;
        mat[0][0] = matinv[0][0] = pfcnml[0];
        mat[0][1] = matinv[1][0] = pfcnml[1];
#if NDIM == 3
        mat[0][2] = matinv[2][0] = pfcnml[2];
        pndcrd = exd->ndcrd + pfcnds[1]*NDIM;
        mat[1][0] = pndcrd[0] - pfccnd[0];
        mat[1][1] = pndcrd[1] - pfccnd[1];
        mat[1][2] = pndcrd[2] - pfccnd[2];
        len = sqrt(mat[1][0]*mat[1][0] + mat[1][1]*mat[1][1]
                 + mat[1][2]*mat[1][2]);
        mat[1][0] = matinv[0][1] = mat[1][0]/len;
        mat[1][1] = matinv[1][1] = mat[1][1]/len;
        mat[1][2] = matinv[2][1] = mat[1][2]/len;
        mat[2][0] = matinv[0][2] = mat[0][1]*mat[1][2] - mat[0][2]*mat[1][1];
        mat[2][1] = matinv[1][2] = mat[0][2]*mat[1][0] - mat[0][0]*mat[1][2];
        mat[2][2] = matinv[2][2] = mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0];
#else
        mat[1][0] = matinv[0][1] =  pfcnml[1];
        mat[1][1] = matinv[1][1] = -pfcnml[0];
#endif
        pdsol = pidsol; pdsoln = pidsoln;
        for (ieq=0; ieq<NEQ; ieq++) {
            vec[ieq][0] = 0.0;  // set perpendicular gradient to zero.
            dif[0] = pdsoln[0] + exd->taylor*(pdsol[0] - pdsoln[0]);
            dif[1] = pdsoln[1] + exd->taylor*(pdsol[1] - pdsoln[1]);
#if NDIM == 3
            dif[2] = pdsoln[2] + exd->taylor*(pdsol[2] - pdsoln[2]);
            vec[ieq][1] = mat[1][0]*dif[0] + mat[1][1]*dif[1]
                        + mat[1][2]*dif[2];
            vec[ieq][2] = mat[2][0]*dif[0] + mat[2][1]*dif[1]
                        + mat[2][2]*dif[2];
#else
            vec[ieq][1] = mat[1][0]*dif[0] + mat[1][1]*dif[1];
#endif
            pdsol += NDIM; pdsoln += NDIM;
        };
        // inversely transform the coordinate and set ghost gradient.
        pdsoln = pjdsoln;
        for (ieq=0; ieq<NEQ; ieq++) {
#if NDIM == 3
            pdsoln[0] = matinv[0][0]*vec[ieq][0] + matinv[0][1]*vec[ieq][1]
                      + matinv[0][2]*vec[ieq][2];
            pdsoln[1] = matinv[1][0]*vec[ieq][0] + matinv[1][1]*vec[ieq][1]
                      + matinv[1][2]*vec[ieq][2];
            pdsoln[2] = matinv[2][0]*vec[ieq][0] + matinv[2][1]*vec[ieq][1]
                      + matinv[2][2]*vec[ieq][2];
#else
            pdsoln[0] = matinv[0][0]*vec[ieq][0] + matinv[0][1]*vec[ieq][1];
            pdsoln[1] = matinv[1][0]*vec[ieq][0] + matinv[1][1]*vec[ieq][1];
#endif
            pdsoln += NDIM;
        };
        // advance boundary face.
        pfacn += BFREL;
    };
    return 0;
}
// vim: set ts=4 et:

