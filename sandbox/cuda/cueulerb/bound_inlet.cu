/*
 * Copyright (C) 2010 Yung-Yu Chen <yyc@solvcon.net>.
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

#include "euler.h"

int bound_inlet_soln(exedata *exd, int nbnd, int *facn,
        int nvalue, double *value) {
    // pointers.
    int *pfacn, *pfccls;
    double *pvalue, *pjsoln;
    // scalars.
    double rho, p, ga, ke;
    double v1, v2, v3;
    // iterators.
    int ibnd, ifc, icl, jcl, ieq;
    pfacn = facn;
    pvalue = value;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        // extract parameters.
        rho = pvalue[0];
        v1 = pvalue[1];
        v2 = pvalue[2];
        v3 = pvalue[3];
        ke = (v1*v1 + v2*v2
#if NDIM == 3
            + v3*v3
#endif
        )*rho/2.0;
        p = pvalue[4];
        ga = pvalue[5];
        // set solutions.
        pjsoln = exd->soln + jcl*NEQ;
        pjsoln[0] = rho;
        pjsoln[1] = v1*rho;
        pjsoln[2] = v2*rho;
#if NDIM == 3
        pjsoln[3] = v3*rho;
#endif
        pjsoln[1+NDIM] = p/(ga-1.0) + ke;
        // advance boundary face.
        pfacn += BFREL;
        pvalue += nvalue;
    };
    return 0;
}
int bound_inlet_dsoln(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls;
    double *pjdsoln;
    // iterators.
    int ibnd, ifc, icl, jcl;
    int it;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pjdsoln = exd->dsoln + jcl*NEQ*NDIM;
        // set to zero.
        for (it=0; it<NEQ*NDIM; it++) {
            pjdsoln[it] = 0.0;
        };
        // advance boundary face.
        pfacn += BFREL;
    };
    return 0;
}
// vim: set ts=4 et:
