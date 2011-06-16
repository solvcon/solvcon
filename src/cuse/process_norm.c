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

#include "cuse.h"

void process_norm_diff(exedata *exd, double *diff) {
    // pointers.
    double *psol, *psoln, *pdiff;
    // interators.
    int icl, ieq;
    #pragma omp parallel for private(psol, psoln, pdiff, icl, ieq)
    for (icl=-exd->ngstcell; icl<exd->ngstcell; icl++) {
        pdiff = diff + icl*NEQ;
        psol = exd->sol + icl*NEQ;
        psoln = exd->soln + icl*NEQ;
        for (ieq=0; ieq<NEQ; ieq++) {
            pdiff[ieq] = fabs(psoln[ieq] - psol[ieq]);
        };
    };
};
double process_norm_L1(exedata *exd, double *diff, int teq) {
    // pointers.
    double *pdiff;
    // scalars.
    double smd;
    // interators.
    int icl;
    smd = 0.0;
    #pragma omp parallel for private(pdiff, icl) reduction(+:smd)
    for (icl=0; icl<exd->ncell; icl++) {
        pdiff = diff + icl*NEQ;
        smd += pdiff[teq] * exd->clvol[icl];
    };
    return smd;
};
double process_norm_L2(exedata *exd, double *diff, int teq) {
    // pointers.
    double *pdiff;
    // scalars.
    double smd;
    // interators.
    int icl;
    smd = 0.0;
    #pragma omp parallel for private(pdiff, icl) reduction(+:smd)
    for (icl=0; icl<exd->ncell; icl++) {
        pdiff = diff + icl*NEQ;
        smd += pdiff[teq]*pdiff[teq] * exd->clvol[icl];
    };
    return smd;
};
// vim: set ts=4 et:
