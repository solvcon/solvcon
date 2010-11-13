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

#include "cese.h"

int calc_norm_diff(exedata *exd, int istart, int iend, double *diff) {
    // pointers.
    double *psol, *psoln, *pdiff;
    // interators.
    int icl, ieq;
    pdiff = diff + istart*NEQ;
    psol = exd->sol + istart*NEQ;
    psoln = exd->soln + istart*NEQ;
    for (icl=istart; icl<iend; icl++) {
        for (ieq=0; ieq<NEQ; ieq++) {
            pdiff[ieq] = fabs(psoln[ieq] - psol[ieq]);
        };
        // advance pointers.
        pdiff += NEQ;
        psol += NEQ;
        psoln += NEQ;
    };
};
double calc_norm_L1(exedata *exd, int istart, int iend,
        double *diff, int teq) {
    // pointers.
    double *pclvol, *pdiff;
    // scalars.
    double smd;
    // interators.
    int icl, ieq;
    smd = 0.0;
    pdiff = diff + istart*NEQ;
    pclvol = exd->clvol + istart;
    for (icl=istart; icl<iend; icl++) {
        smd += pdiff[teq] * pclvol[0];
        // advance pointers.
        pdiff += NEQ;
        pclvol += 1;
    };
    return smd;
};
double calc_norm_L2(exedata *exd, int istart, int iend,
        double *diff, int teq) {
    // pointers.
    double *pclvol, *pdiff;
    // scalars.
    double smd;
    // interators.
    int icl, ieq;
    smd = 0.0;
    pdiff = diff + istart*NEQ;
    pclvol = exd->clvol + istart;
    for (icl=istart; icl<iend; icl++) {
        smd += pdiff[teq]*pdiff[teq] * pclvol[0];
        // advance pointers.
        pdiff += NEQ;
        pclvol += 1;
    };
    return smd;
};
// vim: set ts=4 et:
