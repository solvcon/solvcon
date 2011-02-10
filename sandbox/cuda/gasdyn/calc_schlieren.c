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

#include "gasdyn.h"

int calc_schlieren_rhog(exedata *exd, int istart, int iend,
        double *rhog) {
    int cputicks;
    struct tms timm0, timm1;
    // pointers.
    double *pdsoln;
    double *prhog;
    // iterators.
    int icl;
    times(&timm0);
#ifdef SOLVCESE_FE
    feenableexcept(SOLVCESE_FE);
#endif
    pdsoln = exd->dsoln + istart*NEQ*NDIM;
    prhog = rhog + istart+exd->ngstcell;
    for (icl=istart; icl<iend; icl++) {
        // density gradient.
        prhog[0] = pdsoln[0]*pdsoln[0] + pdsoln[1]*pdsoln[1];
#if NDIM == 3
        prhog[0] += pdsoln[2]*pdsoln[2];
#endif
        prhog[0] = sqrt(prhog[0]);
        // advance pointers.
        pdsoln += NEQ*NDIM;
        prhog += 1;
    };
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
int calc_schlieren_sch(exedata *exd, int istart, int iend,
        double k, double k0, double k1, double rhogmax, double *sch) {
    int cputicks;
    struct tms timm0, timm1;
    // pointers.
    double *psch;
    // scalars.
    double fac0, fac1;
    // iterators.
    int icl;
    times(&timm0);
#ifdef SOLVCESE_FE
    feenableexcept(SOLVCESE_FE);
#endif
    fac0 = k0 * rhogmax;
    fac1 = -k / ((k1-k0) * rhogmax + SOLVCESE_ALMOST_ZERO);
    psch = sch + istart+exd->ngstcell;
    for (icl=istart; icl<iend; icl++) {
        // density gradient.
        psch[0] = exp((psch[0]-fac0)*fac1);
        // advance pointers.
        psch += 1;
    };
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
// vim: set ts=4 et:
