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

#include "lincuse.h"

int calc_planewave(exedata *exd, double *asol, double *adsol,
        double *amp, double *ctr, double *wvec, double afreq) {
    // pointers.
    double *pasol, *padsol, *pcecnd;
    // scalars.
    double tdep, sdep, sxdep, sydep
#if NDIM == 3
        , szdep
#endif
        ;
    // iterators.
    int icl, ieq;
    tdep = afreq * exd->time;
    pasol = asol + exd->ngstcell*NEQ;
    padsol = adsol + exd->ngstcell*NEQ*NDIM;
    pcecnd = exd->cecnd;
    for (icl=0; icl<exd->ncell; icl++) {
        sdep = wvec[0]*(pcecnd[0]-ctr[0]) + wvec[1]*(pcecnd[1]-ctr[1])
#if NDIM == 3
            + wvec[2]*(pcecnd[2]-ctr[2])
#endif
            ;
        sxdep = -sin(sdep - tdep);
        sdep = cos(sdep - tdep);
#if NDIM == 3
        szdep = wvec[2]*sxdep;
#endif
        sydep = wvec[1]*sxdep;
        sxdep = wvec[0]*sxdep;
        for (ieq=0; ieq<NEQ; ieq++) {
            pasol[ieq] += amp[ieq] * sdep;
            padsol[0] += amp[ieq] * sxdep;
            padsol[1] += amp[ieq] * sydep;
#if NDIM == 3
            padsol[2] += amp[ieq] * szdep;
#endif
            padsol += NDIM;
        };
        pasol += NEQ;
        pcecnd += (CLMFC+1) * NDIM;
    };
    return 0;
};
// vim: set ts=4 et:
