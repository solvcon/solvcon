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

int calc_solt(exedata *exd, int istart, int iend) {
    int cputicks;
    struct tms timm0, timm1;
    // pointers.
    double *psolt, *pidsol, *pdsol;
    // scalars.
    double val;
    // arrays.
    double jacos[NEQ][NEQ][NDIM];
    double fcn[NEQ][NDIM];
    // interators.
    int icl, ieq, jeq, idm;
    times(&timm0);
#ifdef SOLVCESE_FE
    feenableexcept(SOLVCESE_FE);
#endif
    psolt = exd->solt + istart*NEQ;
    pidsol = exd->dsol + istart*NEQ*NDIM;
    for (icl=istart; icl<iend; icl++) {
        exd->jacofunc(exd, icl, (double *)fcn, (double *)jacos);
        for (ieq=0; ieq<NEQ; ieq++) {
            psolt[ieq] = 0.0;
            for (idm=0; idm<NDIM; idm++) {
                val = 0.0;
                pdsol = pidsol;
                for (jeq=0; jeq<NEQ; jeq++) {
                    val += jacos[ieq][jeq][idm]*pdsol[idm];
                    pdsol += NDIM;
                };
                psolt[ieq] -= val;
            };
        };
        // advance pointers.
        psolt += NEQ;
        pidsol += NEQ*NDIM;
    };
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
// vim: set ts=4 et:
