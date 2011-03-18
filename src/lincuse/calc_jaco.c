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

void calc_jaco(exedata *exd, int icl,   
        double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM]) {
    // pointers.
    double *psol;
    double *pjaco, *pljaco;
    // interators.
    int nt;
    int it, ieq, jeq;
    // fill jacobian.
    pjaco = exd->grpda + exd->clgrp[icl]*exd->gdlen;
    pljaco = (double *)jacos;
    nt = NEQ*NEQ*NDIM;
    for (it=0; it<nt; it++) {
        pljaco[it] = pjaco[it];
    };
    // calculate flux function.
    psol = exd->sol + icl*NEQ;
    for (ieq=0; ieq<NEQ; ieq++) {
        fcn[ieq][0] = 0.0;
        fcn[ieq][1] = 0.0;
#if NDIM == 3
        fcn[ieq][2] = 0.0;
#endif
        for (jeq=0; jeq<NEQ; jeq++) {
            fcn[ieq][0] += jacos[ieq][jeq][0] * psol[jeq];
            fcn[ieq][1] += jacos[ieq][jeq][1] * psol[jeq];
#if NDIM == 3
            fcn[ieq][2] += jacos[ieq][jeq][2] * psol[jeq];
#endif
        };
    };
    return;
};
// vim: set ts=4 et:
