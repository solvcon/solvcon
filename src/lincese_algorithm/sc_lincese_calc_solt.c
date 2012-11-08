/*
 * Copyright (C) 2008-2012 Yung-Yu Chen <yyc@solvcon.net>.
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

#include <Python.h>

#include "mesh.h"
#include "lincese_algorithm.h"

#define NDIM msd->ndim
#define NEQ exd->neq

void sc_lincese_calc_solt(sc_mesh_t *msd, sc_lincese_algorithm_t *exd) {
    // pointers.
    double *psolt, *pidsol, *pdsol;
    // scalars.
    double val;
    // arrays.
    double jacos[NEQ][NEQ][NDIM];
    double fcn[NEQ][NDIM];
    // interators.
    int icl, ieq, jeq, idm;
    #pragma omp parallel for \
    private(psolt, pidsol, pdsol, val, jacos, fcn, ieq, jeq, idm)
    for (icl=-msd->ngstcell; icl<msd->ncell; icl++) {
        psolt = exd->solt + icl*NEQ;
        pidsol = exd->dsol + icl*NEQ*NDIM;
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
    };
};

// vim: set ts=4 et:
