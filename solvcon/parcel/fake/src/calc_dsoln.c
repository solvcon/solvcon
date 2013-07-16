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
#include "fake_algorithm.h"

int calc_dsoln(sc_mesh_t *msd, sc_fake_algorithm_t *alg) {
    double *pdsol, *pdsoln, *pclcnd;
    int icl, ieq, idm;
    pdsol = alg->dsol;
    pdsoln = alg->dsoln;
    pclcnd = msd->clcnd;
    for (icl=0; icl<msd->ncell; icl++) {
        for (ieq=0; ieq<alg->neq; ieq++) {
            for (idm=0; idm<msd->ndim; idm++) {
                pdsoln[idm] = pdsol[idm]
                            + pclcnd[idm] * alg->time_increment / 2.0;
            };
            pdsol += msd->ndim;
            pdsoln += msd->ndim;
        };
        pclcnd += msd->ndim;
    };
    return 0;
};

// vim: set ts=4 et:
