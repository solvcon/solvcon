/*
 * Copyright (C) 2008-2010 Yung-Yu Chen.
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

#include "solvcon.h"
int calc_dsoln(MeshData *msd, ExecutionData *exd,
        FPTYPE *clcnd, FPTYPE *dsol, FPTYPE *dsoln) {
    FPTYPE *pdsol, *pdsoln, *pclcnd;
    int icl, ieq, idm;
    pdsol = dsol + msd->ngstcell*exd->neq*msd->ndim;
    pdsoln = dsoln + msd->ngstcell*exd->neq*msd->ndim;
    pclcnd = clcnd + msd->ngstcell*msd->ndim;
    for (icl=0; icl<msd->ncell; icl++) {
        for (ieq=0; ieq<exd->neq; ieq++) {
            for (idm=0; idm<msd->ndim; idm++) {
                pdsoln[idm] = pdsol[idm]
                            + pclcnd[idm] * exd->time_increment / 2.0;
            };
            pdsol += msd->ndim;
            pdsoln += msd->ndim;
        };
        pclcnd += msd->ndim;
    };
    return 0;
};
// vim: set ts=4 et:
