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

#include "elaslin.h"

void *calc_energy(exedata *exd, double *rhos, double *comps, double *en) {
    // pointers.
    int *pclgrp;
    double *psoln, *pcomp, *prho, *pen;
    // vectors.
    double stn[6];
    // iterators.
    int igp, icl;
    int it, jt;
    // loop over cells.
    pclgrp = exd->clgrp;
    psoln = exd->soln;
    pen = en + exd->ngstcell;
    for (icl=0; icl<exd->ncell; icl++) {
        igp = pclgrp[0];
        pcomp = comps + igp*36;
        prho = rhos + igp;
        // strain.
        for (it=0; it<6; it++) {
            stn[it] = 0.0;
            for (jt=0; jt<6; jt++) {
                stn[it] += pcomp[jt*6+it] * psoln[3+jt];
            };
        };
        // kinetic energy.
        pen[0] = psoln[0]*psoln[0] + psoln[1]*psoln[1] + psoln[2]*psoln[2];
        pen[0] *= prho[0];
        // strain energy.
        pen[0] += psoln[3]*stn[0] + psoln[4]*stn[1] + psoln[5]*stn[2]
                + psoln[6]*stn[3] + psoln[7]*stn[4] + psoln[8]*stn[5];
        // divided by two;
        pen[0] /= 2.0;
        // advance.
        pclgrp += 1;
        psoln += NEQ;
        pen += 1;
    };
    return NULL;
};
// vim: set ts=4 et:
