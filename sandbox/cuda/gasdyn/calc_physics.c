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

#include "gasdyn.h"

int calc_physics(exedata *exd, int istart, int iend,
        double *vel, double *rho, double *pre, double *tem, double *ken,
        double *sos, double *mac) {
    int cputicks;
    struct tms timm0, timm1;
    times(&timm0);
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
    // pointers.
    double *pclcnd, *pcecnd;
    double *pamsca, *psoln, *pdsoln;
    double *prho, *pvel, *ppre, *ptem, *pken, *psos, *pmac;
    // scalars.
    double ga, ga1;
    // arrays.
    double sft[NDIM];
    // iterators.
    int icl;
    pclcnd = exd->clcnd + istart*NDIM;
    pcecnd = exd->cecnd + istart*(CLMFC+1)*NDIM;
    pamsca = exd->amsca + istart*NSCA;
    psoln = exd->soln + istart*NEQ;
    pvel = vel + (istart+exd->ngstcell)*NDIM;
    prho = rho + istart+exd->ngstcell;
    ppre = pre + istart+exd->ngstcell;
    ptem = tem + istart+exd->ngstcell;
    pken = ken + istart+exd->ngstcell;
    psos = sos + istart+exd->ngstcell;
    pmac = mac + istart+exd->ngstcell;
    for (icl=istart; icl<iend; icl++) {
        ga = pamsca[0];
        ga1 = ga - 1;
        pdsoln = exd->dsoln + icl*NEQ*NDIM;
        sft[0] = pclcnd[0] - pcecnd[0];
        sft[1] = pclcnd[1] - pcecnd[1];
#if NDIM == 3
        sft[2] = pclcnd[2] - pcecnd[2];
#endif
        // density.
        prho[0] = psoln[0] + pdsoln[0]*sft[0] + pdsoln[1]*sft[1];
#if NDIM == 3
        prho[0] += pdsoln[2]*sft[2];
#endif
        // velocity.
        pdsoln += NDIM;
        pvel[0] = psoln[1] + pdsoln[0]*sft[0] + pdsoln[1]*sft[1];
#if NDIM == 3
        pvel[0] += pdsoln[2]*sft[2];
#endif
        pvel[0] /= prho[0];
        pken[0] = pvel[0]*pvel[0];
        pdsoln += NDIM;
        pvel[1] = psoln[2] + pdsoln[0]*sft[0] + pdsoln[1]*sft[1];
#if NDIM == 3
        pvel[1] += pdsoln[2]*sft[2];
#endif
        pvel[1] /= prho[0];
        pken[0] += pvel[1]*pvel[1];
#if NDIM == 3
        pdsoln += NDIM;
        pvel[2] = psoln[3] + pdsoln[0]*sft[0] + pdsoln[1]*sft[1];
        pvel[2] += pdsoln[2]*sft[2];
        pvel[2] /= prho[0];
        pken[0] += pvel[2]*pvel[2];
#endif
        // kinetic energy.
        pken[0] *= prho[0]/2;
        // pressure.
        pdsoln += NDIM;
        ppre[0] = psoln[NDIM+1] + pdsoln[0]*sft[0] + pdsoln[1]*sft[1];
#if NDIM == 3
        ppre[0] += pdsoln[2]*sft[2];
#endif
        ppre[0] = (ppre[0] - pken[0]) * ga1;
        ppre[0] = (ppre[0] + fabs(ppre[0])) / 2; // make sure it's positive.
        // temperature.
        ptem[0] = ppre[0]/prho[0];
        // speed of sound.
        psos[0] = sqrt(ga*ppre[0]/prho[0]);
        // Mach number.
        pmac[0] = sqrt(pken[0]/prho[0]*2);
        pmac[0] *= psos[0]
            / (psos[0]*psos[0] + SOLVCON_ALMOST_ZERO); // prevent nan/inf.
        // advance pointer.
        pclcnd += NDIM;
        pcecnd += (CLMFC+1)*NDIM;
        pamsca += 1;
        psoln += NEQ;
        pvel += NDIM;
        prho += 1;
        ppre += 1;
        ptem += 1;
        pken += 1;
        psos += 1;
        pmac += 1;
    };
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
// vim: set ts=4 et:
