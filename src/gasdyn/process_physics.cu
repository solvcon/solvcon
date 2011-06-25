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

#ifdef __CUDACC__
// FIXME: this function shouldn't go to CUDA, doesn't make sense.
__global__ void cuda_process_physics(exedata *exd, double gasconst,
        double *vel, double *vor, double *vorm, double *rho, double *pre,
        double *tem, double *ken, double *sos, double *mac) {
    // and this starting index is incorrect.
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
#else
int process_physics(exedata *exd, double gasconst,
        double *vel, double *vor, double *vorm, double *rho, double *pre,
        double *tem, double *ken, double *sos, double *mac) {
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    // pointers.
    double *pclcnd, *pcecnd;
    double *pamsca, *psoln, *pdsoln;
    double (*pvd)[NDIM];    // shorthand for derivative.
    double *prho, *pvel, *pvor, *pvorm, *ppre, *ptem, *pken, *psos, *pmac;
    // scalars.
    double ga, ga1;
    // arrays.
    double sft[NDIM];
    // iterators.
    int icl;
#ifndef __CUDACC__
    #pragma omp parallel for private(pclcnd, pcecnd, pamsca, psoln, pdsoln, \
    pvd, prho, pvel, pvor, pvorm, ppre, ptem, pken, psos, pmac, \
    ga, ga1, sft, icl)
    for (icl=-exd->ngstcell; icl<exd->ncell; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
        pclcnd = exd->clcnd + icl*NDIM;
        pcecnd = exd->cecnd + icl*(CLMFC+1)*NDIM;
        pamsca = exd->amsca + icl*NSCA;
        psoln = exd->soln + icl*NEQ;
        pvel = vel + (icl+exd->ngstcell)*NDIM;
        pvor = vor + (icl+exd->ngstcell)*NDIM;
        pvorm = vorm + icl+exd->ngstcell;
        prho = rho + icl+exd->ngstcell;
        ppre = pre + icl+exd->ngstcell;
        ptem = tem + icl+exd->ngstcell;
        pken = ken + icl+exd->ngstcell;
        psos = sos + icl+exd->ngstcell;
        pmac = mac + icl+exd->ngstcell;
        // obtain flow parameters.
        ga = pamsca[0];
        ga1 = ga - 1;
        pdsoln = exd->dsoln + icl*NEQ*NDIM;
        pvd = (double (*)[NDIM])pdsoln;
        // shift from solution point to cell center.
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
        // vorticity.
#if NDIM == 3
        pvor[0] = ((pvd[3][1] - pvd[2][2])
                 - (pvel[2]*pvd[0][1] - pvel[1]*pvd[0][2])) / prho[0];
        pvor[1] = ((pvd[1][2] - pvd[3][0])
                 - (pvel[0]*pvd[0][2] - pvel[2]*pvd[0][0])) / prho[0];
        pvor[2] = ((pvd[2][0] - pvd[1][1])
                 - (pvel[1]*pvd[0][0] - pvel[0]*pvd[0][1])) / prho[0];
        pvorm[0] = sqrt(pvor[0]*pvor[0] + pvor[1]*pvor[1] + pvor[2]*pvor[2]);
#else
        pvor[0] = ((pvd[2][0] - pvd[1][1])
                 - (pvel[1]*pvd[0][0] - pvel[0]*pvd[0][1])) / prho[0];
        pvor[1] = pvor[0];
        pvorm[0] = fabs(pvor[0]);
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
        ptem[0] = ppre[0]/(prho[0]*gasconst);
        // speed of sound.
        psos[0] = sqrt(ga*ppre[0]/prho[0]);
        // Mach number.
        pmac[0] = sqrt(pken[0]/prho[0]*2);
        pmac[0] *= psos[0]
            / (psos[0]*psos[0] + SOLVCON_ALMOST_ZERO); // prevent nan/inf.
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int process_physics(int nthread, exedata *exc, void *gexc,
        double gasconst,
        double *vel, double *vor, double *vorm, double *rho, double *pre,
        double *tem, double *ken, double *sos, double *mac) {
    int nblock = (exc->ncell + nthread-1) / nthread;
    cuda_process_physics<<<nblock, nthread>>>((exedata *)gexc, gasconst,
        vel, vor, vorm, rho, pre, tem, ken, sos, mac);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:
