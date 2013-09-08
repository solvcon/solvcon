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

#include "bulk.h"

#ifdef __CUDACC__
__global__ void cuda_calc_cfl(exedata *exd) {
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
#else
int calc_cfl(exedata *exd) {
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    int clnfc;
    // pointers.
    int *pclfcs;
    double *pamsca, *pcfl, *pocfl, *psoln, *picecnd, *pcecnd;
    // scalars.
    double hdt, dist, wspd, pr, ke;
    double bulk, p0, rho0, eta;
    // arrays.
    double vec[NDIM];
    // iterators.
    int icl, ifl;
    hdt = exd->time_increment / 2.0;
#ifndef __CUDACC__
    #pragma omp parallel for private(clnfc, \
    pclfcs, pamsca, pcfl, pocfl, psoln, picecnd, pcecnd, \
    dist, wspd, pr, ke, vec, icl, ifl, bulk, p0, rho0, eta) \
    firstprivate(hdt)
    for (icl=0; icl<exd->ncell; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
        pamsca = exd->amsca + icl*NSCA;
        pcfl = exd->cfl + icl;
        pocfl = exd->ocfl + icl;
        psoln = exd->soln + icl*NEQ;
        picecnd = exd->cecnd + icl*(CLMFC+1)*NDIM;
        pclfcs = exd->clfcs + icl*(CLMFC+1);
        // estimate distance.
        dist = 1.e200;
        pcecnd = picecnd;
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ifl++) {
            pcecnd += NDIM;
            // distance.
            vec[0] = picecnd[0] - pcecnd[0];
            vec[1] = picecnd[1] - pcecnd[1];
#if NDIM == 3
            vec[2] = picecnd[2] - pcecnd[2];
#endif
            wspd = sqrt(vec[0]*vec[0] + vec[1]*vec[1]
#if NDIM == 3
                      + vec[2]*vec[2]
#endif
            );
            // minimal value.
            dist = fmin(wspd, dist);
        };
        // wave speed.
        bulk = pamsca[0];
        p0 = pamsca[1];
        rho0 = pamsca[2];
        eta = pamsca[3];
        wspd = psoln[1]*psoln[1] + psoln[2]*psoln[2]
#if NDIM == 3
             + psoln[3]*psoln[3]
#endif
        ;
        ke = wspd/(2.0*psoln[0]);
        // density base
        //wspd = sqrt(bulk/psoln[0]) + sqrt(wspd)/psoln[0];
        // pressure base
        wspd = sqrt(bulk/(eta*psoln[0])) + sqrt(wspd)/psoln[0];
        // CFL.
        pocfl[0] = hdt*wspd/dist;
        // if pressure is null, make CFL to be 1.
        //pcfl[0] = (pocfl[0]-1.0) * pr/(pr+SOLVCON_TINY) + 1.0;
        pcfl[0] = pocfl[0];
        // correct negative pressure.
        //psoln[1+NDIM] = pr/ga1 + ke + SOLVCON_TINY;
        // advance.
    };
#ifndef __CUDACC__
    return 0;
};
#else
};
extern "C" int calc_cfl(int nthread, exedata *exc, void *gexc) {
    int nblock = (exc->ncell + nthread-1) / nthread;
    cuda_calc_cfl<<<nblock, nthread>>>((exedata *)gexc);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:
