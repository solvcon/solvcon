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
// FIXME: this function shouldn't go to CUDA, doesn't make sense.
__global__ void cuda_process_schelieren_rhog(exedata *exd,
        double *rhog) {
    // and this starting index is incorrect.
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
#else
int process_schlieren_rhog(exedata *exd, double *rhog) {
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    // pointers.
    double *pdsoln;
    double *prhog;
    // iterators.
    int icl;
#ifndef __CUDACC__
    #pragma omp parallel for private(pdsoln, prhog, icl)
    for (icl=-exd->ngstcell; icl<exd->ncell; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
        pdsoln = exd->dsoln + icl*NEQ*NDIM;
        prhog = rhog + icl+exd->ngstcell;
        // density gradient.
        prhog[0] = pdsoln[0]*pdsoln[0] + pdsoln[1]*pdsoln[1];
#if NDIM == 3
        prhog[0] += pdsoln[2]*pdsoln[2];
#endif
        prhog[0] = sqrt(prhog[0]);
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int process_schelieren_rhog(int nthread, exedata *exc, void *gexc,
        double *rhog) {
    int nblock = (exc->ncell + nthread-1) / nthread;
    cuda_process_schelieren_rhog<<<nblock, nthread>>>((exedata *)gexc, rhog);
    cudaThreadSynchronize();
    return 0;
};
#endif

#ifdef __CUDACC__
// FIXME: this function shouldn't go to CUDA, doesn't make sense.
__global__ void cuda_process_schelieren_sch(exedata *exd,
        double k, double k0, double k1, double rhogmax, double *sch) {
    // and this starting index is incorrect.
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
#else
int process_schlieren_sch(exedata *exd,
        double k, double k0, double k1, double rhogmax, double *sch) {
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    // pointers.
    double *psch;
    // scalars.
    double fac0, fac1;
    // iterators.
    int icl;
    fac0 = k0 * rhogmax;
    fac1 = -k / ((k1-k0) * rhogmax + SOLVCON_ALMOST_ZERO);
#ifndef __CUDACC__
    #pragma omp parallel for private(psch, icl) \
    firstprivate(fac0, fac1)
    for (icl=-exd->ngstcell; icl<exd->ncell; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
        psch = sch + icl+exd->ngstcell;
        // density gradient.
        psch[0] = exp((psch[0]-fac0)*fac1);
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int process_schelieren_sch(int nthread, exedata *exc, void *gexc,
        double k, double k0, double k1, double rhogmax, double *sch) {
    int nblock = (exc->ncell + nthread-1) / nthread;
    cuda_process_schelieren_sch<<<nblock, nthread>>>((exedata *)gexc,
        k, k0, k1, rhogmax, sch);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:
