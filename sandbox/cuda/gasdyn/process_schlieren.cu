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
__global__ void cuda_process_schelieren_rhog(exedata *exd,
        double *rhog) {
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
#else
int process_schlieren_rhog(exedata *exd, int istart, int iend,
        double *rhog) {
    struct tms timm0, timm1;
    int cputicks;
    times(&timm0);
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    // pointers.
    double *pdsoln;
    double *prhog;
    // iterators.
    int icl;
    pdsoln = exd->dsoln + istart*NEQ*NDIM;
    prhog = rhog + istart+exd->ngstcell;
#ifdef __CUDACC__
    icl = istart;
    if (icl < exd->ncell) {
#else
    for (icl=istart; icl<iend; icl++) {
#endif
        // density gradient.
        prhog[0] = pdsoln[0]*pdsoln[0] + pdsoln[1]*pdsoln[1];
#if NDIM == 3
        prhog[0] += pdsoln[2]*pdsoln[2];
#endif
        prhog[0] = sqrt(prhog[0]);
#ifndef __CUDACC__
        // advance pointers.
        pdsoln += NEQ*NDIM;
        prhog += 1;
    };
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
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
__global__ void cuda_process_schelieren_sch(exedata *exd,
        double k, double k0, double k1, double rhogmax, double *sch) {
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
#else
int process_schlieren_sch(exedata *exd, int istart, int iend,
        double k, double k0, double k1, double rhogmax, double *sch) {
    struct tms timm0, timm1;
    int cputicks;
    times(&timm0);
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
    psch = sch + istart+exd->ngstcell;
#ifdef __CUDACC__
    icl = istart;
    if (icl < exd->ncell) {
#else
    for (icl=istart; icl<iend; icl++) {
#endif
        // density gradient.
        psch[0] = exp((psch[0]-fac0)*fac1);
#ifndef __CUDACC__
        // advance pointers.
        psch += 1;
    };
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
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
