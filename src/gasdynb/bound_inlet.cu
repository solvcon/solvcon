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
__global__ void cuda_bound_inlet_soln(exedata *exd, int nbnd, int *facn,
    int nvalue, double *value) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_inlet_soln(exedata *exd, int nbnd, int *facn,
    int nvalue, double *value) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls;
    double *pvalue, *pjsoln;
    // scalars.
    double rho, p, ga, ke;
    double v1, v2, v3;
    // iterators.
    int ifc, jcl;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, pfacn, pfccls, \
    pvalue, pjsoln, rho, p, ga, ke, v1, v2, v3, ifc, jcl)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        pvalue = value + ibnd*nvalue;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        jcl = pfccls[1];
        // extract parameters.
        rho = pvalue[0];
        v1 = pvalue[1];
        v2 = pvalue[2];
#if NDIM == 3
        v3 = pvalue[3];
#endif
        ke = (v1*v1 + v2*v2
#if NDIM == 3
            + v3*v3
#endif
        )*rho/2.0;
        p = pvalue[4];
        ga = pvalue[5];
        // set solutions.
        pjsoln = exd->soln + jcl*NEQ;
        pjsoln[0] = rho;
        pjsoln[1] = v1*rho;
        pjsoln[2] = v2*rho;
#if NDIM == 3
        pjsoln[3] = v3*rho;
#endif
        pjsoln[1+NDIM] = p/(ga-1.0) + ke;
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_inlet_soln(int nthread, void *gexc,
    int nbnd, void *gfacn, int nvalue, void *value) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_inlet_soln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn, nvalue, (double *)value);
    cudaThreadSynchronize();
    return 0;
};
#endif

#ifdef __CUDACC__
__global__ void cuda_bound_inlet_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_inlet_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls;
    double *pjdsoln;
    // iterators.
    int ifc, jcl, it;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, pfacn, pfccls, \
    pjdsoln, ifc, jcl, it)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        jcl = pfccls[1];
        pjdsoln = exd->dsoln + jcl*NEQ*NDIM;
        // set to zero.
        for (it=0; it<NEQ*NDIM; it++) {
            pjdsoln[it] = 0.0;
        };
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_inlet_dsoln(int nthread, void *gexc,
    int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_inlet_dsoln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:
