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
    double *pamsca;
    double *pvalue, *pjsoln, *pisol;
    // scalars.
    double rhoi, bulk, rho, pi;
    double v1i, v2i, v3i, v1;
    double left, right;
    // pressure base
    double pini, p, eta;
    // iterators.
    int ifc, jcl, icl;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, pfacn, pfccls, \
    pamsca, pvalue, pjsoln, pisol, rhoi, bulk, rho, v1i, v2i, v3i, v1, left, right,\
    pini, p, eta, ifc, jcl, icl, pi)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        pvalue = value + ibnd*nvalue;
        pamsca = exd->amsca + ibnd*NSCA;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pisol = exd->sol + icl*NEQ;
        // extract parameters.
        bulk = pamsca[0];
        eta  = pamsca[3];
        pini = pamsca[5];
        // density base
        //rhoi = pvalue[0];
        // pressure base
        pi  = pvalue[0];
        v1i = pvalue[1];
        v2i = pvalue[2];
#if NDIM == 3
        v3i = pvalue[3];
#endif
        // density base
        /*
        rho = pisol[0];
        v1 = pisol[1]/pisol[0];
        right = -pow(rhoi,-0.5) + v1i/(2*sqrt(bulk));
        left = -pow(rho,-0.5) - v1/(2*sqrt(bulk));
        pjsoln = exd->soln +jcl*NEQ;
        pjsoln[0] = 4/pow(right+left,2);
        pjsoln[1] = pjsoln[0]*(right-left)*sqrt(bulk);
        pjsoln[2] = 0.0;
        */
        // pressure base 
        //
        p = pisol[0];
        v1 = pisol[1]/pisol[0];
        right = -pow(pi,-0.5) + v1i*sqrt(eta/bulk)/2;
        left = -pow(p,-0.5) + v1*sqrt(eta/bulk)/2;
        pjsoln = exd->soln + jcl*NEQ;
        pjsoln[0] = 4/pow(right+left,2);
        pjsoln[1] = pjsoln[0]*(right-left)*sqrt(bulk/eta);
        pjsoln[2] = 0.0;
        //
        // force inlet
        /*
        pjsoln = exd->soln +jcl*NEQ;
        pjsoln[0] = rhoi;
        pjsoln[1] = rhoi*v1i;
        pjsoln[2] = 0.0;
        */

#if NDIM == 3
        pjsoln[3] = 0.0;
#endif
       
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
