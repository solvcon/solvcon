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
__global__ void cuda_bound_nswall_soln(exedata *exd, int nbnd, int *facn) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_nswall_soln(exedata *exd, int nbnd, int *facn) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd, *pfccnd;
    double *pisoln, *pjsoln;
    // scalars.
    double len;
    // arrays.
    double mat[NDIM][NDIM], mvt[NDIM][NDIM];
    double mom[NDIM];
    // iterators.
    int ifc, icl, jcl;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, \
    pfacn, pfccls, pfcnds, pfcnml, pndcrd, pfccnd, \
    pisoln, pjsoln, len, mat, mvt, mom, ifc, icl, jcl)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pisoln = exd->soln + icl*NEQ;
        pjsoln = exd->soln + jcl*NEQ;
        // rotation and inverse rotation matrices.
        pfcnml = exd->fcnml + ifc*NDIM;
        mat[0][0] = mvt[0][0] = pfcnml[0];
        mat[0][1] = mvt[1][0] = pfcnml[1];
#if NDIM == 3
        mat[0][2] = mvt[2][0] = pfcnml[2];
        pfcnds = exd->fcnds + ifc*(FCMND+1);
        pndcrd = exd->ndcrd + pfcnds[1]*NDIM;
        pfccnd = exd->fccnd + ifc*NDIM;
        mat[1][0] = pndcrd[0] - pfccnd[0];
        mat[1][1] = pndcrd[1] - pfccnd[1];
        mat[1][2] = pndcrd[2] - pfccnd[2];
        len = sqrt(mat[1][0]*mat[1][0] + mat[1][1]*mat[1][1]
                 + mat[1][2]*mat[1][2]);
        mat[1][0] = mvt[0][1] = mat[1][0]/len;
        mat[1][1] = mvt[1][1] = mat[1][1]/len;
        mat[1][2] = mvt[2][1] = mat[1][2]/len;
        mat[2][0] = mvt[0][2] = mat[0][1]*mat[1][2] - mat[0][2]*mat[1][1];
        mat[2][1] = mvt[1][2] = mat[0][2]*mat[1][0] - mat[0][0]*mat[1][2];
        mat[2][2] = mvt[2][2] = mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0];
#else
        mat[1][0] = mvt[0][1] =  pfcnml[1];
        mat[1][1] = mvt[1][1] = -pfcnml[0];
#endif
        // rotate momentum vector.
#if NDIM == 3
        mom[0] = mat[0][0]*pisoln[1] + mat[0][1]*pisoln[2]
               + mat[0][2]*pisoln[3];
        mom[1] = mat[1][0]*pisoln[1] + mat[1][1]*pisoln[2]
               + mat[1][2]*pisoln[3];
        mom[2] = mat[2][0]*pisoln[1] + mat[2][1]*pisoln[2]
               + mat[2][2]*pisoln[3];
#else
        mom[0] = mat[0][0]*pisoln[1] + mat[0][1]*pisoln[2];
        mom[1] = mat[1][0]*pisoln[1] + mat[1][1]*pisoln[2];
#endif
        // set momentum.
        mom[0] = -mom[0];
        mom[1] = -mom[1];
#if NDIM == 3
        mom[2] = -mom[2];
#endif
        // inversely rotate momentum vector.
#if NDIM == 3
        pjsoln[1] = mvt[0][0]*mom[0] + mvt[0][1]*mom[1]
                  + mvt[0][2]*mom[2];
        pjsoln[2] = mvt[1][0]*mom[0] + mvt[1][1]*mom[1]
                  + mvt[1][2]*mom[2];
        pjsoln[3] = mvt[2][0]*mom[0] + mvt[2][1]*mom[1]
                  + mvt[2][2]*mom[2];
#else
        pjsoln[1] = mvt[0][0]*mom[0] + mvt[0][1]*mom[1];
        pjsoln[2] = mvt[1][0]*mom[0] + mvt[1][1]*mom[1];
#endif
        // set solutions.
        pjsoln[0] = pisoln[0];
        pjsoln[1+NDIM] = pisoln[1+NDIM];
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_nswall_soln(int nthread, void *gexc,
        int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_nswall_soln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

#ifdef __CUDACC__
__global__ void cuda_bound_nswall_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_nswall_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd, *pfccnd, (*pten)[NDIM];
    double *pidsoln, *pjdsoln, *pdsoln;
    // scalars.
    double len;
    // arrays.
    double vec[NEQ][NDIM];
    double vmt[NDIM][NDIM];
    double mat[NDIM][NDIM], mvt[NDIM][NDIM];
    // iterators.
    int ifc, icl, jcl, ieq, it, jt;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, \
    pfacn, pfccls, pfcnds, pfcnml, pndcrd, pfccnd, pten, \
    pidsoln, pjdsoln, pdsoln, \
    len, vec, vmt, mat, mvt, ifc, icl, jcl, ieq, it, jt)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pidsoln = exd->dsoln + icl*NEQ*NDIM;
        pjdsoln = exd->dsoln + jcl*NEQ*NDIM;
        // coordinate transformation and set transformed vectors.
        pfcnml = exd->fcnml + ifc*NDIM;
        mat[0][0] = mvt[0][0] = pfcnml[0];
        mat[0][1] = mvt[1][0] = pfcnml[1];
#if NDIM == 3
        mat[0][2] = mvt[2][0] = pfcnml[2];
        pfcnds = exd->fcnds + ifc*(FCMND+1);
        pndcrd = exd->ndcrd + pfcnds[1]*NDIM;
        pfccnd = exd->fccnd + ifc*NDIM;
        mat[1][0] = pndcrd[0] - pfccnd[0];
        mat[1][1] = pndcrd[1] - pfccnd[1];
        mat[1][2] = pndcrd[2] - pfccnd[2];
        len = sqrt(mat[1][0]*mat[1][0] + mat[1][1]*mat[1][1]
                 + mat[1][2]*mat[1][2]);
        mat[1][0] = mvt[0][1] = mat[1][0]/len;
        mat[1][1] = mvt[1][1] = mat[1][1]/len;
        mat[1][2] = mvt[2][1] = mat[1][2]/len;
        mat[2][0] = mvt[0][2] = mat[0][1]*mat[1][2] - mat[0][2]*mat[1][1];
        mat[2][1] = mvt[1][2] = mat[0][2]*mat[1][0] - mat[0][0]*mat[1][2];
        mat[2][2] = mvt[2][2] = mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0];
#else
        mat[1][0] = mvt[0][1] =  pfcnml[1];
        mat[1][1] = mvt[1][1] = -pfcnml[0];
#endif
        pdsoln = pidsoln;
        for (ieq=0; ieq<NEQ; ieq+=(NDIM+1)) {
#if NDIM == 3
            vec[ieq][0] = mat[0][0]*pdsoln[0] + mat[0][1]*pdsoln[1]
                        + mat[0][2]*pdsoln[2];
            vec[ieq][1] = mat[1][0]*pdsoln[0] + mat[1][1]*pdsoln[1]
                        + mat[1][2]*pdsoln[2];
            vec[ieq][2] = mat[2][0]*pdsoln[0] + mat[2][1]*pdsoln[1]
                        + mat[2][2]*pdsoln[2];
#else
            vec[ieq][0] = mat[0][0]*pdsoln[0] + mat[0][1]*pdsoln[1];
            vec[ieq][1] = mat[1][0]*pdsoln[0] + mat[1][1]*pdsoln[1];
#endif
            pdsoln += (NDIM+1)*NDIM;
        };
        pten = (double(*)[NDIM])(pidsoln+NDIM);
        for (it=0; it<NDIM; it++) {
            for (jt=0; jt<NDIM; jt++) {
                vmt[it][jt] = mat[it][0]*pten[0][jt] + mat[it][1]*pten[1][jt]
#if NDIM == 3
                            + mat[it][2]*pten[2][jt]
#endif
                ;
            };
        };
        for (it=0; it<NDIM; it++) {
            for (jt=0; jt<NDIM; jt++) {
                vec[it+1][jt] = vmt[it][0]*mvt[0][jt] + vmt[it][1]*mvt[1][jt]
#if NDIM == 3
                              + vmt[it][2]*mvt[2][jt]
#endif
                ;
            };
        };
        // set wall condition in the rotated coordinate;
        vec[0][0] = -vec[0][0];
        vec[1][1] = -vec[1][1];
#if NDIM == 3
        vec[1][2] = -vec[1][2];
#endif
        vec[2][1] = -vec[2][1];
#if NDIM == 3
        vec[2][2] = -vec[2][2];
        vec[3][1] = -vec[3][1];
        vec[3][2] = -vec[3][2];
#endif
        vec[1+NDIM][0] = -vec[1+NDIM][0];
        // inversely transform the coordinate and set ghost gradient.
        pdsoln = pjdsoln;
        for (ieq=0; ieq<NEQ; ieq+=(NDIM+1)) {
#if NDIM == 3
            pdsoln[0] = mvt[0][0]*vec[ieq][0] + mvt[0][1]*vec[ieq][1]
                      + mvt[0][2]*vec[ieq][2];
            pdsoln[1] = mvt[1][0]*vec[ieq][0] + mvt[1][1]*vec[ieq][1]
                      + mvt[1][2]*vec[ieq][2];
            pdsoln[2] = mvt[2][0]*vec[ieq][0] + mvt[2][1]*vec[ieq][1]
                      + mvt[2][2]*vec[ieq][2];
#else
            pdsoln[0] = mvt[0][0]*vec[ieq][0] + mvt[0][1]*vec[ieq][1];
            pdsoln[1] = mvt[1][0]*vec[ieq][0] + mvt[1][1]*vec[ieq][1];
#endif
            pdsoln += (NDIM+1)*NDIM;
        };
        pten = (double(*)[NDIM])(pjdsoln+NDIM);
        for (it=0; it<NDIM; it++) {
            for (jt=0; jt<NDIM; jt++) {
                vmt[it][jt] = mvt[it][0]*vec[1][jt] + mat[it][1]*vec[2][jt]
#if NDIM == 3
                            + mvt[it][2]*vec[3][jt]
#endif
                ;
            };
        };
        for (it=0; it<NDIM; it++) {
            for (jt=0; jt<NDIM; jt++) {
                pten[it][jt] = vmt[it][0]*mat[0][jt] + vmt[it][1]*mat[1][jt]
#if NDIM == 3
                             + vmt[it][2]*mat[2][jt]
#endif
                ;
            };
        };
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_nswall_dsoln(int nthread, void *gexc,
        int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_nswall_dsoln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:
