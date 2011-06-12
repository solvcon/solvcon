/*
 * Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
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

#include "cuse.h"

#define CLNFC (NDIM+1)
#define FCNND NDIM

#ifdef __CUDACC__
__global__ void cuda_calcs_soln(exedata *exd) {
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
#else
int calcs_soln(exedata *exd, int istart, int iend) {
    struct tms timm0, timm1;
    int cputicks;
    times(&timm0);
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    // partial pointers.
    int *pclfcs, *pfcnds, *pfccls;
    double *pndcrd,  *pclcnd;
    double *pjcecnd, *pcecnd, *pcevol;
    double *pjsol, *pdsol, *pjsolt, *psoln;
    // scalars.
    double hdt, qdt, voe;
    double fusp, futm;
#if NDIM == 3
    double disu0, disu1, disu2;
    double disv0, disv1, disv2;
#endif
    // arrays.
    double crd[FCNND+1][NDIM];
    double cnde[NDIM];
    double sfnml[FCMND][NDIM];
    double sfcnd[FCMND][NDIM];
    double usfc[NEQ];
    double fcn[NEQ][NDIM], dfcn[NEQ][NDIM];
    double jacos[NEQ][NEQ][NDIM];
    // interators.
    int icl, ifl, inf, ifc, jcl, ieq, jeq;
    qdt = exd->time_increment * 0.25;
    hdt = exd->time_increment * 0.5;
#ifndef __CUDACC__
    psoln = exd->soln + istart*NEQ;
    pcevol = exd->cevol + istart*(CLMFC+1);
    for (icl=istart; icl<iend; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
        psoln = exd->soln + istart*NEQ;
        pcevol = exd->cevol + istart*(CLMFC+1);
#endif
        // initialize fluxes.
        for (ieq=0; ieq<NEQ; ieq++) {
            psoln[ieq] = 0.0;
        };

        pclfcs = exd->clfcs + icl*(CLMFC+1);
        for (ifl=1; ifl<=CLNFC; ifl++) {
            ifc = pclfcs[ifl];
            // face node coordinates. (unrolled)
            pfcnds = exd->fcnds + ifc*(FCMND+1);
            pndcrd = exd->ndcrd + pfcnds[0+1]*NDIM;   // node 0.
            crd[0][0] = crd[FCNND][0] = pndcrd[0];
            crd[0][1] = crd[FCNND][1] = pndcrd[1];
#if NDIM == 3
            crd[0][2] = crd[FCNND][2] = pndcrd[2];
#endif
            pndcrd = exd->ndcrd + pfcnds[1+1]*NDIM;   // node 1.
            crd[1][0] = pndcrd[0];
            crd[1][1] = pndcrd[1];
#if NDIM == 3
            crd[1][2] = pndcrd[2];
            pndcrd = exd->ndcrd + pfcnds[2+1]*NDIM;   // node 2.
            crd[2][0] = pndcrd[0];
            crd[2][1] = pndcrd[1];
            crd[2][2] = pndcrd[2];
#endif
            // neighboring cell center.
            pfccls = exd->fccls + ifc*FCREL;
            jcl = pfccls[0] + pfccls[1] - icl;
            pclcnd = exd->clcnd + jcl*NDIM;
            cnde[0] = pclcnd[0];
            cnde[1] = pclcnd[1];
#if NDIM == 3
            cnde[2] = pclcnd[2];
#endif
            // calculate geometric center of the bounding sub-face. (unrolled)
            sfcnd[0][0] = cnde[0] + crd[0][0];    // sub-face 0.
#if NDIM == 3
            sfcnd[0][0] += crd[0+1][0];
#endif
            sfcnd[0][0] /= NDIM;
            sfcnd[0][1] = cnde[1] + crd[0][1];
#if NDIM == 3
            sfcnd[0][1] += crd[0+1][1];
#endif
            sfcnd[0][1] /= NDIM;
#if NDIM == 3
            sfcnd[0][2] = cnde[2] + crd[0][2];
            sfcnd[0][2] += crd[0+1][2];
            sfcnd[0][2] /= NDIM;
#endif
            sfcnd[1][0] = cnde[0] + crd[1][0];    // sub-face 1.
#if NDIM == 3
            sfcnd[1][0] += crd[1+1][0];
#endif
            sfcnd[1][0] /= NDIM;
            sfcnd[1][1] = cnde[1] + crd[1][1];
#if NDIM == 3
            sfcnd[1][1] += crd[1+1][1];
#endif
            sfcnd[1][1] /= NDIM;
#if NDIM == 3
            sfcnd[1][2] = cnde[2] + crd[1][2];
            sfcnd[1][2] += crd[1+1][2];
            sfcnd[1][2] /= NDIM;
            sfcnd[2][0] = cnde[0] + crd[2][0];    // sub-face 2.
            sfcnd[2][0] += crd[2+1][0];
            sfcnd[2][0] /= NDIM;
            sfcnd[2][1] = cnde[1] + crd[2][1];
            sfcnd[2][1] += crd[2+1][1];
            sfcnd[2][1] /= NDIM;
            sfcnd[2][2] = cnde[2] + crd[2][2];
            sfcnd[2][2] += crd[2+1][2];
            sfcnd[2][2] /= NDIM;
#endif
            // calculate outward area vector of the bounding sub-face.
            // (unrolled)
#if NDIM == 3
            voe = (pfccls[0] - icl) + SOLVCON_ALMOST_ZERO;
            voe /= (icl - pfccls[0]) + SOLVCON_ALMOST_ZERO;
            voe *= 0.5;
            disu0 = crd[0  ][0] - cnde[0];  // sub-face 0.
            disu1 = crd[0  ][1] - cnde[1];
            disu2 = crd[0  ][2] - cnde[2];
            disv0 = crd[0+1][0] - cnde[0];
            disv1 = crd[0+1][1] - cnde[1];
            disv2 = crd[0+1][2] - cnde[2];
            sfnml[0][0] = (disu1*disv2 - disu2*disv1) * voe;
            sfnml[0][1] = (disu2*disv0 - disu0*disv2) * voe;
            sfnml[0][2] = (disu0*disv1 - disu1*disv0) * voe;
            disu0 = crd[1  ][0] - cnde[0];  // sub-face 1.
            disu1 = crd[1  ][1] - cnde[1];
            disu2 = crd[1  ][2] - cnde[2];
            disv0 = crd[1+1][0] - cnde[0];
            disv1 = crd[1+1][1] - cnde[1];
            disv2 = crd[1+1][2] - cnde[2];
            sfnml[1][0] = (disu1*disv2 - disu2*disv1) * voe;
            sfnml[1][1] = (disu2*disv0 - disu0*disv2) * voe;
            sfnml[1][2] = (disu0*disv1 - disu1*disv0) * voe;
            disu0 = crd[2  ][0] - cnde[0];  // sub-face 2.
            disu1 = crd[2  ][1] - cnde[1];
            disu2 = crd[2  ][2] - cnde[2];
            disv0 = crd[2+1][0] - cnde[0];
            disv1 = crd[2+1][1] - cnde[1];
            disv2 = crd[2+1][2] - cnde[2];
            sfnml[2][0] = (disu1*disv2 - disu2*disv1) * voe;
            sfnml[2][1] = (disu2*disv0 - disu0*disv2) * voe;
            sfnml[2][2] = (disu0*disv1 - disu1*disv0) * voe;
#else
            voe = (crd[0][0]-cnde[0])*(crd[1][1]-cnde[1])
                - (crd[0][1]-cnde[1])*(crd[1][0]-cnde[0]);
            voe /= fabs(voe);
            sfnml[0][0] = -(cnde[1]-crd[0][1]) * voe;
            sfnml[0][1] =  (cnde[0]-crd[0][0]) * voe;
            sfnml[1][0] =  (cnde[1]-crd[1][1]) * voe;
            sfnml[1][1] = -(cnde[0]-crd[1][0]) * voe;
#endif

            // spatial flux (given time).
            pjcecnd = exd->cecnd + jcl*(CLMFC+1)*NDIM;
            pcecnd = exd->cecnd + (icl*(CLMFC+1)+ifl)*NDIM;
            pjsol = exd->sol + jcl*NEQ;
            pdsol = exd->dsol + jcl*NEQ*NDIM;
            for (ieq=0; ieq<NEQ; ieq++) {
                fusp = pjsol[ieq];
                fusp += (pcecnd[0]-pjcecnd[0]) * pdsol[0];
                fusp += (pcecnd[1]-pjcecnd[1]) * pdsol[1];
#if NDIM == 3
                fusp += (pcecnd[2]-pjcecnd[2]) * pdsol[2];
#endif
                psoln[ieq] += fusp * pcevol[ifl];
                pdsol += NDIM;
            };

            // temporal flux (give space).
#ifndef __CUDACC__
            exd->jacofunc(exd, jcl, (double *)fcn, (double *)jacos);
#else
            cuda_calc_jaco(exd, jcl, fcn, jacos);
#endif
            pjsolt = exd->solt + jcl*NEQ;
            for (inf=0; inf<FCNND; inf++) {
                // solution at sub-face center.
                pdsol = exd->dsol + jcl*NEQ*NDIM;
                for (ieq=0; ieq<NEQ; ieq++) {
                    usfc[ieq] = qdt * pjsolt[ieq];
                    usfc[ieq] += (sfcnd[inf][0]-pjcecnd[0]) * pdsol[0];
                    usfc[ieq] += (sfcnd[inf][1]-pjcecnd[1]) * pdsol[1];
#if NDIM == 3
                    usfc[ieq] += (sfcnd[inf][2]-pjcecnd[2]) * pdsol[2];
#endif
                    pdsol += NDIM;
                };
                // spatial derivatives.
                for (ieq=0; ieq<NEQ; ieq++) {
                    dfcn[ieq][0] = fcn[ieq][0];
                    dfcn[ieq][1] = fcn[ieq][1];
#if NDIM == 3
                    dfcn[ieq][2] = fcn[ieq][2];
#endif
                    for (jeq=0; jeq<NEQ; jeq++) {
                        dfcn[ieq][0] += jacos[ieq][jeq][0] * usfc[jeq];
                        dfcn[ieq][1] += jacos[ieq][jeq][1] * usfc[jeq];
#if NDIM == 3
                        dfcn[ieq][2] += jacos[ieq][jeq][2] * usfc[jeq];
#endif
                    };
                };
                // temporal flux.
                for (ieq=0; ieq<NEQ; ieq++) {
                    futm = 0.0;
                    futm += dfcn[ieq][0] * sfnml[inf][0];
                    futm += dfcn[ieq][1] * sfnml[inf][1];
#if NDIM == 3
                    futm += dfcn[ieq][2] * sfnml[inf][2];
#endif
                    psoln[ieq] -= hdt*futm;
                };
            };
        };

        // update solutions.
        for (ieq=0; ieq<NEQ; ieq++) {
            psoln[ieq] /= pcevol[0];
        };
#ifndef __CUDACC__
        // advance pointers.
        psoln += NEQ;
        pcevol += CLMFC+1;
    };
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
#else
    };
};
extern "C" int calcs_soln(int nthread, exedata *exc, void *gexc) {
    int nblock = (exc->ncell + nthread-1) / nthread;
    cuda_calcs_soln<<<nblock, nthread>>>((exedata *)gexc);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ft=cuda ts=4 et:
