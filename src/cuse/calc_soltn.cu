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

#ifdef __CUDACC__
__device__ void cuda_calc_jaco(exedata *exd, int icl,
        double (*fcn)[NDIM], double (*jacos)[NDIM]) {
    return;
};
#endif

#ifdef __CUDACC__
__global__ void cuda_calc_solt(exedata *exd) {
    int istart = -exd->ngstcell + blockDim.x * blockIdx.x + threadIdx.x;
#else
int calc_solt(exedata *exd, int istart, int iend) {
    struct tms timm0, timm1;
    int cputicks;
    times(&timm0);
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    // pointers.
    double *psolt, *pidsol, *pdsol;
    // scalars.
    double val;
    // arrays.
#ifdef __CUDACC__
    double (*jacos)[NDIM];
    double (*fcn)[NDIM];
    jacos = (double (*)[NDIM])malloc(NEQ*NEQ*NDIM*sizeof(double));
    fcn = (double (*)[NDIM])malloc(NEQ*NDIM*sizeof(double));
#else
    double jacos[NEQ*NEQ][NDIM];
    double fcn[NEQ][NDIM];
#endif
    // interators.
    int icl, ieq, jeq, idm;
    psolt = exd->solt + istart*NEQ;
    pidsol = exd->dsol + istart*NEQ*NDIM;
#ifndef __CUDACC__
    for (icl=istart; icl<iend; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
#ifndef __CUDACC__
        exd->jacofunc(exd, icl, (double *)fcn, (double *)jacos);
#else
        cuda_calc_jaco(exd, icl, fcn, jacos);
#endif
        for (ieq=0; ieq<NEQ; ieq++) {
            psolt[ieq] = 0.0;
            for (idm=0; idm<NDIM; idm++) {
                val = 0.0;
                pdsol = pidsol;
                for (jeq=0; jeq<NEQ; jeq++) {
                    val += jacos[ieq*NEQ+jeq][idm]*pdsol[idm];
                    pdsol += NDIM;
                };
                psolt[ieq] -= val;
            };
        };
#ifndef __CUDACC__
        // advance pointers.
        psolt += NEQ;
        pidsol += NEQ*NDIM;
    };
};
#else
    };
    free(jacos);
    free(fcn);
};
extern "C" int calc_solt(int nthread, exedata *exc, void *gexc) {
    int nblock = (exc->ngstcell + exc->ncell + nthread-1) / nthread;
    cuda_calc_solt<<<nblock, nthread>>>((exedata *)gexc);
    cudaThreadSynchronize();
    return 0;
};
#endif

#ifdef __CUDACC__
__global__ void cuda_calc_soln(exedata *exd) {
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
#else
int calc_soln(exedata *exd, int istart, int iend) {
    struct tms timm0, timm1;
    int cputicks;
    times(&timm0);
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    int clnfc, fcnnd;
    // partial pointers.
    int *pclfcs, *pfcnds, *pfccls;
    double *pndcrd,  *pclcnd;
    double *pjcecnd, *pcecnd, *pcevol;
    double *pjsol, *pdsol, *pjsolt, *psoln;
    // scalars.
    double hdt, qdt, voe;
#if NDIM == 3
    double disu0, disu1, disu2;
    double disv0, disv1, disv2;
#endif
    // arrays.
    double crd[FCMND+1][NDIM];
    double cnde[NDIM];
    double sfnml[FCMND][NDIM];
    double sfcnd[FCMND][NDIM];
#ifdef __CUDACC__
    double *futo, *fusp, *futm;
    double (*jacos)[NDIM];
    double *usfc;
    double (*fcn)[NDIM], (*dfcn)[NDIM];
    futo = (double *)malloc(NEQ*sizeof(double));
    fusp = (double *)malloc(NEQ*sizeof(double));
    futm = (double *)malloc(NEQ*sizeof(double));
    jacos = (double (*)[NDIM])malloc(NEQ*NEQ*NDIM*sizeof(double));
    usfc = (double *)malloc(NEQ*sizeof(double));
    fcn = (double (*)[NDIM])malloc(NEQ*NDIM*sizeof(double));
    dfcn = (double (*)[NDIM])malloc(NEQ*NDIM*sizeof(double));
#else
    double futo[NEQ], fusp[NEQ], futm[NEQ];
    double jacos[NEQ*NEQ][NDIM];
    double usfc[NEQ];
    double fcn[NEQ][NDIM], dfcn[NEQ][NDIM];
#endif
    // interators.
    int icl, ifl, inf, ifc, jcl, ieq, jeq;
    qdt = exd->time_increment * 0.25;
    hdt = exd->time_increment * 0.5;
#ifndef __CUDACC__
    for (icl=istart; icl<iend; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
        // initialize fluxes.
        for (ieq=0; ieq<NEQ; ieq++) {
            futo[ieq] = 0.0;
        };

        pclfcs = exd->clfcs + icl*(CLMFC+1);
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ifl++) {
            ifc = pclfcs[ifl];
            // face node coordinates.
            pfcnds = exd->fcnds + ifc*(FCMND+1);
            fcnnd = pfcnds[0];
            for (inf=0; inf<fcnnd; inf++) {
                pndcrd = exd->ndcrd + pfcnds[inf+1]*NDIM;
                crd[inf][0] = pndcrd[0];
                crd[inf][1] = pndcrd[1];
#if NDIM == 3
                crd[inf][2] = pndcrd[2];
#endif
            };
            crd[fcnnd][0] = crd[0][0];
            crd[fcnnd][1] = crd[0][1];
#if NDIM == 3
            crd[fcnnd][2] = crd[0][2];
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
            // calculate geometric center of the bounding sub-face.
            for (inf=0; inf<fcnnd; inf++) {
                sfcnd[inf][0] = cnde[0] + crd[inf][0];
#if NDIM == 3
                sfcnd[inf][0] += crd[inf+1][0];
#endif
                sfcnd[inf][0] /= NDIM;
                sfcnd[inf][1] = cnde[1] + crd[inf][1];
#if NDIM == 3
                sfcnd[inf][1] += crd[inf+1][1];
#endif
                sfcnd[inf][1] /= NDIM;
#if NDIM == 3
                sfcnd[inf][2] = cnde[2] + crd[inf][2] + crd[inf+1][2];
                sfcnd[inf][2] /= NDIM;
#endif
            };
            // calculate outward area vector of the bounding sub-face.
#if NDIM == 3
            voe = (pfccls[0] - icl) + SOLVCON_ALMOST_ZERO;
            voe /= (icl - pfccls[0]) + SOLVCON_ALMOST_ZERO;
            voe *= 0.5;
            for (inf=0; inf<fcnnd; inf++) {
                disu0 = crd[inf  ][0] - cnde[0];
                disu1 = crd[inf  ][1] - cnde[1];
                disu2 = crd[inf  ][2] - cnde[2];
                disv0 = crd[inf+1][0] - cnde[0];
                disv1 = crd[inf+1][1] - cnde[1];
                disv2 = crd[inf+1][2] - cnde[2];
                sfnml[inf][0] = (disu1*disv2 - disu2*disv1) * voe;
                sfnml[inf][1] = (disu2*disv0 - disu0*disv2) * voe;
                sfnml[inf][2] = (disu0*disv1 - disu1*disv0) * voe;
            };
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
                fusp[ieq] = pjsol[ieq];
                fusp[ieq] += (pcecnd[0]-pjcecnd[0]) * pdsol[0];
                fusp[ieq] += (pcecnd[1]-pjcecnd[1]) * pdsol[1];
#if NDIM == 3
                fusp[ieq] += (pcecnd[2]-pjcecnd[2]) * pdsol[2];
#endif
                pdsol += NDIM;
            };
            pcevol = exd->cevol + icl*(CLMFC+1)+ifl;
            for (ieq=0; ieq<NEQ; ieq++) {
                fusp[ieq] *= pcevol[0];
            };

            // temporal flux (give space).
#ifndef __CUDACC__
            exd->jacofunc(exd, jcl, (double *)fcn, (double *)jacos);
#else
            cuda_calc_jaco(exd, jcl, fcn, jacos);
#endif
            pjsolt = exd->solt + jcl*NEQ;
            for (ieq=0; ieq<NEQ; ieq++) futm[ieq] = 0.0;
            for (inf=0; inf<fcnnd; inf++) {
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
                        dfcn[ieq][0] += jacos[ieq*NEQ+jeq][0] * usfc[jeq];
                        dfcn[ieq][1] += jacos[ieq*NEQ+jeq][1] * usfc[jeq];
#if NDIM == 3
                        dfcn[ieq][2] += jacos[ieq*NEQ+jeq][2] * usfc[jeq];
#endif
                    };
                };
                // temporal flux.
                for (ieq=0; ieq<NEQ; ieq++) {
                    futm[ieq] += dfcn[ieq][0] * sfnml[inf][0];
                    futm[ieq] += dfcn[ieq][1] * sfnml[inf][1];
#if NDIM == 3
                    futm[ieq] += dfcn[ieq][2] * sfnml[inf][2];
#endif
                };
            };

            // sum fluxes.
            for (ieq=0; ieq<NEQ; ieq++) {
                futo[ieq] += fusp[ieq] - hdt*futm[ieq];
            };
        };

        // update solutions.
        psoln = exd->soln + icl*NEQ;
        pcevol = exd->cevol + icl*(CLMFC+1);
        for (ieq=0; ieq<NEQ; ieq++) {
            psoln[ieq] = futo[ieq] / pcevol[0];
        };
    };
#ifndef __CUDACC__
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
#else
    free(futo);
    free(fusp);
    free(futm);
    free(jacos);
    free(usfc);
    free(fcn);
    free(dfcn);
};
extern "C" int calc_soln(int nthread, exedata *exc, void *gexc) {
    int nblock = (exc->ncell + nthread-1) / nthread;
    cuda_calc_soln<<<nblock, nthread>>>((exedata *)gexc);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:
