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

#include "cueuler.h"

__device__ void cuda_calc_jaco(exedata *exd, int icl,
        double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM]) {
    // pointers.
    double *psol;
    // scalars.
    double ga, ga1, ga3, ga1h;
    double u1, u2, u3, u4;
#if NDIM == 3
    double u5;
#endif
    // accelerating variables.
    double rho2, ke2, g1ke2, vs, gretot, getot, pr, v1o2, v2o2, v1, v2;
#if NDIM == 3
    double v3o2, v3;
#endif

    // initialize values.
    ga = exd->amsca[icl*NSCA];
    ga1 = ga-1;
    ga3 = ga-3;
    ga1h = ga1/2;
    psol = exd->sol + icl*NEQ;
    u1 = psol[0] + SOLVCESE_TINY;
    u2 = psol[1];
    u3 = psol[2];
    u4 = psol[3];
#if NDIM == 3
    u5 = psol[4];
#endif

    // accelerating variables.
    rho2 = u1*u1;
    v1 = u2/u1; v1o2 = v1*v1;
    v2 = u3/u1; v2o2 = v2*v2;
#if NDIM == 3
    v3 = u4/u1; v3o2 = v3*v3;
#endif
    ke2 = (u2*u2 + u3*u3
#if NDIM == 3
        + u4*u4
#endif
    )/u1;
    g1ke2 = ga1*ke2;
    vs = ke2/u1;
    gretot = ga * 
#if NDIM == 3
        u5
#else
        u4
#endif
    ;
    getot = gretot/u1;
    pr = ga1*
#if NDIM == 3
        u5
#else
        u4
#endif
        - ga1h * ke2;

    // flux function.
#if NDIM == 3
    fcn[0][0] = u2; fcn[0][1] = u3; fcn[0][2] = u4;
    fcn[1][0] = pr + u2*v1;
    fcn[1][1] = u2*v2;
    fcn[1][2] = u2*v3;
    fcn[2][0] = u3*v1;
    fcn[2][1] = pr + u3*v2;
    fcn[2][2] = u3*v3;
    fcn[3][0] = u4*v1;
    fcn[3][1] = u4*v2;
    fcn[3][2] = pr + u4*v3;
    fcn[4][0] = (pr + u5)*v1;
    fcn[4][1] = (pr + u5)*v2;
    fcn[4][2] = (pr + u5)*v3;
#else
    fcn[0][0] = u2; fcn[0][1] = u3;
    fcn[1][0] = pr + u2*v1;
    fcn[1][1] = u2*v2;
    fcn[2][0] = u3*v1;
    fcn[2][1] = pr + u3*v2;
    fcn[3][0] = (pr + u4)*v1;
    fcn[3][1] = (pr + u4)*v2;
#endif
 
    // Jacobian matrices.
#if NDIM == 3
    jacos[0][0][0] = 0; jacos[0][0][1] = 0; jacos[0][0][2] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0; jacos[0][1][2] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1; jacos[0][2][2] = 0;
    jacos[0][3][0] = 0; jacos[0][3][1] = 0; jacos[0][3][2] = 1;
    jacos[0][4][0] = 0; jacos[0][4][1] = 0; jacos[0][4][2] = 0;

    jacos[1][0][0] = -v1o2 + ga1h*vs;
    jacos[1][0][1] = -v1*v2;
    jacos[1][0][2] = -v1*v3;
    jacos[1][1][0] = -ga3*v1; jacos[1][1][1] = v2; jacos[1][1][2] = v3;
    jacos[1][2][0] = -ga1*v2; jacos[1][2][1] = v1; jacos[1][2][2] = 0;
    jacos[1][3][0] = -ga1*v3; jacos[1][3][1] = 0;  jacos[1][3][2] = v1;
    jacos[1][4][0] = ga1;     jacos[1][4][1] = 0;  jacos[1][4][2] = 0;

    jacos[2][0][0] = -v2*v1;
    jacos[2][0][1] = -v2o2 + ga1h*vs;
    jacos[2][0][2] = -v2*v3;
    jacos[2][1][0] = v2; jacos[2][1][1] = -ga1*v1; jacos[2][1][2] = 0;
    jacos[2][2][0] = v1; jacos[2][2][1] = -ga3*v2; jacos[2][2][2] = v3;
    jacos[2][3][0] = 0;  jacos[2][3][1] = -ga1*v3; jacos[2][3][2] = v2;
    jacos[2][4][0] = 0;  jacos[2][4][1] = ga1;     jacos[2][4][2] = 0;

    jacos[3][0][0] = -v3*v1;
    jacos[3][0][1] = -v3*v2;
    jacos[3][0][2] = -v3o2 + ga1h*vs;
    jacos[3][1][0] = v3; jacos[3][1][1] = 0;  jacos[3][1][2] = -ga1*v1;
    jacos[3][2][0] = 0;  jacos[3][2][1] = v3; jacos[3][2][2] = -ga1*v2;
    jacos[3][3][0] = v1; jacos[3][3][1] = v2; jacos[3][3][2] = -ga3*v3;
    jacos[3][4][0] = 0;  jacos[3][4][1] = 0;  jacos[3][4][2] = ga1;

    jacos[4][0][0] = (-gretot + g1ke2)*u2/rho2;
    jacos[4][0][1] = (-gretot + g1ke2)*u3/rho2;
    jacos[4][0][2] = (-gretot + g1ke2)*u4/rho2;
    jacos[4][1][0] = getot - ga1h*(vs + 2*v1o2);
    jacos[4][1][1] = -ga1*v1*v2;
    jacos[4][1][2] = -ga1*v1*v3;
    jacos[4][2][0] = -ga1*v2*v1;
    jacos[4][2][1] = getot - ga1h*(vs + 2*v2o2);
    jacos[4][2][2] = -ga1*v2*v3;
    jacos[4][3][0] = -ga1*v3*v1;
    jacos[4][3][1] = -ga1*v3*v2;
    jacos[4][3][2] = getot - ga1h*(vs + 2*v3o2);
    jacos[4][4][0] = ga*v1; jacos[4][4][1] = ga*v2; jacos[4][4][2] = ga*v3;
#else
    jacos[0][0][0] = 0; jacos[0][0][1] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1;
    jacos[0][3][0] = 0; jacos[0][3][1] = 0;

    jacos[1][0][0] = -v1o2 + ga1h*vs;
    jacos[1][0][1] = -v1*v2;
    jacos[1][1][0] = -ga3*v1; jacos[1][1][1] = v2;
    jacos[1][2][0] = -ga1*v2; jacos[1][2][1] = v1;
    jacos[1][3][0] = ga1;     jacos[1][3][1] = 0;

    jacos[2][0][0] = -v2*v1;
    jacos[2][0][1] = -v2o2 + ga1h*vs;
    jacos[2][1][0] = v2; jacos[2][1][1] = -ga1*v1;
    jacos[2][2][0] = v1; jacos[2][2][1] = -ga3*v2;
    jacos[2][3][0] = 0;  jacos[2][3][1] = ga1;

    jacos[3][0][0] = (-gretot + g1ke2)*u2/rho2;
    jacos[3][0][1] = (-gretot + g1ke2)*u3/rho2;
    jacos[3][1][0] = getot - ga1h*(vs + 2*v1o2);
    jacos[3][1][1] = -ga1*v1*v2;
    jacos[3][2][0] = -ga1*v2*v1;
    jacos[3][2][1] = getot - ga1h*(vs + 2*v2o2);
    jacos[3][3][0] = ga*v1; jacos[3][3][1] = ga*v2;
#endif

    return;
};

__global__ void cuda_calc_solt(exedata *exd) {
    // pointers.
    double *psolt, *pidsol, *pdsol;
    // scalars.
    double val;
    // arrays.
    double jacos[NEQ][NEQ][NDIM];
    double fcn[NEQ][NDIM];
    // interators.
    int icl, ieq, jeq, idm;
#ifdef __CUDACC__
    // CUDA thread control.
    int istart = -exd->ngstcell + blockDim.x * blockIdx.x + threadIdx.x;
#endif
    psolt = exd->solt + istart*NEQ;
    pidsol = exd->dsol + istart*NEQ*NDIM;
#ifndef __CUDACC__
    for (icl=istart; icl<iend; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
        cuda_calc_jaco(exd, icl, fcn, jacos);
        for (ieq=0; ieq<NEQ; ieq++) {
            psolt[ieq] = 0.0;
            for (idm=0; idm<NDIM; idm++) {
                val = 0.0;
                pdsol = pidsol;
                for (jeq=0; jeq<NEQ; jeq++) {
                    val += jacos[ieq][jeq][idm]*pdsol[idm];
                    pdsol += NDIM;
                };
                psolt[ieq] -= val;
            };
        };
#ifndef __CUDACC__
        // advance pointers.
        psolt += NEQ;
        pidsol += NEQ*NDIM;
#endif
    };
};
#ifdef __CUDACC__
extern "C" int calc_solt(int nthread, exedata *exc, void *gexc) {
    dim3 nblock = (exc->ngstcell + exc->ncell + nthread-1) / nthread;
    cuda_calc_solt<<<nblock, nthread>>>((exedata *)gexc);
    cudaThreadSynchronize();
    return 0;
};
#endif

#ifdef __CUDACC__
__global__ void cuda_calc_soln(exedata *exd) {
    int istart = -exd->ngstcell + blockDim.x * blockIdx.x + threadIdx.x;
#else
int calc_soln(exedata *exd, int istart, int iend) {
    struct tms timm0, timm1;
    int cputicks;
    times(&timm0);
#ifdef SOLVCESE_FE
    feenableexcept(SOLVCESE_FE);
#endif
#endif
    int clnfc, fcnnd;
    // partial pointers.
    int *pclfcs, *pfcnds, *pfccls;
    double *pndcrd, *pfccnd, *pclcnd;
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
    double futo[NEQ];
    double fusp[NEQ];
    double futm[NEQ];
    double jacos[NEQ][NEQ][NDIM];
    double usfc[NEQ];
    double fcn[NEQ][NDIM];
    double dfcn[NEQ][NDIM];
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
            voe = (pfccls[0] - icl) + SOLVCESE_ALMOST_ZERO;
            voe /= (icl - pfccls[0]) + SOLVCESE_ALMOST_ZERO;
            voe *= 0.5;
            pfccnd = exd->fccnd + ifc*NDIM;
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
            exd->jacofunc(exd, jcl, (double *)fcn, (double *)jacos);
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
                        dfcn[ieq][0] += jacos[ieq][jeq][0] * usfc[jeq];
                        dfcn[ieq][1] += jacos[ieq][jeq][1] * usfc[jeq];
#if NDIM == 3
                        dfcn[ieq][2] += jacos[ieq][jeq][2] * usfc[jeq];
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
};
extern "C" int calc_soln(int nthread, exedata *exc, void *gexc) {
    dim3 nblock = (exc->ngstcell + exc->ncell + nthread-1) / nthread;
    cuda_calc_soln<<<nblock, nthread>>>((exedata *)gexc);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:
