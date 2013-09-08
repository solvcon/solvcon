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
__global__ void cuda_calc_solt(exedata *exd) {
    int istart = -exd->ngstcell + blockDim.x * blockIdx.x + threadIdx.x;
#else
int calc_solt(exedata *exd, int istart, int iend) {
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    // pointers.
    double *psolt, *pidsol, *pdsol;
    // scalars.
    double val;
    // arrays.
    double jacos[NEQ][NEQ][NDIM];
    double fcn[NEQ][NDIM];
    // interators.
    int icl, ieq, jeq, idm;
#ifndef __CUDACC__
    #pragma omp parallel for \
    private(psolt, pidsol, pdsol, val, jacos, fcn, ieq, jeq, idm)
    for (icl=istart; icl<iend; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
        psolt = exd->solt + icl*NEQ;
        pidsol = exd->dsol + icl*NEQ*NDIM;
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
                    val += jacos[ieq][jeq][idm]*pdsol[idm];
                    pdsol += NDIM;
                };
                psolt[ieq] -= val;
            };
        };
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
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
int calc_soln(exedata *exd) {
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    int clnfc, fcnnd;
    // partial pointers.
    int *pclfcs, *pfcnds, *pfccls;
    double *pjcecnd, *pcecnd, *pcevol, (*psfmrc)[NDIM];
    double *pjsol, *pdsol, *pjsolt, *psoln;
    // scalars.
    double hdt, qdt;
    double voe, fusp, futm;
    // arrays.
    double usfc[NEQ];
    double fcn[NEQ][NDIM], dfcn[NEQ][NDIM];
    double jacos[NEQ][NEQ][NDIM];
    double visc[NEQ][NDIM];
    // interators.
    int icl, ifl, inf, ifc, jcl, ieq, jeq;
    // source indicator
    int viscosity;
    qdt = exd->time_increment * 0.25;
    hdt = exd->time_increment * 0.5;
#ifndef __CUDACC__
    #pragma omp parallel for private(clnfc, fcnnd, \
    pclfcs, pfcnds, pfccls, pjcecnd, pcecnd, pcevol, psfmrc, \
    pjsol, pdsol, pjsolt, psoln, \
    voe, fusp, futm, usfc, fcn, dfcn, jacos, \
    icl, ifl, inf, ifc, jcl, ieq, jeq, viscosity, visc) \
    firstprivate(hdt, qdt)
    for (icl=0; icl<exd->ncell; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
        psoln = exd->soln + icl*NEQ;
        pcevol = exd->cevol + icl*(CLMFC+1);
        // initialize fluxes.
        for (ieq=0; ieq<NEQ; ieq++) {
            psoln[ieq] = 0.0;
        };

        pclfcs = exd->clfcs + icl*(CLMFC+1);
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ifl++) {
            ifc = pclfcs[ifl];

            // spatial flux (given time).
            pfccls = exd->fccls + ifc*FCREL;
            jcl = pfccls[0] + pfccls[1] - icl;
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
            viscosity = exd->viscosity;
#ifndef __CUDACC__
            exd->jacofunc(exd, jcl, (double *)fcn, (double *)jacos);
            if (viscosity == 1) exd->viscfunc(exd, jcl, (double *)visc);
#else
            cuda_calc_jaco(exd, jcl, fcn, jacos);
            if (viscosity == 1) cuda_calc_visc(exd, jcl, visc);
#endif
            pjsolt = exd->solt + jcl*NEQ;
            fcnnd = exd->fcnds[ifc*(FCMND+1)];
            for (inf=0; inf<fcnnd; inf++) {
                psfmrc = (double (*)[NDIM])(exd->sfmrc
                    + (((icl*CLMFC + ifl-1)*FCMND+inf)*2*NDIM));
                // solution at sub-face center.
                pdsol = exd->dsol + jcl*NEQ*NDIM;
                for (ieq=0; ieq<NEQ; ieq++) {
                    usfc[ieq] = qdt * pjsolt[ieq];
                    usfc[ieq] += (psfmrc[0][0]-pjcecnd[0]) * pdsol[0];
                    usfc[ieq] += (psfmrc[0][1]-pjcecnd[1]) * pdsol[1];
#if NDIM == 3
                    usfc[ieq] += (psfmrc[0][2]-pjcecnd[2]) * pdsol[2];
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
                    // foc
                    if(viscosity==1 && pcecnd[0]<0.013 && pcecnd[0]>-0.013 && pcecnd[1]<0.013 && pcecnd[1]>-0.013){
                            dfcn[ieq][0] += visc[ieq][0];
                            dfcn[ieq][1] += visc[ieq][1];
#if NDIM == 3  
                            dfcn[ieq][2] += visc[ieq][2];
#endif
                    }
                    else if(viscosity==2 && pcecnd[1]<0.0127) {
                            dfcn[ieq][0] += visc[ieq][0];
                            dfcn[ieq][1] += visc[ieq][1];
#if NDIM == 3  
                            dfcn[ieq][2] += visc[ieq][2];
#endif
                    }
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
                    futm += dfcn[ieq][0] * psfmrc[1][0];
                    futm += dfcn[ieq][1] * psfmrc[1][1];
#if NDIM == 3
                    futm += dfcn[ieq][2] * psfmrc[1][2];
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
    };
    return 0;
};
#else
    };
};
extern "C" int calc_soln(int nthread, exedata *exc, void *gexc) {
    int nblock = (exc->ncell + nthread-1) / nthread;
    cuda_calc_soln<<<nblock, nthread>>>((exedata *)gexc);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ft=cuda ts=4 et:
