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
__global__ void cuda_calc_dsoln(exedata *exd) {
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
    // Two-/three-dimensional GGE definition (in c-tau scheme).
    const int ggefcs[31][3] = {
        // quadrilaterals.
        {1, 2, -1}, {2, 3, -1}, {3, 4, -1}, {4, 1, -1},
        // triangles.
        {1, 2, -1}, {2, 3, -1}, {3, 1, -1},
        // hexahedra.
        {2, 3, 5}, {6, 3, 2}, {4, 3, 6}, {5, 3, 4},
        {5, 1, 2}, {2, 1, 6}, {6, 1, 4}, {4, 1, 5},
        // tetrahedra.
        {3, 1, 2}, {2, 1, 4}, {4, 1, 3}, {2, 4, 3},
        // prisms.
        {5, 2, 4}, {3, 2, 5}, {4, 2, 3},
        {4, 1, 5}, {5, 1, 3}, {3, 1, 4},
        // pyramids
        {1, 5, 2}, {2, 5, 3}, {3, 5, 4}, {4, 5, 1},
        {1, 3, 4}, {3, 1, 2},
    };
    const int ggerng[8][2] = {
        {-1, -1}, {-2, -1}, {0, 4}, {4, 7},
        {7, 15}, {15, 19}, {19, 25}, {25, 31},
        //{0, 8}, {8, 12}, {12, 18}, {18, 24},
    };
#else
int calc_dsoln(exedata *exd, int istart, int iend) {
    struct tms timm0, timm1;
    int cputicks;
    times(&timm0);
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    int clnfc;
    // pointers.
    int *pcltpn;
    int *pclfcs, *pfccls;
    double *pcecnd, *picecnd, *pjcecnd;
    double *pisoln, *pjsol, *pjsoln, *pdsol, *pdsoln;
    double *pjsolt;
    // scalars.
    double tau, hdt, vob, voc, wgt;
    // arrays.
    double xps[CLMFC][NDIM], dsp[CLMFC][NDIM];
    double crd[NDIM], cnd[NDIM], cndge[NDIM], sft[NDIM];
#ifdef __CUDACC__
    double *deno;
    double (*nume)[NDIM];
    deno = (double *)malloc(NEQ*sizeof(double));
    nume = (double (*)[NDIM])malloc(NEQ*NDIM*sizeof(double));
#else
    double deno[NEQ];
    double nume[NEQ][NDIM];
#endif
    double grd[NDIM];
    double dst[NDIM][NDIM];
    double dnv[NDIM][NDIM];
#ifdef __CUDACC__
    double (*udf)[NDIM];
    udf = (double (*)[NDIM])malloc(NEQ*NDIM*sizeof(double));
#else
    double udf[NEQ][NDIM];
#endif
    // interators.
    int icl, ifl, ifl1, ifc, jcl, ieq, ivx;
    int ig0, ig1, ig;
    hdt = exd->time_increment * 0.5;
#ifdef __CUDACC__
    icl = istart;
    if (icl < exd->ncell) {
#else
    for (icl=istart; icl<iend; icl++) {
#endif
        pcltpn = exd->cltpn + icl;
        ig0 = ggerng[pcltpn[0]][0];
        ig1 = ggerng[pcltpn[0]][1];
        pclfcs = exd->clfcs + icl*(CLMFC+1);

        // determine tau.
#ifdef __CUDACC__
        tau = exd->taumin + fabs(exd->cfl[icl]) * exd->tauscale;
#else
        tau = exd->taufunc(exd, icl);
#endif

        // calculate the vertices of GGE with the tau parameter.
        pclfcs = exd->clfcs + icl*(CLMFC+1);
        picecnd = exd->cecnd + icl*(CLMFC+1)*NDIM;
        pcecnd = picecnd;
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ifl++) {
            ifl1 = ifl - 1;
            ifc = pclfcs[ifl];
            pfccls = exd->fccls + ifc*FCREL;
            jcl = pfccls[0] + pfccls[1] - icl;
            pjcecnd = exd->cecnd + jcl*(CLMFC+1)*NDIM;
            pcecnd += NDIM;
            // location of P/M points and displacement to neighboring solution
            // points.
            sft[0] = (picecnd[0] + pjcecnd[0])/2;
            sft[0] += exd->cnbfac*(pcecnd[0] - sft[0]);
            xps[ifl1][0] = (pjcecnd[0] - sft[0])*tau + sft[0];
            dsp[ifl1][0] = xps[ifl1][0] - pjcecnd[0];
            sft[1] = (picecnd[1] + pjcecnd[1])/2;
            sft[1] += exd->cnbfac*(pcecnd[1] - sft[1]);
            xps[ifl1][1] = (pjcecnd[1] - sft[1])*tau + sft[1];
            dsp[ifl1][1] = xps[ifl1][1] - pjcecnd[1];
#if NDIM == 3
            sft[2] = (picecnd[2] + pjcecnd[2])/2;
            sft[2] += exd->cnbfac*(pcecnd[2] - sft[2]);
            xps[ifl1][2] = (pjcecnd[2] - sft[2])*tau + sft[2];
            dsp[ifl1][2] = xps[ifl1][2] - pjcecnd[2];
#endif
        };

        // calculate average point.
        crd[0] = crd[1] = 0.0;
#if NDIM == 3
        crd[2] = 0.0;
#endif
        for (ifl=0; ifl<clnfc; ifl++) {
            crd[0] += xps[ifl][0];
            crd[1] += xps[ifl][1];
#if NDIM == 3
            crd[2] += xps[ifl][2];
#endif
        };
        crd[0] /= clnfc;
        crd[1] /= clnfc;
#if NDIM == 3
        crd[2] /= clnfc;
#endif
        // calculate GGE centroid.
        voc = cndge[0] = cndge[1] = 0.0;
#if NDIM == 3
        cndge[2] = 0.0;
#endif
        for (ig=ig0; ig<ig1; ig++) {
            cnd[0] = crd[0];
            cnd[1] = crd[1];
#if NDIM == 3
            cnd[2] = crd[2];
#endif
            for (ivx=0; ivx<NDIM; ivx++) {
                ifl = ggefcs[ig][ivx]-1;
                cnd[0] += xps[ifl][0];
                cnd[1] += xps[ifl][1];
#if NDIM == 3
                cnd[2] += xps[ifl][2];
#endif
                dst[ivx][0] = xps[ifl][0] - crd[0];
                dst[ivx][1] = xps[ifl][1] - crd[1];
#if NDIM == 3
                dst[ivx][2] = xps[ifl][2] - crd[2];
#endif
            };
            cnd[0] /= NDIM+1;
            cnd[1] /= NDIM+1;
#if NDIM == 3
            cnd[2] /= NDIM+1;
            sft[0] = dst[0][1]*dst[1][2] - dst[0][2]*dst[1][1];
            sft[1] = dst[0][2]*dst[1][0] - dst[0][0]*dst[1][2];
            sft[2] = dst[0][0]*dst[1][1] - dst[0][1]*dst[1][0];
            vob = fabs(sft[0]*dst[2][0] + sft[1]*dst[2][1] + sft[2]*dst[2][2]);
            vob /= 6;
#else
            vob = fabs(dst[0][0]*dst[1][1] - dst[0][1]*dst[1][0]);
            vob /= 2;
#endif
            voc += vob;
            cndge[0] += cnd[0] * vob;
            cndge[1] += cnd[1] * vob;
#if NDIM == 3
            cndge[2] += cnd[2] * vob;
#endif
        };
        cndge[0] /= voc;
        cndge[1] /= voc;
#if NDIM == 3
        cndge[2] /= voc;
#endif
        // calculate GGE shift.
        pcecnd = exd->cecnd + icl*(CLMFC+1)*NDIM;
        sft[0] = exd->sftfac * (pcecnd[0] - cndge[0]);
        sft[1] = exd->sftfac * (pcecnd[1] - cndge[1]);
#if NDIM == 3
        sft[2] = exd->sftfac * (pcecnd[2] - cndge[2]);
#endif
        for (ifl=0; ifl<clnfc; ifl++) {
            dsp[ifl][0] += sft[0];
            dsp[ifl][1] += sft[1];
#if NDIM == 3
            dsp[ifl][2] += sft[2];
#endif
        };

        // calculate and weight gradient.
        for (ieq=0; ieq<NEQ; ieq++) {
            deno[ieq] = SOLVCON_ALMOST_ZERO;
            nume[ieq][0] = nume[ieq][1] = 0.0;
#if NDIM == 3
            nume[ieq][2] = 0.0;
#endif
        };
        pisoln = exd->soln + icl*NEQ;
        for (ig=ig0; ig<ig1; ig++) {
            for (ivx=0; ivx<NDIM; ivx++) {
                ifl = ggefcs[ig][ivx];
                ifc = pclfcs[ifl];
                ifl -= 1;
                pfccls = exd->fccls + ifc*FCREL;
                jcl = pfccls[0] + pfccls[1] - icl;
                // distance.
                dst[ivx][0] = xps[ifl][0] - cndge[0];
                dst[ivx][1] = xps[ifl][1] - cndge[1];
#if NDIM == 3
                dst[ivx][2] = xps[ifl][2] - cndge[2];
#endif
                // solution difference.
                pjsol = exd->sol + jcl*NEQ;
                pjsoln = exd->soln + jcl*NEQ;
				pjsolt = exd->solt + jcl*NEQ;
                pdsol = exd->dsol + jcl*NEQ*NDIM;
                for (ieq=0; ieq<NEQ; ieq++) {
                    voc = pjsol[ieq] + hdt*pjsolt[ieq] - pjsoln[ieq];
                    udf[ieq][ivx] = pjsoln[ieq] + exd->taylor*voc - pisoln[ieq]
                                  + dsp[ifl][0]*pdsol[0] + dsp[ifl][1]*pdsol[1];
#if NDIM == 3
                    udf[ieq][ivx] += dsp[ifl][2]*pdsol[2];
#endif
                    pdsol += NDIM;
                };
            };
            // prepare inverse matrix for gradient.
#if NDIM == 3
            dnv[0][0] = dst[1][1]*dst[2][2] - dst[1][2]*dst[2][1];
            dnv[0][1] = dst[0][2]*dst[2][1] - dst[0][1]*dst[2][2];
            dnv[0][2] = dst[0][1]*dst[1][2] - dst[0][2]*dst[1][1];
            dnv[1][0] = dst[1][2]*dst[2][0] - dst[1][0]*dst[2][2];
            dnv[1][1] = dst[0][0]*dst[2][2] - dst[0][2]*dst[2][0];
            dnv[1][2] = dst[0][2]*dst[1][0] - dst[0][0]*dst[1][2];
            dnv[2][0] = dst[1][0]*dst[2][1] - dst[1][1]*dst[2][0];
            dnv[2][1] = dst[0][1]*dst[2][0] - dst[0][0]*dst[2][1];
            dnv[2][2] = dst[0][0]*dst[1][1] - dst[0][1]*dst[1][0];
            voc = dnv[0][2]*dst[2][0] + dnv[1][2]*dst[2][1]
                + dnv[2][2]*dst[2][2];
#else
            dnv[0][0] =  dst[1][1]; dnv[0][1] = -dst[0][1];
            dnv[1][0] = -dst[1][0]; dnv[1][1] =  dst[0][0];
            voc = dst[0][0]*dst[1][1] - dst[0][1]*dst[1][0];
#endif
            // calculate and weight grdient.
            for (ieq=0; ieq<NEQ; ieq++) {
                grd[0] = dnv[0][0]*udf[ieq][0] + dnv[0][1]*udf[ieq][1];
#if NDIM == 3
                grd[0] += dnv[0][2]*udf[ieq][2];
#endif
                grd[0] /= voc;
                grd[1] = dnv[1][0]*udf[ieq][0] + dnv[1][1]*udf[ieq][1];
#if NDIM == 3
                grd[1] += dnv[1][2]*udf[ieq][2];
#endif
                grd[1] /= voc;
#if NDIM == 3
                grd[2] = dnv[2][0]*udf[ieq][0] + dnv[2][1]*udf[ieq][1];
                grd[2] += dnv[2][2]*udf[ieq][2];
                grd[2] /= voc;
#endif
                wgt = grd[0]*grd[0] + grd[1]*grd[1];
#if NDIM == 3
                wgt += grd[2]*grd[2];
#endif
                wgt = 1.0 / pow(sqrt(wgt+SOLVCON_ALMOST_ZERO), exd->alpha);
                deno[ieq] += wgt;
                nume[ieq][0] += wgt*grd[0];
                nume[ieq][1] += wgt*grd[1];
#if NDIM == 3
                nume[ieq][2] += wgt*grd[2];
#endif
            };
        };

        // update grdient.
        pdsoln = exd->dsoln + icl*NEQ*NDIM;
        for (ieq=0; ieq<NEQ; ieq++) {
            pdsoln[0] = nume[ieq][0] / deno[ieq];
            pdsoln[1] = nume[ieq][1] / deno[ieq];
#if NDIM == 3
            pdsoln[2] = nume[ieq][2] / deno[ieq];
#endif
            pdsoln += NDIM;
        };
    };
#ifndef __CUDACC__
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
#else
    free(deno);
    free(nume);
    free(udf);
};
extern "C" int calc_dsoln(int nthread, exedata *exc, void *gexc) {
    int nblock = (exc->ncell + nthread-1) / nthread;
    cuda_calc_dsoln<<<nblock, nthread>>>((exedata *)gexc);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:
