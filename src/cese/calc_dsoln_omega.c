/*
 * Copyright (C) 2008-2010 Yung-Yu Chen <yyc@solvcon.net>.
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

#include "cese.h"

int calc_dsoln_omega(exedata *exd, int istart, int iend) {
    int cputicks;
    struct tms timm0, timm1;
    int clnfc, fpn;
    // pointers.
    int *pclfcs, *pfccls, *pfcnds;
    double *pndcrd;
    double *picecnd, *pjcecnd;
    double *pisoln, *pjsol, *pjsoln, *pjsolt, *pdsol, *pdsoln;
    // scalars.
    double omega, hdt, vob, voc, wgt;
    // arrays.
    double cndge[NDIM], grd[NDIM], sft[NDIM];
    double deno[NEQ];
    double nume[NEQ][NDIM];
    double dst[NDIM][NDIM];
    double dnv[NDIM][NDIM];
    double udf[NEQ][NDIM];
    // interators.
    int icl, ifl, ifc, inf, ivx, jcl, ind, ieq;
    int isf, isf0, isf1;
    times(&timm0);
#ifdef SOLVCESE_FE
    feenableexcept(SOLVCESE_FE);
#endif
    hdt = exd->time_increment * 0.5;
    for (icl=istart; icl<iend; icl++) {
        // determine omega.
        omega = exd->omegafunc(exd, icl);

        // calculate GGE centroid.
        voc = cndge[0] = cndge[1] = 0.0;
#if NDIM == 3
        cndge[2] = 0.0;
#endif
        pclfcs = exd->clfcs + icl*(CLMFC+1);
        picecnd = exd->cecnd + icl*(CLMFC+1)*NDIM;
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ifl++) {
            ifc = pclfcs[ifl];
            fpn = exd->fctpn[ifc];
            isf0 = sfng[fpn][0];
            isf1 = sfng[fpn][1];
            pfcnds = exd->fcnds + ifc*(FCMND+1);
            for (isf=isf0; isf<isf1; isf++) {
                grd[0] = grd[1] = 0;
#if NDIM == 3
                grd[2] = 0;
#endif
                // NOTE: grd means centroid of sub-simplex in this loop.
                for (ivx=0; ivx<NDIM; ivx++) {
                    ind = pfcnds[sfcs[isf][ivx]];
                    pndcrd = exd->ndcrd + ind*NDIM;
                    dst[ivx][0] = omega * (pndcrd[0] - picecnd[0]);
                    grd[0] += dst[ivx][0];
                    dst[ivx][1] = omega * (pndcrd[1] - picecnd[1]);
                    grd[1] += dst[ivx][1];
#if NDIM == 3
                    dst[ivx][2] = omega * (pndcrd[2] - picecnd[2]);
                    grd[2] += dst[ivx][2];
#endif
                };
                grd[0] /= NDIM;
                grd[0] += picecnd[0];
                grd[1] /= NDIM;
                grd[1] += picecnd[1];
#if NDIM == 3
                grd[2] /= NDIM;
                grd[2] += picecnd[2];
                sft[0] = dst[0][1]*dst[1][2] - dst[0][2]*dst[1][1];
                sft[1] = dst[0][2]*dst[1][0] - dst[0][0]*dst[1][2];
                sft[2] = dst[0][0]*dst[1][1] - dst[0][1]*dst[1][0];
                // NOTE: sft is only temp array for cross product.
                vob = fabs(sft[0]*dst[2][0]+sft[1]*dst[2][1]+sft[2]*dst[2][2]);
                vob /= 6;
#else
                vob = fabs(dst[0][0]*dst[1][1] - dst[0][1]*dst[1][0]);
                vob /= 2;
#endif
                voc += vob;
                cndge[0] += grd[0] * vob;
                cndge[1] += grd[1] * vob;
#if NDIM == 3
                cndge[2] += grd[2] * vob;
#endif
            };
        };
        // calculate centroid shift.
        cndge[0] /= voc;
        cndge[0] = exd->sftfac * (picecnd[0] - cndge[0]);
        cndge[1] /= voc;
        cndge[1] = exd->sftfac * (picecnd[1] - cndge[1]);
#if NDIM == 3
        cndge[2] /= voc;
        cndge[2] = exd->sftfac * (picecnd[2] - cndge[2]);
#endif
        // NOTE: cndge changed meaning to centroid shift.

        // calculate and weight gradient.
        for (ieq=0; ieq<NEQ; ieq++) {
            deno[ieq] = SOLVCESE_ALMOST_ZERO;
            nume[ieq][0] = nume[ieq][1] = 0.0;
#if NDIM == 3
            nume[ieq][2] = 0.0;
#endif
        };
        pclfcs = exd->clfcs + icl*(CLMFC+1);
        pisoln = exd->soln + icl*NEQ;
        for (ifl=1; ifl<=clnfc; ifl++) {
            ifc = pclfcs[ifl];
            fpn = exd->fctpn[ifc];
            isf0 = sfng[fpn][0];
            isf1 = sfng[fpn][1];
            pfcnds = exd->fcnds + ifc*(FCMND+1);
            pfccls = exd->fccls + ifc*FCREL;
            jcl = pfccls[0] + pfccls[1] - icl;
            pjcecnd = exd->cecnd + jcl*(CLMFC+1)*NDIM;
            pjsol = exd->sol + jcl*NEQ;
            pjsoln = exd->soln + jcl*NEQ;
            pjsolt = exd->solt + jcl*NEQ;
            for (isf=isf0; isf<isf1; isf++) {
                for (ivx=0; ivx<NDIM; ivx++) {
                    ind = pfcnds[sfcs[isf][ivx]];
                    pndcrd = exd->ndcrd + ind*NDIM;
                    // distance.
                    dst[ivx][0] = omega * (pndcrd[0] - picecnd[0]);
                    dst[ivx][1] = omega * (pndcrd[1] - picecnd[1]);
#if NDIM == 3
                    dst[ivx][2] = omega * (pndcrd[2] - picecnd[2]);
#endif
                    // solution difference.
                    sft[0] = cndge[0] + picecnd[0]+dst[ivx][0] - pjcecnd[0];
                    sft[1] = cndge[1] + picecnd[1]+dst[ivx][1] - pjcecnd[1];
#if NDIM == 3
                    sft[2] = cndge[2] + picecnd[2]+dst[ivx][2] - pjcecnd[2];
#endif
                    pdsol = exd->dsol + jcl*NEQ*NDIM;
                    for (ieq=0; ieq<NEQ; ieq++) {
                        voc = pjsol[ieq] + hdt*pjsolt[ieq] - pjsoln[ieq];
                        udf[ieq][ivx] = pjsoln[ieq] + exd->taylor*voc
                                      - pisoln[ieq]
                                      + sft[0]*pdsol[0] + sft[1]*pdsol[1];
#if NDIM == 3
                        udf[ieq][ivx] += sft[2]*pdsol[2];
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
                    wgt = 1.0 / pow(sqrt(wgt+SOLVCESE_ALMOST_ZERO),
                                    exd->alpha);
                    deno[ieq] += wgt;
                    nume[ieq][0] += wgt*grd[0];
                    nume[ieq][1] += wgt*grd[1];
#if NDIM == 3
                    nume[ieq][2] += wgt*grd[2];
#endif
                };
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
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
// vim: set ts=4 et:
