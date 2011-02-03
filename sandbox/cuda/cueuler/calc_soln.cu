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

int calc_soln(exedata *exd, int istart, int iend) {
    struct tms timm0, timm1;
    int cputicks;
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
    times(&timm0);
#ifdef SOLVCESE_FE
    feenableexcept(SOLVCESE_FE);
#endif
    qdt = exd->time_increment * 0.25;
    hdt = exd->time_increment * 0.5;
    for (icl=istart; icl<iend; icl++) {
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
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
// vim: set ts=4 et:
