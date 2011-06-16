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

int prepare_sf(exedata *exd) {
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
    int clnfc, fcnnd;
    // partial pointers.
    int *pclfcs, *pfcnds, *pfccls;
    double *pndcrd, *pclcnd, *pjcecnd, *pcecnd, (*psfmrc)[2][NDIM];
    // scalars.
    double voe, disu0, disu1, disu2, disv0, disv1, disv2;
    // arrays.
    double crd[FCMND+1][NDIM], cnde[NDIM];
    // interators.
    int icl, ifl, inf, ifc, jcl;
    #pragma omp parallel for private(clnfc, fcnnd, \
    pclfcs, pfcnds, pfccls, pndcrd, pclcnd, pjcecnd, pcecnd, psfmrc, \
    voe, disu0, disu1, disu2, disv0, disv1, disv2, crd, cnde, \
    icl, ifl, inf, ifc, jcl)
    for (icl=0; icl<exd->ncell; icl++) {
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
            psfmrc = (double (*)[2][NDIM])(exd->sfmrc
                + ((icl*CLMFC + ifl-1)*FCMND*2*NDIM));
            for (inf=0; inf<fcnnd; inf++) {
                psfmrc[inf][0][0] = cnde[0] + crd[inf][0];
#if NDIM == 3
                psfmrc[inf][0][0] += crd[inf+1][0];
#endif
                psfmrc[inf][0][0] /= NDIM;
                psfmrc[inf][0][1] = cnde[1] + crd[inf][1];
#if NDIM == 3
                psfmrc[inf][0][1] += crd[inf+1][1];
#endif
                psfmrc[inf][0][1] /= NDIM;
#if NDIM == 3
                psfmrc[inf][0][2] = cnde[2] + crd[inf][2];
                psfmrc[inf][0][2] += crd[inf+1][2];
                psfmrc[inf][0][2] /= NDIM;
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
                psfmrc[inf][1][0] = (disu1*disv2 - disu2*disv1) * voe;
                psfmrc[inf][1][1] = (disu2*disv0 - disu0*disv2) * voe;
                psfmrc[inf][1][2] = (disu0*disv1 - disu1*disv0) * voe;
            };
#else
            voe = (crd[0][0]-cnde[0])*(crd[1][1]-cnde[1])
                - (crd[0][1]-cnde[1])*(crd[1][0]-cnde[0]);
            voe /= fabs(voe);
            psfmrc[0][1][0] = -(cnde[1]-crd[0][1]) * voe;
            psfmrc[0][1][1] =  (cnde[0]-crd[0][0]) * voe;
            psfmrc[1][1][0] =  (cnde[1]-crd[1][1]) * voe;
            psfmrc[1][1][1] = -(cnde[0]-crd[1][0]) * voe;
#endif
        };
    };
    return 0;
};

// vim: set ft=cuda ts=4 et:
