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

int prepare_ce(exedata *exd) {
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
    int clnfc, fcnnd;
    // pointers.
    int *pclfcs, *pfccls, *pfcnds;
    double *pclcnd, *pfccnd, *pndcrd, *pcevol, *p2cevol, *pcecnd, *p2cecnd;
    // vectors.
    double crdi[NDIM], crde[NDIM];
    double cndi[NDIM], cnde[NDIM];
    double crd[FCMND+2][NDIM];
    // scalars.
    double disu0, disu1, disu2, disv0, disv1, disv2;
    double dist0, dist1, dist2, disw0, disw1, disw2;
    double voli, vole, volb, volc;  // internal, external, BCE, CCE.
    // iterators.
    int icl, jcl, ind, ifl, inf, ifc;
    // loop for cells.
    #pragma omp parallel for private(clnfc, fcnnd, pclfcs, pfccls, pfcnds, \
    crdi, crde, cndi, cnde, crd, \
    disu0, disu1, disu2, disv0, disv1, disv2, \
    dist0, dist1, dist2, disw0, disw1, disw2, \
    voli, vole, volb, volc, \
    icl, jcl, ind, ifl, inf, ifc)
    for (icl=0; icl<exd->ncell; icl++) {
        pcevol = exd->cevol + icl*(CLMFC+1);
        pcecnd = exd->cecnd + icl*(CLMFC+1)*NDIM;
        // self cell center.
        pclcnd = exd->clcnd + icl*NDIM;
        crdi[0] = pclcnd[0];
        crdi[1] = pclcnd[1];
#if NDIM == 3
        crdi[2] = pclcnd[2];
#endif

        // loop for each face in cell.
        p2cevol = pcevol;
        p2cecnd = pcecnd;
        volc = 0.0;
        pcecnd[0] = 0.0;
        pcecnd[1] = 0.0;
#if NDIM == 3
        pcecnd[2] = 0.0;
#endif
        pclfcs = exd->clfcs + icl*(CLMFC+1);
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ifl++) {
            ifc = pclfcs[ifl];
            pfccnd = exd->fccnd + ifc*NDIM;
            pfccls = exd->fccls + ifc*FCREL;
            jcl = pfccls[0] + pfccls[1] - icl;

            // neighbor cell center.
            pclcnd = exd->clcnd + jcl*NDIM;
            crde[0] = pclcnd[0];
            crde[1] = pclcnd[1];
#if NDIM == 3
            crde[2] = pclcnd[2];
#endif

            // node coordinates.
            pfcnds = exd->fcnds + ifc*(FCMND+1);
            fcnnd = pfcnds[0];
            for (inf=1; inf<=fcnnd; inf++) {
                ind = pfcnds[inf];
                pndcrd = exd->ndcrd + ind*NDIM;
                crd[inf][0] = pndcrd[0];
                crd[inf][1] = pndcrd[1];
#if NDIM == 3
                crd[inf][2] = pndcrd[2];
#endif
            };
            crd[fcnnd+1][0] = crd[1][0];
            crd[fcnnd+1][1] = crd[1][1];
#if NDIM == 3
            crd[fcnnd+1][2] = crd[1][2];
#endif

            // calculate volume and center of BCEs and in term CCEs.
            p2cevol += 1;
            p2cecnd += NDIM;
#if NDIM == 3
            volb = 0.0;
            p2cecnd[0] = p2cecnd[1] = p2cecnd[2] = 0.0;
            for (inf=1; inf<=fcnnd; inf++) {
                // base triangle.
                disu0 = crd[inf  ][0] - pfccnd[0];
                disu1 = crd[inf  ][1] - pfccnd[1];
                disu2 = crd[inf  ][2] - pfccnd[2];
                disv0 = crd[inf+1][0] - pfccnd[0];
                disv1 = crd[inf+1][1] - pfccnd[1];
                disv2 = crd[inf+1][2] - pfccnd[2];
                dist0 = disu1*disv2 - disu2*disv1;
                dist1 = disu2*disv0 - disu0*disv2;
                dist2 = disu0*disv1 - disu1*disv0;
                // outer tetrahedron.
                disw0 = crde[0] - pfccnd[0];
                disw1 = crde[1] - pfccnd[1];
                disw2 = crde[2] - pfccnd[2];
                vole = fabs(dist0*disw0 + dist1*disw1 + dist2*disw2) / 6;
                cnde[0] = (crd[inf][0]+crd[inf+1][0]+pfccnd[0] + crde[0]) / 4;
                cnde[1] = (crd[inf][1]+crd[inf+1][1]+pfccnd[1] + crde[1]) / 4;
                cnde[2] = (crd[inf][2]+crd[inf+1][2]+pfccnd[2] + crde[2]) / 4;
                // inner tetrahedron.
                disw0 = crdi[0] - pfccnd[0];
                disw1 = crdi[1] - pfccnd[1];
                disw2 = crdi[2] - pfccnd[2];
                voli = fabs(dist0*disw0 + dist1*disw1 + dist2*disw2) / 6;
                cndi[0] = (crd[inf][0]+crd[inf+1][0]+pfccnd[0] + crdi[0]) / 4;
                cndi[1] = (crd[inf][1]+crd[inf+1][1]+pfccnd[1] + crdi[1]) / 4;
                cndi[2] = (crd[inf][2]+crd[inf+1][2]+pfccnd[2] + crdi[2]) / 4;
                // accumulate volume and centroid for BCE.
                volb += voli + vole;
                p2cecnd[0] += cndi[0]*voli + cnde[0]*vole;
                p2cecnd[1] += cndi[1]*voli + cnde[1]*vole;
                p2cecnd[2] += cndi[2]*voli + cnde[2]*vole;
            };
            volc += volb;
            pcecnd[0] += p2cecnd[0];
            pcecnd[1] += p2cecnd[1];
            pcecnd[2] += p2cecnd[2];
            p2cevol[0] = volb;
            p2cecnd[0] /= volb;
            p2cecnd[1] /= volb;
            p2cecnd[2] /= volb;
#else
            // triangle formed by cell point and two face nodes.
            cndi[0] = (crd[1][0] + crd[2][0] + crdi[0]) / 3;
            cndi[1] = (crd[1][1] + crd[2][1] + crdi[1]) / 3;
            voli = fabs((crd[1][0]-crdi[0])*(crd[2][1]-crdi[1])
                      - (crd[1][1]-crdi[1])*(crd[2][0]-crdi[0])) / 2;
            // triangle formed by neighbor cell point and two face nodes.
            cnde[0] = (crd[1][0] + crd[2][0] + crde[0]) / 3;
            cnde[1] = (crd[1][1] + crd[2][1] + crde[1]) / 3;
            vole = fabs((crd[1][0]-crde[0])*(crd[2][1]-crde[1])
                      - (crd[1][1]-crde[1])*(crd[2][0]-crde[0])) / 2;
            // volume of BCE (quadrilateral) formed by the two triangles.
            volb = voli + vole;
            p2cevol[0] = volb;
            // geometry center of each BCE for cell j.
            p2cecnd[0] = (cndi[0]*voli+cnde[0]*vole)/volb;
            p2cecnd[1] = (cndi[1]*voli+cnde[1]*vole)/volb;
            // volume and geometry center of the CCE for each cell.
            volc += volb;
            pcecnd[0] += p2cecnd[0]*volb;
            pcecnd[1] += p2cecnd[1]*volb;
#endif
        };
        pcevol[0] = volc;
        pcecnd[0] /= volc;
        pcecnd[1] /= volc;
#if NDIM == 3
        pcecnd[2] /= volc;
#endif
    };
    return 0;
};
// vim: set ts=4 et:
