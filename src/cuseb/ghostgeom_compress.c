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

// TODO: make it openmp.
int ghostgeom_compress(exedata *exd, int nbnd, int *facn) {
    int clnfc, fcnnd;
    // pointers.
    int *pfacn, *pfccls, *pfcnds, *pclfcs;
    double *pclcnd, *pndcrd, *pfccnd, *pfcnml, *pcevol, *pcecnd, *p1cecnd;
    // scalars.
    double vola, vole, dis;
    // arrays.
    double crd[FCMND+2][NDIM];
#if NDIM == 3
    double cnd[NDIM];
    double disu0, disu1, disu2, disv0, disv1, disv2;
    double dist0, dist1, dist2, disw0, disw1, disw2;
#endif
    // iterators.
    int ibnd, ifc, icl, jcl, tfl, inf, ifl;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];

        // determine tfl.
        pclfcs = exd->clfcs + icl*(CLMFC+1);
        clnfc = pclfcs[0];
        for (tfl=1; tfl<=clnfc; tfl++) {
            if (ifc == pclfcs[tfl]) break;
        };

        // get coordinates.
        pfcnds = exd->fcnds + ifc*(FCMND+1);
        fcnnd = pfcnds[0];
        for (inf=1; inf<=fcnnd; inf++) {
            pndcrd = exd->ndcrd + pfcnds[inf] * NDIM;
            crd[inf][0] = pndcrd[0];
            crd[inf][1] = pndcrd[1];
#if NDIM == 3
            crd[inf][2] = pndcrd[2];
#endif
        };
        crd[inf+1][0] = crd[1][0];
        crd[inf+1][1] = crd[1][1];
#if NDIM == 3
        crd[inf+1][2] = crd[1][2];
#endif

        // set ghost CE volume.
        pclcnd = exd->clcnd + jcl * NDIM;
        crd[0][0] = pclcnd[0];
        crd[0][1] = pclcnd[1];
#if NDIM == 3
        crd[0][2] = pclcnd[2];
#endif
#if NDIM == 3
        pfccnd = exd->fccnd + ifc*NDIM;
        vole = cnd[0] = cnd[1] = cnd[2] = 0.0;
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
            disw0 = crd[0][0] - pfccnd[0];
            disw1 = crd[0][1] - pfccnd[1];
            disw2 = crd[0][2] - pfccnd[2];
            vola = abs(dist0*disw0 + dist1*disw1 + dist2*disw2) / 6;
            // accumulate volume and centroid for BCE.
            vole += vola;
            cnd[0] += (crd[inf][0]+crd[inf+1][0]+pfccnd[0] + crd[0][0])/4*vola;
            cnd[1] += (crd[inf][1]+crd[inf+1][1]+pfccnd[1] + crd[0][1])/4*vola;
            cnd[2] += (crd[inf][2]+crd[inf+1][2]+pfccnd[2] + crd[0][2])/4*vola;
        };
        cnd[0] /= vole;
        cnd[1] /= vole;
        cnd[2] /= vole;
#else
        vole = fabs((crd[1][0]-crd[0][0])*(crd[2][1]-crd[0][1])
                  - (crd[1][1]-crd[0][1])*(crd[2][0]-crd[0][0])) / 2.0;
#endif
        pcevol = exd->cevol + icl * (CLMFC+1);
        pcevol[0] -= vole;
        pcevol[tfl] -= vole;

        // set ghost cell center.
        pcecnd = exd->cecnd + icl * (CLMFC+1) * NDIM;
        pfccnd = exd->fccnd + ifc * NDIM;
        pfcnml = exd->fcnml + ifc * NDIM;
        dis = pfcnml[0]*(pfccnd[0]-pcecnd[0])
            + pfcnml[1]*(pfccnd[1]-pcecnd[1])
#if NDIM == 3
            + pfcnml[2]*(pfccnd[2]-pcecnd[2])
#endif
            ;
        crd[0][0] = pfcnml[0]*dis + pcecnd[0];
        crd[0][1] = pfcnml[1]*dis + pcecnd[1];
#if NDIM == 3
        crd[0][2] = pfcnml[2]*dis + pcecnd[2];
#endif
        pclcnd[0] = crd[0][0];
        pclcnd[1] = crd[0][1];
#if NDIM == 3
        pclcnd[2] = crd[0][2];
#endif

        // set ghost solution points.
        pcecnd = exd->cecnd + jcl * (CLMFC+1) * NDIM;
        pcecnd[0] = crd[0][0];
        pcecnd[1] = crd[0][1];
#if NDIM == 3
        pcecnd[2] = crd[0][2];
#endif

        // reset interior center of BCE.
        pclcnd = exd->clcnd + icl * NDIM;
        pcecnd = exd->cecnd + (icl * (CLMFC+1) + tfl) * NDIM;
#if NDIM == 3
        pcecnd[0] = (pcecnd[0]*(pcevol[tfl]+vole) - cnd[0]*vole) / pcevol[tfl];
        pcecnd[1] = (pcecnd[1]*(pcevol[tfl]+vole) - cnd[1]*vole) / pcevol[tfl];
        pcecnd[2] = (pcecnd[2]*(pcevol[tfl]+vole) - cnd[2]*vole) / pcevol[tfl];
#else
        pcecnd[0] = (pclcnd[0] + crd[1][0] + crd[2][0])/3.0;
        pcecnd[1] = (pclcnd[1] + crd[1][1] + crd[2][1])/3.0;
#endif

        // reset interior solution point.
        pcecnd = exd->cecnd + icl * (CLMFC+1) * NDIM;
        pcecnd[0] = 0.0;
        pcecnd[1] = 0.0;
#if NDIM == 3
        pcecnd[2] = 0.0;
#endif
#if NDIM == 3
        pcecnd[0] = (pcecnd[0]*(pcevol[0]+vole) - cnd[0]*vole) / pcevol[0];
        pcecnd[1] = (pcecnd[1]*(pcevol[0]+vole) - cnd[1]*vole) / pcevol[0];
        pcecnd[2] = (pcecnd[2]*(pcevol[0]+vole) - cnd[2]*vole) / pcevol[0];
#else
        p1cecnd = pcecnd + NDIM;
        for (ifl=1; ifl<=clnfc; ifl++) {
            pcecnd[0] += p1cecnd[0] * pcevol[ifl];
            pcecnd[1] += p1cecnd[1] * pcevol[ifl];
            p1cecnd += NDIM;
        };
        pcecnd[0] /= pcevol[0];
        pcecnd[1] /= pcevol[0];
#endif

        // advance boundary face.
        pfacn += 3;
    };
    return 0;
}
// vim: set ts=4 et:
