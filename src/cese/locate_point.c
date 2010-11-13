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

void *locate_point(exedata *exd, double *crd,
        int *picl, int *pifl, int *pjcl, int *pjfl) {
    int clnfc, fpn;
    // pointers.
    int *pclfcs, *pfccls, *pfcnds;
    double *pclcnd, *pndcrd;
    // arrays.
    double v0[NDIM], v1[NDIM], v2[NDIM], bcc[NDIM];
#if NDIM == 3
    double v3[NDIM];
    double rhs[NDIM];
    double mat00, mat01, mat02, mat10, mat11, mat12, mat20, mat21, mat22;
    double mai00, mai01, mai02, mai10, mai11, mai12, mai20, mai21, mai22;
    double vol;
#endif
    // scalars.
    double v1s, v2s, v01, v02, v12;
    double bca;
    // iterators.
    int icl, ifl, ifc, jcl;
    int it0, it1, it;
    picl[0] = -1;
    pifl[0] = -1;
    pjcl[0] = -1;
    pjfl[0] = -1;
    // loop over cells.
    for (icl=0; icl<exd->ncell; icl++) {
        pclfcs = exd->clfcs + icl * (CLMFC+1);
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ifl++) {
            ifc = pclfcs[ifl];
            fpn = exd->fctpn[ifc];
            it0 = sfng[fpn][0];
            it1 = sfng[fpn][1];
            pfcnds = exd->fcnds + ifc * (FCMND+1);
            pfccls = exd->fccls + ifc * 4;
            jcl = pfccls[0] + pfccls[1] - icl;
            // search.
            for (it=it0; it<it1; it++) {
                // inner triangle.
                pclcnd = exd->clcnd + icl * NDIM;
                v0[0] = crd[0] - pclcnd[0]; // line AP.
                v0[1] = crd[1] - pclcnd[1];
#if NDIM == 3
                v0[2] = crd[2] - pclcnd[2];
#endif
                pndcrd = exd->ndcrd + pfcnds[sfcs[it][0]] * NDIM;
                v1[0] = pndcrd[0] - pclcnd[0];  // line AB.
                v1[1] = pndcrd[1] - pclcnd[1];
#if NDIM == 3
                v1[2] = pndcrd[2] - pclcnd[2];
#endif
                pndcrd = exd->ndcrd + pfcnds[sfcs[it][1]] * NDIM;
                v2[0] = pndcrd[0] - pclcnd[0];  // line AC.
                v2[1] = pndcrd[1] - pclcnd[1];
#if NDIM == 3
                v2[2] = pndcrd[2] - pclcnd[2];
                pndcrd = exd->ndcrd + pfcnds[sfcs[it][2]] * NDIM;
                v3[0] = pndcrd[0] - pclcnd[0];  // line AD.
                v3[1] = pndcrd[1] - pclcnd[1];
                v3[2] = pndcrd[2] - pclcnd[2];
                rhs[0] = v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];
                rhs[1] = v0[0]*v2[0] + v0[1]*v2[1] + v0[2]*v2[2];
                rhs[2] = v0[0]*v3[0] + v0[1]*v3[1] + v0[2]*v3[2];
                mat00 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
                mat01 = mat10 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
                mat02 = mat20 = v1[0]*v3[0] + v1[1]*v3[1] + v1[2]*v3[2];
                mat11 = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
                mat12 = mat21 = v2[0]*v3[0] + v2[1]*v3[1] + v2[2]*v3[2];
                mat22 = v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2];
                mai00 = mat11*mat22 - mat12*mat21;
                mai01 = mat02*mat21 - mat01*mat22;
                mai02 = mat01*mat12 - mat02*mat11;
                mai10 = mat12*mat20 - mat10*mat22;
                mai11 = mat00*mat22 - mat02*mat20;
                mai12 = mat02*mat10 - mat00*mat12;
                mai20 = mat10*mat21 - mat11*mat20;
                mai21 = mat01*mat20 - mat00*mat21;
                mai22 = mat00*mat11 - mat01*mat10;
                vol = mat00*mai00 + mat01*mai10 + mat02*mai20;
                mai00 /= vol; mai01 /= vol; mai02 /= vol;
                mai10 /= vol; mai11 /= vol; mai12 /= vol;
                mai20 /= vol; mai21 /= vol; mai22 /= vol;
                bcc[0] = mai00*rhs[0] + mai01*rhs[1] + mai02*rhs[2];
                bcc[1] = mai10*rhs[0] + mai11*rhs[1] + mai12*rhs[2];
                bcc[2] = mai20*rhs[0] + mai21*rhs[1] + mai22*rhs[2];
                bca = bcc[0] + bcc[1] + bcc[2];
#else
                v1s = v1[0]*v1[0] + v1[1]*v1[1];
                v2s = v2[0]*v2[0] + v2[1]*v2[1];
                v01 = v0[0]*v1[0] + v0[1]*v1[1];
                v02 = v0[0]*v2[0] + v0[1]*v2[1];
                v12 = v1[0]*v2[0] + v1[1]*v2[1];
                bcc[0] = (v01*v2s - v02*v12)/(v1s*v2s - v12*v12);
                bcc[1] = (v02*v1s - v01*v12)/(v1s*v2s - v12*v12);
                bca = bcc[0] + bcc[1];
#endif
                if ( (bcc[0]>=0.0) && (bcc[1]>=0.0) &&
#if NDIM == 3
                     (bcc[2]>=0.0) &&
#endif
                     (bca<=1.0) ) {
                    if (picl[0] == -1) {   // not set to indicate error of dup.
                        picl[0] = icl;
                        pifl[0] = ifl;
                    };
                };
                // outer triangle.
                pclcnd = exd->clcnd + jcl * NDIM;
                v0[0] = crd[0] - pclcnd[0]; // line AP.
                v0[1] = crd[1] - pclcnd[1];
#if NDIM == 3
                v0[2] = crd[2] - pclcnd[2];
#endif
                pndcrd = exd->ndcrd + pfcnds[sfcs[it][0]] * NDIM;
                v1[0] = pndcrd[0] - pclcnd[0];  // line AB.
                v1[1] = pndcrd[1] - pclcnd[1];
#if NDIM == 3
                v1[2] = pndcrd[2] - pclcnd[2];
#endif
                pndcrd = exd->ndcrd + pfcnds[sfcs[it][1]] * NDIM;
                v2[0] = pndcrd[0] - pclcnd[0];  // line AC.
                v2[1] = pndcrd[1] - pclcnd[1];
#if NDIM == 3
                v2[2] = pndcrd[2] - pclcnd[2];
                pndcrd = exd->ndcrd + pfcnds[sfcs[it][2]] * NDIM;
                v3[0] = pndcrd[0] - pclcnd[0];  // line AD.
                v3[1] = pndcrd[1] - pclcnd[1];
                v3[2] = pndcrd[2] - pclcnd[2];
                rhs[0] = v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];
                rhs[1] = v0[0]*v2[0] + v0[1]*v2[1] + v0[2]*v2[2];
                rhs[2] = v0[0]*v3[0] + v0[1]*v3[1] + v0[2]*v3[2];
                mat00 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
                mat01 = mat10 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
                mat02 = mat20 = v1[0]*v3[0] + v1[1]*v3[1] + v1[2]*v3[2];
                mat11 = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
                mat12 = mat21 = v2[0]*v3[0] + v2[1]*v3[1] + v2[2]*v3[2];
                mat22 = v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2];
                mai00 = mat11*mat22 - mat12*mat21;
                mai01 = mat02*mat21 - mat01*mat22;
                mai02 = mat01*mat12 - mat02*mat11;
                mai10 = mat12*mat20 - mat10*mat22;
                mai11 = mat00*mat22 - mat02*mat20;
                mai12 = mat02*mat10 - mat00*mat12;
                mai20 = mat10*mat21 - mat11*mat20;
                mai21 = mat01*mat20 - mat00*mat21;
                mai22 = mat00*mat11 - mat01*mat10;
                vol = mat00*mai00 + mat01*mai10 + mat02*mai20;
                mai00 /= vol; mai01 /= vol; mai02 /= vol;
                mai10 /= vol; mai11 /= vol; mai12 /= vol;
                mai20 /= vol; mai21 /= vol; mai22 /= vol;
                bcc[0] = mai00*rhs[0] + mai01*rhs[1] + mai02*rhs[2];
                bcc[1] = mai10*rhs[0] + mai11*rhs[1] + mai12*rhs[2];
                bcc[2] = mai20*rhs[0] + mai21*rhs[1] + mai22*rhs[2];
                bca = bcc[0] + bcc[1] + bcc[2];
#else
                v1s = v1[0]*v1[0] + v1[1]*v1[1];
                v2s = v2[0]*v2[0] + v2[1]*v2[1];
                v01 = v0[0]*v1[0] + v0[1]*v1[1];
                v02 = v0[0]*v2[0] + v0[1]*v2[1];
                v12 = v1[0]*v2[0] + v1[1]*v2[1];
                bcc[0] = (v01*v2s - v02*v12)/(v1s*v2s - v12*v12);
                bcc[1] = (v02*v1s - v01*v12)/(v1s*v2s - v12*v12);
                bca = bcc[0] + bcc[1];
#endif
                if ( (bcc[0]>=0.0) && (bcc[1]>=0.0) &&
#if NDIM == 3
                     (bcc[2]>=0.0) &&
#endif
                     (bca<=1.0) ) {
                    if (pjcl[0] == -1) {   // not set to indicate error of dup.
                        pjcl[0] = icl;
                        pjfl[0] = ifl;
                    };
                };
            };
        };
    };
    return NULL;
};
// vim: set ts=4 et:
