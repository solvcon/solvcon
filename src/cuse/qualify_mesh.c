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

int qualify_mesh(exedata *exd, int istart, int iend) {
    int ctpn;
    // pointers.
    int *pclfcs, *pclnds;
    double *pndcrd, *pfccnd, *pfcnml, *pindcrd, *pjndcrd;
    // scalars.
    double sht, lnt, lgh, dst;
    // iterators.
    int icl, ifc, ind;
    int it0, it1, it;
    // loop over cells.
    for (icl=istart; icl<iend; icl++) {
        lnt = SOLVCON_ALMOST_ZERO;
        sht = 1./SOLVCON_ALMOST_ZERO;
        ctpn = exd->cltpn[icl];
        pclnds = exd->clnds + icl*(CLMND+1);
        pclfcs = exd->clfcs + icl*(CLMFC+1);
        // height.
        it0 = hrng[ctpn][0];
        it1 = hrng[ctpn][1];
        for (it=it0; it<it1; it++) {
            // vertex.
            ind = pclnds[hvfs[it][0]];
            pndcrd = exd->ndcrd + ind*NDIM;
            // face.
            ifc = pclfcs[hvfs[it][1]];
            pfccnd = exd->fccnd + ifc*NDIM;
            pfcnml = exd->fcnml + ifc*NDIM;
            // calculate length.
            lgh  = (pndcrd[0] - pfccnd[0]) * pfcnml[0];
            lgh += (pndcrd[1] - pfccnd[1]) * pfcnml[1];
#if NDIM == 3
            lgh += (pndcrd[2] - pfccnd[2]) * pfcnml[2];
#endif
            lgh = fabs(lgh);
            if (lgh < sht) sht = lgh;
        };
        // width.
        it0 = egng[ctpn][0];
        it1 = egng[ctpn][1];
        for (it=it0; it<it1; it++) {
            pindcrd = exd->ndcrd + pclnds[evts[it][0]] * NDIM;
            pjndcrd = exd->ndcrd + pclnds[evts[it][1]] * NDIM;
            // calculate length.
            dst = pindcrd[0] - pjndcrd[0];
            lgh  = dst*dst;
            dst = pindcrd[1] - pjndcrd[1];
            lgh += dst*dst;
#if NDIM == 3
            dst = pindcrd[2] - pjndcrd[2];
            lgh += dst*dst;
#endif
            lgh = sqrt(lgh);
            if (lgh > lnt) lnt = lgh;
        };
        // calculate and save mesh quality.
        exd->mqlty[icl] = lnt/sht;
    };
    return 0;
};
// vim: set ts=4 et:
