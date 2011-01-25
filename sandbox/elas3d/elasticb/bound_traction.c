/*
 * Copyright (C) 2010 Yung-Yu Chen <yyc@solvcon.net>.
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

#include <stddef.h>
#include <math.h>
#include "elastic.h"

int bound_traction_soln(exedata *exd, int nbnd, int *facn,
        int nvalue, double *value) {
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd, *pvalue;
    double *pisol, *pjsoln;
    // scalars.
    double tdep;
    // arrays.
    double nvec[3], svec[3];
    double rotm[3][3], bondm[6][6], bondminv[6][6];
    double sts[6], trc[3];
    // iterators.
    int ibnd, ifc, icl, jcl, ieq;
    int it, jt;
    nvec[2] = 0.0;
    svec[2] = 0.0;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc * FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        // set through velocity.
        pisol = exd->sol + icl * NEQ;
        pjsoln = exd->soln + jcl * NEQ;
        pjsoln[0] = pisol[0];
        pjsoln[1] = pisol[1];
        pjsoln[2] = pisol[2];
        // get transformation matrices.
        pfcnml = exd->fcnml + ifc * NDIM;
        nvec[0] = pfcnml[0];
        nvec[1] = pfcnml[1];
        pfcnds = exd->fcnds + ifc * (FCMND+1);
        pndcrd = exd->ndcrd + pfcnds[2] * NDIM;
        svec[0] = pndcrd[0];
        svec[1] = pndcrd[1];
        pndcrd = exd->ndcrd + pfcnds[1] * NDIM;
        svec[0] -= pndcrd[0];
        svec[1] -= pndcrd[1];
        get_transformation(nvec, svec, rotm, bondm, bondminv);
        // rotate original stress values to boundary coordinate system.
        for (it=0; it<6; it++) {
            sts[it] = 0.0;
            for (jt=0; jt<6; jt++) {
                sts[it] += bondm[it][jt] * pisol[jt+3];
            };
        };
        // get the traction force vector.
        pvalue = value + ibnd * nvalue;
        if (pvalue[0] != 0.0) {
            trc[0] = pvalue[1];
            trc[1] = pvalue[2];
            trc[2] = pvalue[3];
        } else {
            trc[0] = trc[1] = trc[2] = 0.0;
            for (it=0; it<3; it++) {
                trc[it] = 0.0;
                for (jt=0; jt<3; jt++) {
                    trc[it] += rotm[it][jt] * pvalue[jt+1];
                };
            };
        };
        // set rotated stress.
        tdep = sin(pvalue[4]*exd->time + pvalue[5]);
        sts[0] = trc[0] * tdep;
        sts[5] = trc[1] * tdep;
        sts[4] = trc[2] * tdep;
        // rotate the boundary stress back to the original coordinate system.
        for (it=0; it<6; it++) {
            pjsoln[it+3] = 0.0;
            for (jt=0; jt<6; jt++) {
                pjsoln[it+3] += bondminv[it][jt] * sts[jt];
            };
        };
        // advance boundary face.
        pfacn += 3;
    };
    return 0;
}
int bound_traction_dsoln(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd;
    double *pidsol, *pjdsoln;
    // arrays.
    double nvec[3], svec[3];
    double rotm[3][3], bondm[6][6], bondminv[6][6];
    double dvel0[3][3], dvel1[3][3], dsts0[6][3], dsts1[6][3];
    // iterators.
    int ibnd, ifc, icl, jcl, ieq;
    int ii, ij, ik;
    nvec[2] = 0.0;
    svec[2] = 0.0;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc * FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        // get transformation matrices.
        pfcnml = exd->fcnml + ifc * NDIM;
        nvec[0] = pfcnml[0];
        nvec[1] = pfcnml[1];
        pfcnds = exd->fcnds + ifc * (FCMND+1);
        pndcrd = exd->ndcrd + pfcnds[2] * NDIM;
        svec[0] = pndcrd[0];
        svec[1] = pndcrd[1];
        pndcrd = exd->ndcrd + pfcnds[1] * NDIM;
        svec[0] -= pndcrd[0];
        svec[1] -= pndcrd[1];
        get_transformation(nvec, svec, rotm, bondm, bondminv);
        // rotate velocity gradient to boundary coordinate system.
        pidsol = exd->dsol + icl * NEQ * NDIM;
        for (ieq=0; ieq<3; ieq++) {
            dvel1[ieq][0] = pidsol[0];
            dvel1[ieq][1] = pidsol[1];
            dvel1[ieq][2] = 0.0;
            pidsol += NDIM;
        };
        for (ii=0; ii<3; ii++) {
            for (ij=0; ij<3; ij++) {
                dvel0[ii][ij] = 0.0;
                for (ik=0; ik<3; ik++) {
                    dvel0[ii][ij] += rotm[ii][ik]*dvel1[ik][ij];
                };
            };
        };
        for (ii=0; ii<3; ii++) {
            for (ij=0; ij<3; ij++) {
                dvel1[ii][ij] = 0.0;
                for (ik=0; ik<3; ik++) {
                    dvel1[ii][ij] += dvel0[ii][ik]*rotm[ij][ik];
                };
            };
        };
        // rotate stress gradient to boundary coordinate system.
        for (ieq=0; ieq<6; ieq++) {
            dsts1[ieq][0] = pidsol[0];
            dsts1[ieq][1] = pidsol[1];
            dsts1[ieq][2] = 0.0;
            pidsol += NDIM;
        };
        for (ii=0; ii<6; ii++) {
            for (ij=0; ij<3; ij++) {
                dsts0[ii][ij] = 0.0;
                for (ik=0; ik<6; ik++) {
                    dsts0[ii][ij] += bondm[ii][ik]*dsts1[ik][ij];
                };
            };
        };
        for (ii=0; ii<6; ii++) {
            for (ij=0; ij<3; ij++) {
                dsts1[ii][ij] = 0.0;
                for (ik=0; ik<3; ik++) {
                    dsts1[ii][ij] += dsts0[ii][ik]*rotm[ij][ik];
                };
            };
        };
        // set on boundary coordinate system.
        // unchanged.
        dvel1[0][0] = -dvel1[0][0];
        dvel1[1][0] = -dvel1[1][0];
        dvel1[2][0] = -dvel1[2][0];
        dsts1[1][0] = -dsts1[1][0];
        dsts1[2][0] = -dsts1[2][0];
        dsts1[3][0] = -dsts1[3][0];
        // vanishing.
        dsts1[0][1] = dsts1[4][1] = dsts1[5][1] = 0.0;
        // rotate velocity to global coordinate system.
        for (ii=0; ii<3; ii++) {
            for (ij=0; ij<3; ij++) {
                dvel0[ii][ij] = 0.0;
                for (ik=0; ik<3; ik++) {
                    dvel0[ii][ij] += rotm[ik][ii]*dvel1[ik][ij];
                };
            };
        };
        for (ii=0; ii<3; ii++) {
            for (ij=0; ij<3; ij++) {
                dvel1[ii][ij] = 0.0;
                for (ik=0; ik<3; ik++) {
                    dvel1[ii][ij] += dvel0[ii][ik]*rotm[ik][ij];
                };
            };
        };
        // rotate stress to global coordinate system.
        for (ii=0; ii<6; ii++) {
            for (ij=0; ij<3; ij++) {
                dsts0[ii][ij] = 0.0;
                for (ik=0; ik<6; ik++) {
                    dsts0[ii][ij] += bondminv[ii][ik]*dsts1[ik][ij];
                };
            };
        };
        for (ii=0; ii<6; ii++) {
            for (ij=0; ij<3; ij++) {
                dsts1[ii][ij] = 0.0;
                for (ik=0; ik<3; ik++) {
                    dsts1[ii][ij] += dsts0[ii][ik]*rotm[ik][ij];
                };
            };
        };
        // set to ghost gradient.
        pjdsoln = exd->dsoln + jcl * NEQ * NDIM;
        for (ieq=0; ieq<3; ieq++) {
            pjdsoln[0] = dvel1[ieq][0];
            pjdsoln[1] = dvel1[ieq][1];
            pjdsoln += NDIM;
        };
        for (ieq=0; ieq<6; ieq++) {
            pjdsoln[0] = dsts1[ieq][0];
            pjdsoln[1] = dsts1[ieq][1];
            pjdsoln += NDIM;
        };
        /*for (ieq=0; ieq<NEQ; ieq++) {
            pjdsoln[0] = 0.0;
            pjdsoln[1] = 0.0;
            pjdsoln += NDIM;
        };*/
        // advance boundary face.
        pfacn += 3;
    };
    return 0;
}
/*int bound_traction_dsoln(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls;
    double *pidsoln, *pjdsoln;
    // iterators.
    int ibnd, ifc, icl, jcl, ieq;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc * FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pidsoln = exd->dsoln + icl * NEQ * NDIM;
        pjdsoln = exd->dsoln + jcl * NEQ * NDIM;
        // set through ghost gradient.
        for (ieq=0; ieq<3; ieq++) {
            pjdsoln[0] = pidsoln[0];
            pjdsoln[1] = pidsoln[1];
            pidsoln += NDIM;
            pjdsoln += NDIM;
        };
        // set gradient to 0;
        for (ieq=3; ieq<NEQ; ieq++) {
            pjdsoln[0] = 0.0;
            pjdsoln[1] = 0.0;
            pjdsoln += NDIM;
        };
        // advance boundary face.
        pfacn += 3;
    };
    return 0;
}*/
// vim: set ts=4 et:
