/*
 * Copyright (C) 2010-2011 Yung-Yu Chen <yyc@solvcon.net>.
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
#include "elaslin.h"

int bound_traction_free_soln(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd, *pvalue;
    double *pisol, *pjsoln, *pcfl;
    // scalars.
    double amp;
    // arrays.
    double nvec[3], svec[3];
    double rotm[3][3], bondm[6][6], bondminv[6][6];
    double sts[6], trc[3];
    // iterators.
    int ibnd, ifc, icl, jcl, ieq;
    int it, jt;
    nvec[2] = 0.0;
    svec[2] = 0.0;
    pcfl = exd->cfl;
    amp = 1.0;
    pfacn = facn;
    for (ibnd=0; ibnd<nbnd; ibnd++) {
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc * FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        // determine amplification factor for vanishing variables.
        //amp = pcfl[icl];
        //amp = (1+amp)/fabs(1-amp);
        //amp = 1/fabs(1-amp);
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
        // set vanishing rotated stress.
        sts[0] = -amp*sts[0];
        sts[5] = -amp*sts[5];
        sts[4] = -amp*sts[4];
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
int bound_traction_free_dsoln(exedata *exd, int nbnd, int *facn) {
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd;
    double *pidsol, *pjdsoln, *pdsol, *pdsoln;
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
        pidsol = exd->dsol + icl * NEQ * NDIM;
        pjdsoln = exd->dsoln + jcl * NEQ * NDIM;
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
        // rotate velocity to boundary coordinate system.
        pdsol = pidsol;
        for (ieq=0; ieq<3; ieq++) {
            dvel1[ieq][0] = pdsol[0];
            dvel1[ieq][1] = pdsol[1];
            dvel1[ieq][2] = 0.0;
            pdsol += NDIM;
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
        // rotate stress to boundary coordinate system.
        for (ieq=0; ieq<6; ieq++) {
            dsts1[ieq][0] = pdsol[0];
            dsts1[ieq][1] = pdsol[1];
            dsts1[ieq][2] = 0.0;
            pdsol += NDIM;
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
        for (ieq=0; ieq<3; ieq++) {
            dvel1[ieq][0] = -dvel1[ieq][0]; // unchanged.
        };
        for (ieq=0; ieq<1; ieq++) {
            dsts1[ieq][1] = -dsts1[ieq][1]; // vanishing.
        };
        for (ieq=1; ieq<4; ieq++) {
            dsts1[ieq][0] = -dsts1[ieq][0]; // unchanged.
        };
        for (ieq=4; ieq<6; ieq++) {
            dsts1[ieq][1] = -dsts1[ieq][1]; // vanishing.
        };
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
        pdsoln = pjdsoln;
        for (ieq=0; ieq<3; ieq++) {
            pdsoln[0] = dvel1[ieq][0];
            pdsoln[1] = dvel1[ieq][1];
            pdsoln += NDIM;
        };
        for (ieq=0; ieq<6; ieq++) {
            pdsoln[0] = dsts1[ieq][0];
            pdsoln[1] = dsts1[ieq][1];
            pdsoln += NDIM;
        };
        /*for (ieq=0; ieq<9; ieq++) {
            pjdsoln[0] = pidsoln[0];
            pjdsoln[1] = pidsoln[1];
            pidsoln += NDIM;
            pjdsoln += NDIM;
        };*/
        // advance boundary face.
        pfacn += 3;
    };
    return 0;
}
// vim: set ts=4 et:
