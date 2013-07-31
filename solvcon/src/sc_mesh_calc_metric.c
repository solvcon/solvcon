/*
 * Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the SOLVCON nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <Python.h>
#include <math.h>

#include "mesh.h"

/*
 * Calculate all metric information, including:
 *
 *  1. center of faces.
 *  2. unit normal and area of faces.
 *  3. center of cells.
 *  4. volume of cells.
 *
 * And fcnds could be reordered.
 */

int sc_mesh_calc_metric(sc_mesh_t *msd, int use_incenter) {
    int nnd, nfc;
    // pointers.
    int *pfcnds, *pfccls, *pclnds, *pclfcs;
    double *pndcrd, *p2ndcrd, *pfccnd, *pfcnml, *pfcara, *pclcnd, *pclvol;
    // scalars.
    double vol, vob, voc;
    double du0, du1, du2, dv0, dv1, dv2, dw0, dw1, dw2;
    // arrays.
    int ndstf[FCMND];
    double cfd[FCMND+2][msd->ndim];
    double crd[msd->ndim];
    double radvec[FCMND][msd->ndim];
    // iterators.
    int ifc, inf, ind, icl, inc, ifl;
    int idm, it, jt;

    // compute face centroids.
    pfcnds = msd->fcnds;
    pfccnd = msd->fccnd;
    if (msd->ndim == 2) {
        // 2D faces must be edge.
        for (ifc=0; ifc<msd->nface; ifc++) {
            // point 1.
            ind = pfcnds[1];
            pndcrd = msd->ndcrd + ind*msd->ndim;
            pfccnd[0] = pndcrd[0];
            pfccnd[1] = pndcrd[1];
            // point 2.
            ind = pfcnds[2];
            pndcrd = msd->ndcrd + ind*msd->ndim;
            pfccnd[0] += pndcrd[0];
            pfccnd[1] += pndcrd[1];
            // average.
            pfccnd[0] /= 2;
            pfccnd[1] /= 2;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += msd->ndim;
        };
    } else if (msd->ndim == 3) {
        for (ifc=0; ifc<msd->nface; ifc++) {
            // find averaged point.
            cfd[0][0] = cfd[0][1] = cfd[0][2] = 0.0;
            nnd = pfcnds[0];
            for (inf=1; inf<=nnd; inf++) {
                ind = pfcnds[inf];
                pndcrd = msd->ndcrd + ind*msd->ndim;
                cfd[inf][0]  = pndcrd[0];
                cfd[0  ][0] += pndcrd[0];
                cfd[inf][1]  = pndcrd[1];
                cfd[0  ][1] += pndcrd[1];
                cfd[inf][2]  = pndcrd[2];
                cfd[0  ][2] += pndcrd[2];
            };
            cfd[nnd+1][0] = cfd[1][0];
            cfd[nnd+1][1] = cfd[1][1];
            cfd[nnd+1][2] = cfd[1][2];
            cfd[0][0] /= nnd;
            cfd[0][1] /= nnd;
            cfd[0][2] /= nnd;
            // calculate area.
            pfccnd[0] = pfccnd[1] = pfccnd[2] = voc = 0.0;
            for (inf=1; inf<=nnd; inf++) {
                crd[0] = (cfd[0][0] + cfd[inf][0] + cfd[inf+1][0])/3;
                crd[1] = (cfd[0][1] + cfd[inf][1] + cfd[inf+1][1])/3;
                crd[2] = (cfd[0][2] + cfd[inf][2] + cfd[inf+1][2])/3;
                du0 = cfd[inf][0] - cfd[0][0];
                du1 = cfd[inf][1] - cfd[0][1];
                du2 = cfd[inf][2] - cfd[0][2];
                dv0 = cfd[inf+1][0] - cfd[0][0];
                dv1 = cfd[inf+1][1] - cfd[0][1];
                dv2 = cfd[inf+1][2] - cfd[0][2];
                dw0 = du1*dv2 - du2*dv1;
                dw1 = du2*dv0 - du0*dv2;
                dw2 = du0*dv1 - du1*dv0;
                vob = sqrt(dw0*dw0 + dw1*dw1 + dw2*dw2);
                pfccnd[0] += crd[0] * vob;
                pfccnd[1] += crd[1] * vob;
                pfccnd[2] += crd[2] * vob;
                voc += vob;
            };
            pfccnd[0] /= voc;
            pfccnd[1] /= voc;
            pfccnd[2] /= voc;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += msd->ndim;
        };
    };

    // compute face normal vector and area.
    pfcnds = msd->fcnds;
    pfccnd = msd->fccnd;
    pfcnml = msd->fcnml;
    pfcara = msd->fcara;
    if (msd->ndim == 2) {
        for (ifc=0; ifc<msd->nface; ifc++) {
            // 2D faces are always lines.
            pndcrd = msd->ndcrd + pfcnds[1]*msd->ndim;
            p2ndcrd = msd->ndcrd + pfcnds[2]*msd->ndim;
            // face normal.
            pfcnml[0] = p2ndcrd[1] - pndcrd[1];
            pfcnml[1] = -(p2ndcrd[0] - pndcrd[0]);
            // face ara.
            pfcara[0] = sqrt(pfcnml[0]*pfcnml[0] + pfcnml[1]*pfcnml[1]);
            // normalize face normal.
            pfcnml[0] /= pfcara[0];
            pfcnml[1] /= pfcara[0];
            // advance pointers.
            pfcnds += FCMND+1;
            pfcnml += msd->ndim;
            pfcara += 1;
        };
    } else if (msd->ndim == 3) {
        for (ifc=0; ifc<msd->nface; ifc++) {
            // compute radial vector.
            nnd = pfcnds[0];
            for (inf=0; inf<nnd; inf++) {
                ind = pfcnds[inf+1];
                pndcrd = msd->ndcrd + ind*msd->ndim;
                radvec[inf][0] = pndcrd[0] - pfccnd[0];
                radvec[inf][1] = pndcrd[1] - pfccnd[1];
                radvec[inf][2] = pndcrd[2] - pfccnd[2];
            };
            // compute cross product.
            pfcnml[0] = radvec[nnd-1][1]*radvec[0][2]
                      - radvec[nnd-1][2]*radvec[0][1];
            pfcnml[1] = radvec[nnd-1][2]*radvec[0][0]
                      - radvec[nnd-1][0]*radvec[0][2];
            pfcnml[2] = radvec[nnd-1][0]*radvec[0][1]
                      - radvec[nnd-1][1]*radvec[0][0];
            for (ind=1; ind<nnd; ind++) {
                pfcnml[0] += radvec[ind-1][1]*radvec[ind][2]
                           - radvec[ind-1][2]*radvec[ind][1];
                pfcnml[1] += radvec[ind-1][2]*radvec[ind][0]
                           - radvec[ind-1][0]*radvec[ind][2];
                pfcnml[2] += radvec[ind-1][0]*radvec[ind][1]
                           - radvec[ind-1][1]*radvec[ind][0];
            };
            // compute face area.
            pfcara[0] = sqrt(pfcnml[0]*pfcnml[0] + pfcnml[1]*pfcnml[1]
                           + pfcnml[2]*pfcnml[2]);
            // normalize normal vector.
            pfcnml[0] /= pfcara[0];
            pfcnml[1] /= pfcara[0];
            pfcnml[2] /= pfcara[0];
            // get real face area.
            pfcara[0] /= 2.0;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += msd->ndim;
            pfcnml += msd->ndim;
            pfcara += 1;
        };
    };

    // compute cell centroids.
    pclnds = msd->clnds;
    pclfcs = msd->clfcs;
    pclcnd = msd->clcnd;
    if (msd->ndim == 2) {
        for (icl=0; icl<msd->ncell; icl++) {
            if ((use_incenter == 1) && (msd->cltpn[icl] == 3)) {
                pndcrd = msd->ndcrd + pclnds[1]*msd->ndim;
                vob = msd->fcara[pclfcs[2]];
                voc = vob;
                pclcnd[0] = vob*pndcrd[0];
                pclcnd[1] = vob*pndcrd[1];
                pndcrd = msd->ndcrd + pclnds[2]*msd->ndim;
                vob = msd->fcara[pclfcs[3]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pndcrd = msd->ndcrd + pclnds[3]*msd->ndim;
                vob = msd->fcara[pclfcs[1]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
            } else {
                // averaged point.
                crd[0] = crd[1] = 0.0;
                nnd = pclnds[0];
                for (inc=1; inc<=nnd; inc++) {
                    ind = pclnds[inc];
                    pndcrd = msd->ndcrd + ind*msd->ndim;
                    crd[0] += pndcrd[0];
                    crd[1] += pndcrd[1];
                };
                crd[0] /= nnd;
                crd[1] /= nnd;
                // weight centroid.
                pclcnd[0] = pclcnd[1] = voc = 0.0;
                nfc = pclfcs[0];
                for (ifl=1; ifl<=nfc; ifl++) {
                    ifc = pclfcs[ifl];
                    pfccnd = msd->fccnd + ifc*msd->ndim;
                    pfcnml = msd->fcnml + ifc*msd->ndim;
                    pfcara = msd->fcara + ifc;
                    du0 = crd[0] - pfccnd[0];
                    du1 = crd[1] - pfccnd[1];
                    vob = fabs(du0*pfcnml[0] + du1*pfcnml[1]) * pfcara[0];
                    voc += vob;
                    dv0 = pfccnd[0] + du0/3;
                    dv1 = pfccnd[1] + du1/3;
                    pclcnd[0] += dv0 * vob;
                    pclcnd[1] += dv1 * vob;
                };
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
            };
            // advance pointers.
            pclnds += CLMND+1;
            pclfcs += CLMFC+1;
            pclcnd += msd->ndim;
        };
    } else if (msd->ndim == 3) {
        for (icl=0; icl<msd->ncell; icl++) {
            if ((use_incenter == 1) && (msd->cltpn[icl] == 5)) {
                pndcrd = msd->ndcrd + pclnds[1]*msd->ndim;
                vob = msd->fcara[pclfcs[4]];
                voc = vob;
                pclcnd[0] = vob*pndcrd[0];
                pclcnd[1] = vob*pndcrd[1];
                pclcnd[2] = vob*pndcrd[2];
                pndcrd = msd->ndcrd + pclnds[2]*msd->ndim;
                vob = msd->fcara[pclfcs[3]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pndcrd = msd->ndcrd + pclnds[3]*msd->ndim;
                vob = msd->fcara[pclfcs[2]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pndcrd = msd->ndcrd + pclnds[4]*msd->ndim;
                vob = msd->fcara[pclfcs[1]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
                pclcnd[2] /= voc;
            } else {
                // averaged point.
                crd[0] = crd[1] = crd[2] = 0.0;
                nnd = pclnds[0];
                for (inc=1; inc<=nnd; inc++) {
                    ind = pclnds[inc];
                    pndcrd = msd->ndcrd + ind*msd->ndim;
                    crd[0] += pndcrd[0];
                    crd[1] += pndcrd[1];
                    crd[2] += pndcrd[2];
                };
                crd[0] /= nnd;
                crd[1] /= nnd;
                crd[2] /= nnd;
                // weight centroid.
                pclcnd[0] = pclcnd[1] = pclcnd[2] = voc = 0.0;
                nfc = pclfcs[0];
                for (ifl=1; ifl<=nfc; ifl++) {
                    ifc = pclfcs[ifl];
                    pfccnd = msd->fccnd + ifc*msd->ndim;
                    pfcnml = msd->fcnml + ifc*msd->ndim;
                    pfcara = msd->fcara + ifc;
                    du0 = crd[0] - pfccnd[0];
                    du1 = crd[1] - pfccnd[1];
                    du2 = crd[2] - pfccnd[2];
                    vob = fabs(du0*pfcnml[0] + du1*pfcnml[1] + du2*pfcnml[2])
                        * pfcara[0];
                    voc += vob;
                    dv0 = pfccnd[0] + du0/4;
                    dv1 = pfccnd[1] + du1/4;
                    dv2 = pfccnd[2] + du2/4;
                    pclcnd[0] += dv0 * vob;
                    pclcnd[1] += dv1 * vob;
                    pclcnd[2] += dv2 * vob;
                };
                pclcnd[0] /= voc;
                pclcnd[1] /= voc;
                pclcnd[2] /= voc;
            };
            // advance pointers.
            pclnds += CLMND+1;
            pclfcs += CLMFC+1;
            pclcnd += msd->ndim;
        };
    };

    // compute volume for each cell.
    pclfcs = msd->clfcs;
    pclcnd = msd->clcnd;
    pclvol = msd->clvol;
    for (icl=0; icl<msd->ncell; icl++) {
        pclvol[0] = 0.0;
        nfc = pclfcs[0];
        for (it=1; it<=nfc; it++) {
            ifc = pclfcs[it];
            pfccls = msd->fccls + ifc*FCREL;
            pfcnds = msd->fcnds + ifc*(FCMND+1);
            pfccnd = msd->fccnd + ifc*msd->ndim;
            pfcnml = msd->fcnml + ifc*msd->ndim;
            pfcara = msd->fcara + ifc;
            // calculate volume associated with each face.
            vol = 0.0;
            for (idm=0; idm<msd->ndim; idm++) {
                vol += (pfccnd[idm] - pclcnd[idm]) * pfcnml[idm];
            };
            vol *= pfcara[0];
            // check if need to reorder node definition and connecting cell
            // list for the face.
            if (vol < 0.0) {
                if (pfccls[0] == icl) {
                    nnd = pfcnds[0];
                    for (jt=0; jt<nnd; jt++) {
                        ndstf[jt] = pfcnds[nnd-jt];
                    };
                    for (jt=0; jt<nnd; jt++) {
                        pfcnds[jt+1] = ndstf[jt];
                    };
                    for (idm=0; idm<msd->ndim; idm++) {
                        pfcnml[idm] = -pfcnml[idm];
                    };
                };
                vol = -vol;
            } else {
                if (pfccls[0] != icl) {
                    nnd = pfcnds[0];
                    for (jt=0; jt<nnd; jt++) {
                        ndstf[jt] = pfcnds[nnd-jt];
                    };
                    for (jt=0; jt<nnd; jt++) {
                        pfcnds[jt+1] = ndstf[jt];
                    };
                    for (idm=0; idm<msd->ndim; idm++) {
                        pfcnml[idm] = -pfcnml[idm];
                    };
                };
            };
            // accumulate the volume for the cell.
            pclvol[0] += vol;
        };
        // calculate the real volume.
        pclvol[0] /= msd->ndim;
        // advance pointers.
        pclfcs += CLMFC+1;
        pclcnd += msd->ndim;
        pclvol += 1;
    };

    return 0;
};

// vim: fenc=utf8 ff=unix ft=c ai et sw=4 ts=4 tw=79:
