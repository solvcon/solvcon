#pragma once

/*
 * Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cmath>

#include "march/mesh/UnstructuredBlock/class.hpp"

namespace march
{

/**
 * Calculate all metric information, including:
 *
 *  1. center of faces.
 *  2. unit normal and area of faces.
 *  3. center of cells.
 *  4. volume of cells.
 *
 * And fcnds could be reordered.
 */
template< size_t NDIM >
void UnstructuredBlock< NDIM >::calc_metric() {
    int nnd, nfc;
    // pointers.
    int *pfcnds, *pfccls, *pclnds, *pclfcs;
    double *pndcrd, *p2ndcrd, *pfccnd, *pfcnml, *pfcara, *pclcnd, *pclvol;
    // scalars.
    double vol, vob, voc;
    double du0, du1, du2, dv0, dv1, dv2, dw0, dw1, dw2;
    // arrays.
    int ndstf[FCMND];
    double cfd[FCMND+2][NDIM];
    double crd[NDIM];
    double radvec[FCMND][NDIM];
    // iterators.
    int ifc, inf, ind, icl, inc, ifl;
    int idm, it, jt;

    // utilized arrays.
    real_type  * lndcrd = reinterpret_cast<real_type  *>(ndcrd().row(0));
    index_type * lfcnds = reinterpret_cast<index_type *>(fcnds().row(0));
    index_type * lfccls = reinterpret_cast<index_type *>(fccls().row(0));
    real_type  * lfccnd = reinterpret_cast<real_type  *>(fccnd().row(0));
    real_type  * lfcnml = reinterpret_cast<real_type  *>(fcnml().row(0));
    real_type  * lfcara = reinterpret_cast<real_type  *>(fcara().row(0));
    index_type * lcltpn = reinterpret_cast<shape_type *>(cltpn().row(0));
    index_type * lclnds = reinterpret_cast<index_type *>(clnds().row(0));
    index_type * lclfcs = reinterpret_cast<index_type *>(clfcs().row(0));
    real_type  * lclcnd = reinterpret_cast<real_type  *>(clcnd().row(0));
    real_type  * lclvol = reinterpret_cast<real_type  *>(clvol().row(0));

    // compute face centroids.
    pfcnds = lfcnds;
    pfccnd = lfccnd;
    if (NDIM == 2) {
        // 2D faces must be edge.
        for (ifc=0; ifc<nface(); ifc++) {
            // point 1.
            ind = pfcnds[1];
            pndcrd = lndcrd + ind*NDIM;
            pfccnd[0] = pndcrd[0];
            pfccnd[1] = pndcrd[1];
            // point 2.
            ind = pfcnds[2];
            pndcrd = lndcrd + ind*NDIM;
            pfccnd[0] += pndcrd[0];
            pfccnd[1] += pndcrd[1];
            // average.
            pfccnd[0] /= 2;
            pfccnd[1] /= 2;
            // advance pointers.
            pfcnds += FCMND+1;
            pfccnd += NDIM;
        };
    } else if (NDIM == 3) {
        for (ifc=0; ifc<nface(); ifc++) {
            // find averaged point.
            cfd[0][0] = cfd[0][1] = cfd[0][2] = 0.0;
            nnd = pfcnds[0];
            for (inf=1; inf<=nnd; inf++) {
                ind = pfcnds[inf];
                pndcrd = lndcrd + ind*NDIM;
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
            pfccnd += NDIM;
        };
    };

    // compute face normal vector and area.
    pfcnds = lfcnds;
    pfccnd = lfccnd;
    pfcnml = lfcnml;
    pfcara = lfcara;
    if (NDIM == 2) {
        for (ifc=0; ifc<nface(); ifc++) {
            // 2D faces are always lines.
            pndcrd = lndcrd + pfcnds[1]*NDIM;
            p2ndcrd = lndcrd + pfcnds[2]*NDIM;
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
            pfcnml += NDIM;
            pfcara += 1;
        };
    } else if (NDIM == 3) {
        for (ifc=0; ifc<nface(); ifc++) {
            // compute radial vector.
            nnd = pfcnds[0];
            for (inf=0; inf<nnd; inf++) {
                ind = pfcnds[inf+1];
                pndcrd = lndcrd + ind*NDIM;
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
            pfccnd += NDIM;
            pfcnml += NDIM;
            pfcara += 1;
        };
    };

    // compute cell centroids.
    pclnds = lclnds;
    pclfcs = lclfcs;
    pclcnd = lclcnd;
    if (NDIM == 2) {
        for (icl=0; icl<ncell(); icl++) {
            if ((use_incenter()) && (lcltpn[icl] == 3)) {
                pndcrd = lndcrd + pclnds[1]*NDIM;
                vob = lfcara[pclfcs[2]];
                voc = vob;
                pclcnd[0] = vob*pndcrd[0];
                pclcnd[1] = vob*pndcrd[1];
                pndcrd = lndcrd + pclnds[2]*NDIM;
                vob = lfcara[pclfcs[3]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pndcrd = lndcrd + pclnds[3]*NDIM;
                vob = lfcara[pclfcs[1]];
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
                    pndcrd = lndcrd + ind*NDIM;
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
                    pfccnd = lfccnd + ifc*NDIM;
                    pfcnml = lfcnml + ifc*NDIM;
                    pfcara = lfcara + ifc;
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
            pclcnd += NDIM;
        };
    } else if (NDIM == 3) {
        for (icl=0; icl<ncell(); icl++) {
            if ((use_incenter()) && (lcltpn[icl] == 5)) {
                pndcrd = lndcrd + pclnds[1]*NDIM;
                vob = lfcara[pclfcs[4]];
                voc = vob;
                pclcnd[0] = vob*pndcrd[0];
                pclcnd[1] = vob*pndcrd[1];
                pclcnd[2] = vob*pndcrd[2];
                pndcrd = lndcrd + pclnds[2]*NDIM;
                vob = lfcara[pclfcs[3]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pndcrd = lndcrd + pclnds[3]*NDIM;
                vob = lfcara[pclfcs[2]];
                voc += vob;
                pclcnd[0] += vob*pndcrd[0];
                pclcnd[1] += vob*pndcrd[1];
                pclcnd[2] += vob*pndcrd[2];
                pndcrd = lndcrd + pclnds[4]*NDIM;
                vob = lfcara[pclfcs[1]];
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
                    pndcrd = lndcrd + ind*NDIM;
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
                    pfccnd = lfccnd + ifc*NDIM;
                    pfcnml = lfcnml + ifc*NDIM;
                    pfcara = lfcara + ifc;
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
            pclcnd += NDIM;
        };
    };

    // compute volume for each cell.
    pclfcs = lclfcs;
    pclcnd = lclcnd;
    pclvol = lclvol;
    for (icl=0; icl<ncell(); icl++) {
        pclvol[0] = 0.0;
        nfc = pclfcs[0];
        for (it=1; it<=nfc; it++) {
            ifc = pclfcs[it];
            pfccls = lfccls + ifc*FCREL;
            pfcnds = lfcnds + ifc*(FCMND+1);
            pfccnd = lfccnd + ifc*NDIM;
            pfcnml = lfcnml + ifc*NDIM;
            pfcara = lfcara + ifc;
            // calculate volume associated with each face.
            vol = 0.0;
            for (idm=0; idm<NDIM; idm++) {
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
                    for (idm=0; idm<NDIM; idm++) {
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
                    for (idm=0; idm<NDIM; idm++) {
                        pfcnml[idm] = -pfcnml[idm];
                    };
                };
            };
            // accumulate the volume for the cell.
            pclvol[0] += vol;
        };
        // calculate the real volume.
        pclvol[0] /= NDIM;
        // advance pointers.
        pclfcs += CLMFC+1;
        pclcnd += NDIM;
        pclvol += 1;
    };
};

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
