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
 * Build all information for ghost cells by mirroring information from interior
 * cells.  The action includes:
 *
 * 1. define indices and build connectivities for ghost nodes, faces, 
 *    and cells.  In the same loop, mirror the coordinates of interior 
 *    nodes to ghost nodes.
 * 2. compute center coordinates for faces for ghost cells.
 * 3. compute normal vectors and areas for faces for ghost cells.
 * 4. compute center coordinates for ghost cells.
 * 5. compute volume for ghost cells.
 *
 * NOTE: all the metric, type and connnectivities data passed in this 
 * subroutine are SHARED arrays rather than interior arrays.  The 
 * indices for ghost information should be carefully treated.  All the 
 * ghost indices are negative in shared arrays.
 */
template< size_t NDIM >
void UnstructuredBlock< NDIM >::fill_ghost() {
    int nnd, nfc;
    // pointers.
    int *pbndfcs;
    int *pfctpn, *pcltpn, *pclgrp;
    int *pgfctpn, *pgcltpn, *pgclgrp;
    int *pfcnds, *pfccls, *pclnds, *pclfcs;
    int *pgfcnds, *pgfccls, *pgclnds, *pgclfcs;
    double *pndcrd, *p2ndcrd, *pgndcrd;
    double *pfccnd, *pfcnml, *pfcara, *pclcnd, *pclvol;
    // buffers.
    int *gstndmap;
    // scalars.
    int mk_found;
    double vol, vob, voc;
    double du0, du1, du2, dv0, dv1, dv2, dw0, dw1, dw2;
    // arrays.
    double cfd[FCMND+2][NDIM];
    double crd[NDIM];
    double radvec[FCMND][NDIM];
    // iterators.
    int ind, ibfc, icl, inl, inf, idm, ifl, ifc;
    int ignd, igfc, igcl;
    int it;

    gstndmap = (int *)malloc((size_t)nnode()*sizeof(int));
    for (ind=0; ind<nnode(); ind++) {
        gstndmap[ind] = nnode(); // initialize to the least possible value.
    };

    // create ghost entities and buil connectivities and by the way mirror node
    // coordinate.
    ignd = -1;
    igfc = -1;
    pbndfcs = reinterpret_cast<index_type *>(bndfcs().row(0));
    pgndcrd = reinterpret_cast<real_type  *>(ndcrd().row(0)) - NDIM;
    pgfctpn = reinterpret_cast<index_type *>(fctpn().row(0)) - 1;
    pgfcnds = reinterpret_cast<index_type *>(fcnds().row(0)) - (FCMND+1);
    pgfccls = reinterpret_cast<index_type *>(fccls().row(0)) - FCREL;
    pgcltpn = reinterpret_cast<index_type *>(cltpn().row(0)) - 1;
    pgclgrp = reinterpret_cast<index_type *>(clgrp().row(0)) - 1;
    pgclnds = reinterpret_cast<index_type *>(clnds().row(0)) - (CLMND+1);
    pgclfcs = reinterpret_cast<index_type *>(clfcs().row(0)) - (CLMFC+1);
    for (igcl=-1; igcl>=-ngstcell(); igcl--) {
        ibfc = pbndfcs[0];
        pfctpn = reinterpret_cast<index_type *>(fctpn().row(0)) + ibfc;
        pfcnds = reinterpret_cast<index_type *>(fcnds().row(0)) + ibfc*(FCMND+1);
        pfccls = reinterpret_cast<index_type *>(fccls().row(0)) + ibfc*FCREL;
        pfccnd = reinterpret_cast<real_type  *>(fccnd().row(0)) + ibfc*NDIM;
        pfcnml = reinterpret_cast<real_type  *>(fcnml().row(0)) + ibfc*NDIM;
        icl = pfccls[0];
        pcltpn = reinterpret_cast<index_type *>(cltpn().row(0)) + icl;
        pclgrp = reinterpret_cast<index_type *>(clgrp().row(0)) + icl;
        pclnds = reinterpret_cast<index_type *>(clnds().row(0)) + icl*(CLMND+1);
        pclfcs = reinterpret_cast<index_type *>(clfcs().row(0)) + icl*(CLMFC+1);
        // copy cell type and group.
        pgcltpn[0] = pcltpn[0];
        pgclgrp[0] = pclgrp[0];
        // process node list in ghost cell.
        for (inl=0; inl<=CLMND; inl++) {    // copy nodes from current in-cell.
            pgclnds[inl] = pclnds[inl];
        };
        for (inl=1; inl<=pclnds[0]; inl++) {
            ind = pclnds[inl];
            pndcrd = reinterpret_cast<real_type *>(ndcrd().row(0)) + ind*NDIM;
            // try to find the node in the boundary face.
            mk_found = 0;
            for (inf=1; inf<=pfcnds[0]; inf++) {
                if (ind == pfcnds[inf]) {
                    mk_found = 1;
                    break;
                };
            };
            // if not found, it should be a ghost node.
            if (mk_found == 0) {
                gstndmap[ind] = ignd;   // record map for face processing.
                pgclnds[inl] = ignd;    // save to clnds.
                // mirror coordinate of ghost cell.
                // NOTE: fcnml always points outward.
                vol = 0.0;
                for (idm=0; idm<NDIM; idm++) {
                    vol += (pfccnd[idm] - pndcrd[idm]) * pfcnml[idm];
                };
                for (idm=0; idm<NDIM; idm++) {
                    pgndcrd[idm] = pndcrd[idm] + 2*vol*pfcnml[idm];
                };
                // decrement ghost node counter.
                ignd -= 1;
                pgndcrd -= NDIM;
            };
        };
        // set the relating cell as ghost cell.
        pfccls[1] = igcl;
        // process face list in ghost cell.
        for (ifl=0; ifl<=CLMFC; ifl++) {
            pgclfcs[ifl] = pclfcs[ifl]; // copy in-face to ghost.
        };
        for (ifl=1; ifl<=pclfcs[0]; ifl++) {
            ifc = pclfcs[ifl];  // the face to be processed.
            if (ifc == ibfc) continue;  // if boundary face then skip.
            pfctpn = reinterpret_cast<index_type *>(fctpn().row(0)) + ifc;
            pfcnds = reinterpret_cast<index_type *>(fcnds().row(0)) + ifc*(FCMND+1);
            pgfctpn[0] = pfctpn[0]; // copy face type.
            pgfccls[0] = igcl;  // save to ghost fccls.
            pgclfcs[ifl] = igfc;    // save to ghost clfcs.
            // face-to-node connectivity.
            for (inf=0; inf<=FCMND; inf++) {
                pgfcnds[inf] = pfcnds[inf];
            };
            for (inf=1; inf<=pgfcnds[0]; inf++) {
                ind = pgfcnds[inf];
                if (gstndmap[ind] != nnode()) {
                    pgfcnds[inf] = gstndmap[ind];   // save gstnode to fcnds.
                };
            };
            // decrement ghost face counter.
            igfc -= 1;
            pgfctpn -= 1;
            pgfcnds -= FCMND+1;
            pgfccls -= FCREL;
        };
        // erase node map record.
        for (inl=1; inl<=pclnds[0]; inl++) {
                gstndmap[pclnds[inl]] = nnode();
        };
        // advance pointers.
        pbndfcs += 2;
        pgcltpn -= 1;
        pgclgrp -= 1;
        pgclnds -= CLMND+1;
        pgclfcs -= CLMFC+1;
    };
    free(gstndmap);

    // compute ghost face centroids.
    pfcnds = reinterpret_cast<index_type *>(fcnds().row(0)) - (FCMND+1);
    pfccnd = reinterpret_cast<real_type  *>(fccnd().row(0)) - NDIM;
    if (NDIM == 2) {
        // 2D faces must be edge.
        for (ifc=-1; ifc>=-ngstface(); ifc--) {
            // point 1.
            ind = pfcnds[1];
            pndcrd = reinterpret_cast<real_type *>(ndcrd().row(0)) + ind*NDIM;
            pfccnd[0] = pndcrd[0];
            pfccnd[1] = pndcrd[1];
            // point 2.
            ind = pfcnds[2];
            pndcrd = reinterpret_cast<real_type *>(ndcrd().row(0)) + ind*NDIM;
            pfccnd[0] += pndcrd[0];
            pfccnd[1] += pndcrd[1];
            // average.
            pfccnd[0] /= 2;
            pfccnd[1] /= 2;
            // advance pointers.
            pfcnds -= FCMND+1;
            pfccnd -= NDIM;
        };
    } else if (NDIM == 3) {
        for (ifc=-1; ifc>=-ngstface(); ifc--) {
            // find averaged point.
            cfd[0][0] = cfd[0][1] = cfd[0][2] = 0.0;
            nnd = pfcnds[0];
            for (inf=1; inf<=nnd; inf++) {
                ind = pfcnds[inf];
                pndcrd = reinterpret_cast<real_type *>(ndcrd().row(0)) + ind*NDIM;
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
            pfcnds -= FCMND+1;
            pfccnd -= NDIM;
        };
    };

    // compute ghost face normal vector and area.
    pfcnds = reinterpret_cast<index_type *>(fcnds().row(0)) - (FCMND+1);
    pfccnd = reinterpret_cast<real_type  *>(fccnd().row(0)) - NDIM;
    pfcnml = reinterpret_cast<real_type  *>(fcnml().row(0)) - NDIM;
    pfcara = reinterpret_cast<real_type  *>(fcara().row(0)) - 1;
    if (NDIM == 2) {
        for (ifc=-1; ifc>=-ngstface(); ifc--) {
            // 2D faces are always lines.
            pndcrd = reinterpret_cast<real_type *>(ndcrd().row(0)) + pfcnds[1]*NDIM;
            p2ndcrd = reinterpret_cast<real_type *>(ndcrd().row(0)) + pfcnds[2]*NDIM;
            // face normal.
            pfcnml[0] = p2ndcrd[1] - pndcrd[1];
            pfcnml[1] = -(p2ndcrd[0] - pndcrd[0]);
            // face ara.
            pfcara[0] = sqrt(pfcnml[0]*pfcnml[0] + pfcnml[1]*pfcnml[1]);
            // normalize face normal.
            pfcnml[0] /= pfcara[0];
            pfcnml[1] /= pfcara[0];
            // advance pointers.
            pfcnds -= FCMND+1;
            pfcnml -= NDIM;
            pfcara -= 1;
        };
    } else if (NDIM == 3) {
        for (ifc=-1; ifc>=-ngstface(); ifc--) {
            // compute radial vector.
            nnd = pfcnds[0];
            for (inf=0; inf<nnd; inf++) {
                ind = pfcnds[inf+1];
                pndcrd = reinterpret_cast<real_type *>(ndcrd().row(0)) + ind*NDIM;
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
            pfcnds -= FCMND+1;
            pfccnd -= NDIM;
            pfcnml -= NDIM;
            pfcara -= 1;
        };
    };

    // compute cell centroids.
    pclnds = reinterpret_cast<index_type *>(clnds().row(0)) - (CLMND+1);
    pclfcs = reinterpret_cast<index_type *>(clfcs().row(0)) - (CLMFC+1);
    pclcnd = reinterpret_cast<real_type  *>(clcnd().row(0)) - NDIM;
    if (NDIM == 2) {
        for (icl=-1; icl>=-ngstcell(); icl--) {
            // averaged point.
            crd[0] = crd[1] = 0.0;
            nnd = pclnds[0];
            for (inl=1; inl<=nnd; inl++) {
                ind = pclnds[inl];
                pndcrd = reinterpret_cast<real_type *>(ndcrd().row(0)) + ind*NDIM;
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
                pfccnd = reinterpret_cast<real_type *>(fccnd().row(0)) + ifc*NDIM;
                pfcnml = reinterpret_cast<real_type *>(fcnml().row(0)) + ifc*NDIM;
                pfcara = reinterpret_cast<real_type *>(fcara().row(0)) + ifc;
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
            // advance pointers.
            pclnds -= CLMND+1;
            pclfcs -= CLMFC+1;
            pclcnd -= NDIM;
        };
    } else if (NDIM == 3) {
        for (icl=-1; icl>=-ngstcell(); icl--) {
            // averaged point.
            crd[0] = crd[1] = crd[2] = 0.0;
            nnd = pclnds[0];
            for (inl=1; inl<=nnd; inl++) {
                ind = pclnds[inl];
                pndcrd = reinterpret_cast<real_type *>(ndcrd().row(0)) + ind*NDIM;
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
                pfccnd = reinterpret_cast<real_type *>(fccnd().row(0)) + ifc*NDIM;
                pfcnml = reinterpret_cast<real_type *>(fcnml().row(0)) + ifc*NDIM;
                pfcara = reinterpret_cast<real_type *>(fcara().row(0)) + ifc;
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
            // advance pointers.
            pclnds -= CLMND+1;
            pclfcs -= CLMFC+1;
            pclcnd -= NDIM;
        };
    };

    // compute volume for each ghost cell.
    pclfcs = reinterpret_cast<index_type *>(clfcs().row(0)) - (CLMFC+1);
    pclcnd = reinterpret_cast<real_type  *>(clcnd().row(0)) - NDIM;
    pclvol = reinterpret_cast<real_type  *>(clvol().row(0)) - 1;
    for (icl=-1; icl>=-ngstcell(); icl--) {
        pclvol[0] = 0.0;
        for (it=1; it<=pclfcs[0]; it++) {
            ifc = pclfcs[it];
            pfccls = reinterpret_cast<index_type *>(fccls().row(0)) + ifc*FCREL;
            pfcnds = reinterpret_cast<index_type *>(fcnds().row(0)) + ifc*(FCMND+1);
            pfccnd = reinterpret_cast<real_type  *>(fccnd().row(0)) + ifc*NDIM;
            pfcnml = reinterpret_cast<real_type  *>(fcnml().row(0)) + ifc*NDIM;
            pfcara = reinterpret_cast<real_type  *>(fcara().row(0)) + ifc;
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
                    for (idm=0; idm<NDIM; idm++) {
                        pfcnml[idm] = -pfcnml[idm];
                    };
                };
                vol = -vol;
            };
            // accumulate the volume for the cell.
            pclvol[0] += vol;
        };
        // calculate the real volume.
        pclvol[0] /= NDIM;
        // advance pointers.
        pclfcs -= CLMFC+1;
        pclcnd -= NDIM;
        pclvol -= 1;
    };
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
