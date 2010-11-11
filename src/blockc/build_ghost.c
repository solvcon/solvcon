/*
 * Copyright (C) 2008-2010 Yung-Yu Chen.
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

#include "solvcon.h"
/*
 * subroutine build_ghost: Build all information for ghost cells by 
 * mirroring information from interior cells.  The action includes:
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
int build_ghost(MeshData *msd, int *bndfcs) {
    int nnd, nfc;
    // pointers.
    int *pbndfcs;
    int *pfctpn, *pcltpn, *pclgrp;
    int *pgfctpn, *pgcltpn, *pgclgrp;
    int *pfcnds, *pfccls, *pclnds, *pclfcs;
    int *pgfcnds, *pgfccls, *pgclnds, *pgclfcs;
    FPTYPE *pndcrd, *p2ndcrd, *pgndcrd;
    FPTYPE *pfccnd, *pfcnml, *pfcara, *pclcnd, *pclvol;
    // buffers.
    int *gstndmap;
    // scalars.
    int mk_found;
    FPTYPE vol, vob, voc;
    FPTYPE du0, du1, du2, dv0, dv1, dv2, dw0, dw1, dw2;
    // arrays.
    FPTYPE cfd[FCMND+2][msd->ndim];
    FPTYPE crd[msd->ndim];
    FPTYPE radvec[FCMND][msd->ndim];
    // iterators.
    int ind, ibfc, icl, inl, inf, idm, ifl, ifc;
    int ignd, igfc, igcl;
    int it;

    gstndmap = (int *)malloc((size_t)msd->nnode*sizeof(int));
    for (ind=0; ind<msd->nnode; ind++) {
        gstndmap[ind] = msd->nnode; // initialize to the least possible value.
    };

    // create ghost entities and buil connectivities and by the way mirror node
    // coordinate.
    ignd = -1;
    igfc = -1;
    pbndfcs = bndfcs;
    pgndcrd = msd->ndcrd - msd->ndim;
    pgfctpn = msd->fctpn - 1;
    pgfcnds = msd->fcnds - (FCMND+1);
    pgfccls = msd->fccls - FCREL;
    pgcltpn = msd->cltpn - 1;
    pgclgrp = msd->clgrp - 1;
    pgclnds = msd->clnds - (CLMND+1);
    pgclfcs = msd->clfcs - (CLMFC+1);
    for (igcl=-1; igcl>=-msd->ngstcell; igcl--) {
        ibfc = pbndfcs[0];
        pfctpn = msd->fctpn + ibfc;
        pfcnds = msd->fcnds + ibfc*(FCMND+1);
        pfccls = msd->fccls + ibfc*FCREL;
        pfccnd = msd->fccnd + ibfc*msd->ndim;
        pfcnml = msd->fcnml + ibfc*msd->ndim;
        icl = pfccls[0];
        pcltpn = msd->cltpn + icl;
        pclgrp = msd->clgrp + icl;
        pclnds = msd->clnds + icl*(CLMND+1);
        pclfcs = msd->clfcs + icl*(CLMFC+1);
        // copy cell type and group.
        pgcltpn[0] = pcltpn[0];
        pgclgrp[0] = pclgrp[0];
        // process node list in ghost cell.
        for (inl=0; inl<=CLMND; inl++) {    // copy nodes from current in-cell.
            pgclnds[inl] = pclnds[inl];
        };
        for (inl=1; inl<=pclnds[0]; inl++) {
            ind = pclnds[inl];
            pndcrd = msd->ndcrd + ind*msd->ndim;
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
                for (idm=0; idm<msd->ndim; idm++) {
                    vol += (pfccnd[idm] - pndcrd[idm]) * pfcnml[idm];
                };
                for (idm=0; idm<msd->ndim; idm++) {
                    pgndcrd[idm] = pndcrd[idm] + 2*vol*pfcnml[idm];
                };
                // decrement ghost node counter.
                ignd -= 1;
                pgndcrd -= msd->ndim;
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
            pfctpn = msd->fctpn + ifc;
            pfcnds = msd->fcnds + ifc*(FCMND+1);
            pgfctpn[0] = pfctpn[0]; // copy face type.
            pgfccls[0] = igcl;  // save to ghost fccls.
            pgclfcs[ifl] = igfc;    // save to ghost clfcs.
            // face-to-node connectivity.
            for (inf=0; inf<=FCMND; inf++) {
                pgfcnds[inf] = pfcnds[inf];
            };
            for (inf=1; inf<=pgfcnds[0]; inf++) {
                ind = pgfcnds[inf];
                if (gstndmap[ind] != msd->nnode) {
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
                gstndmap[pclnds[inl]] = msd->nnode;
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
    pfcnds = msd->fcnds - (FCMND+1);
    pfccnd = msd->fccnd - msd->ndim;
    if (msd->ndim == 2) {
        // 2D faces must be edge.
        for (ifc=-1; ifc>=-msd->ngstface; ifc--) {
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
            pfcnds -= FCMND+1;
            pfccnd -= msd->ndim;
        };
    } else if (msd->ndim == 3) {
        for (ifc=-1; ifc>=-msd->ngstface; ifc--) {
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
            pfcnds -= FCMND+1;
            pfccnd -= msd->ndim;
        };
    };

    // compute ghost face normal vector and area.
    pfcnds = msd->fcnds - (FCMND+1);
    pfccnd = msd->fccnd - msd->ndim;
    pfcnml = msd->fcnml - msd->ndim;
    pfcara = msd->fcara - 1;
    if (msd->ndim == 2) {
        for (ifc=-1; ifc>=-msd->ngstface; ifc--) {
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
            pfcnds -= FCMND+1;
            pfcnml -= msd->ndim;
            pfcara -= 1;
        };
    } else if (msd->ndim == 3) {
        for (ifc=-1; ifc>=-msd->ngstface; ifc--) {
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
            pfcnds -= FCMND+1;
            pfccnd -= msd->ndim;
            pfcnml -= msd->ndim;
            pfcara -= 1;
        };
    };

    // compute cell centroids.
    pclnds = msd->clnds - (CLMND+1);
    pclfcs = msd->clfcs - (CLMFC+1);
    pclcnd = msd->clcnd - msd->ndim;
    if (msd->ndim == 2) {
        for (icl=-1; icl>=-msd->ngstcell; icl--) {
            // averaged point.
            crd[0] = crd[1] = 0.0;
            nnd = pclnds[0];
            for (inl=1; inl<=nnd; inl++) {
                ind = pclnds[inl];
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
            // advance pointers.
            pclnds -= CLMND+1;
            pclfcs -= CLMFC+1;
            pclcnd -= msd->ndim;
        };
    } else if (msd->ndim == 3) {
        for (icl=-1; icl>=-msd->ngstcell; icl--) {
            // averaged point.
            crd[0] = crd[1] = crd[2] = 0.0;
            nnd = pclnds[0];
            for (inl=1; inl<=nnd; inl++) {
                ind = pclnds[inl];
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
            // advance pointers.
            pclnds -= CLMND+1;
            pclfcs -= CLMFC+1;
            pclcnd -= msd->ndim;
        };
    };

    // compute volume for each ghost cell.
    pclfcs = msd->clfcs - (CLMFC+1);
    pclcnd = msd->clcnd - msd->ndim;
    pclvol = msd->clvol - 1;
    for (icl=-1; icl>=-msd->ngstcell; icl--) {
        pclvol[0] = 0.0;
        for (it=1; it<=pclfcs[0]; it++) {
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
                    for (idm=0; idm<msd->ndim; idm++) {
                        pfcnml[idm] = -pfcnml[idm];
                    };
                };
                vol = -vol;
            };
            // accumulate the volume for the cell.
            pclvol[0] += vol;
        };
        // calculate the real volume.
        pclvol[0] /= msd->ndim;
        // advance pointers.
        pclfcs -= CLMFC+1;
        pclcnd -= msd->ndim;
        pclvol -= 1;
    };

    return 0;
};
// vim: set ts=4 et:
