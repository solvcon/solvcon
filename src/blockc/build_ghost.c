// Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of
// usage.
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
    int nnd;
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
    FPTYPE vol;
    // arrays.
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

    // compute ghost face center coordinate.
    pfcnds = msd->fcnds - (FCMND+1);
    pfccnd = msd->fccnd - msd->ndim;
    for (ifc=-1; ifc>=-msd->ngstface; ifc--) {
        for (idm=0; idm<msd->ndim; idm++) {
            pfccnd[idm] = 0.0;
        };
        for (inf=1; inf<=pfcnds[0]; inf++) {
            ind = pfcnds[inf];
            pndcrd = msd->ndcrd + ind*msd->ndim;
            for (idm=0; idm<msd->ndim; idm++) {
                pfccnd[idm] += pndcrd[idm];
            };
        };
        for (idm=0; idm<msd->ndim; idm++) {
            pfccnd[idm] /= pfcnds[0];
        };
        // advance pointers.
        pfcnds -= FCMND+1;
        pfccnd -= msd->ndim;
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

    // compute center point coordinate for each ghost cell.
    pclnds = msd->clnds - (CLMND+1);
    pclcnd = msd->clcnd - msd->ndim;
    for (icl=-1; icl>=-msd->ngstcell; icl--) {
        for (idm=0; idm<msd->ndim; idm++) {
            pclcnd[idm] = 0.0;
        };
        nnd = pclnds[0];
        for (inl=1; inl<=nnd; inl++) {
            ind = pclnds[inl];
            pndcrd = msd->ndcrd + ind*msd->ndim;
            for (idm=0; idm<msd->ndim; idm++) {
                pclcnd[idm] += pndcrd[idm];
            };
        };
        for (idm=0; idm<msd->ndim; idm++) {
            pclcnd[idm] /= nnd;
        };
        // advance pointers.
        pclnds -= CLMND+1;
        pclcnd -= msd->ndim;
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
