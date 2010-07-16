#include "solvcon.h"
/*
 * subroutine calc_metric: Calculate all metric information, including:
 *  1. center of faces.
 *  2. unit normal and area of faces.
 *  3. center of cells.
 *  4. volume of cells.
 * And fcnds could be reordered.
*/
int calc_metric(MeshData *msd) {
    int fcnnd;
    int nnd;
    // pointers.
    int *pfcnds;
    double *pndcrd, *p2ndcrd, *pfccnd, *pfcnml, *pfcara;
    // arrays.
    double lvec[msd->ndim];
    double radvec[msd->fcmnd][msd->ndim];
    // iterators.
    int ifc, inf, ind;
    int it;

    // compute face center coordinate.
    pfcnds = msd->fcnds;
    pfccnd = msd->fccnd;
    for (ifc=0; ifc<msd->nface; ifc++) {
        fcnnd = pfcnds[0];
        // empty center.
        for (it=0; it<msd->ndim; it++) {
            pfccnd[it] = 0.0;
        };
        // sum all node coordinates.
        for (inf=1; inf<=msd->fcmnd; inf++) {
            ind = pfcnds[inf];
            pndcrd = msd->ndcrd + ind*msd->ndim;
            for (it=0; it<msd->ndim; it++) {
                pfccnd[it] += pndcrd[it];
            };
        };
        // average.
        for (it=0; it<msd->ndim; it++) {
            pfccnd[it] /= fcnnd;
        };
        // advance.
        pfcnds += msd->fcmnd+1;
        pfccnd += msd->ndim;
    };

    // compute face normal vector and area.
    if (msd->ndim == 2) {
        for (ifc=0; ifc<msd->nface; ifc++) {
            nnd = 2;    // 2D faces are always lines.
            pfcnds = msd->fcnds + ifc*msd->fcmnd;
            pndcrd = msd->ndcrd + pfcnds[1]*msd->ndim;
            p2ndcrd = msd->ndcrd + pfcnds[2]*msd->ndim;
            pfcnml = msd->fcnml + ifc*msd->ndim;
            pfcara = msd->fcara + ifc;
            lvec[0] = p2ndcrd[0] - pndcrd[0];
            lvec[1] = p2ndcrd[1] - pndcrd[1];
            // face normal.
            pfcnml[0] = lvec[1];
            pfcnml[1] = -lvec[0];
            // face ara.
            pfcara[0] = sqrt(pfcnml[0]*pfcnml[0] + pfcnml[1]*pfcnml[1]);
            // normalize face normal.
            pfcnml[0] /= pfcara[0];
            pfcnml[1] /= pfcara[0];
        };
    } else if (msd->ndim == 3) {
        for (ifc=0; ifc<msd->nface; ifc++) {
            pfcnds = msd->fcnds + ifc*msd->fcmnd;
            pfccnd = msd->fccnd + ifc*msd->ndim;
            pfcnml = msd->fcnml + ifc*msd->ndim;
            pfcara = msd->fcara + ifc;
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
        };
    };

    return 0;
};
// vim: set ts=4 et:
