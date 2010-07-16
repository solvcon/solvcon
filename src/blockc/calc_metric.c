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
    };
    return 0;
};
// vim: set ts=4 et:
