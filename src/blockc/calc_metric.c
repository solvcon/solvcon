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
    // pointers.
    int *pfcnds;
    double *pndcrd, *pfccnd;
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
            pndcrd = msd->ndcrd + ind * msd->ndim;
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
    return 0;
};
// vim: set ts=4 et:
