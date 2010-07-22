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
    int nnd, nfc;
    // pointers.
    int *pfcnds, *pfccls, *pclnds, *pclfcs;
    FPTYPE *pndcrd, *p2ndcrd, *pfccnd, *pfcnml, *pfcara, *pclcnd, *pclvol;
    // scalars.
    FPTYPE vol;
    // arrays.
    FPTYPE radvec[FCMND][msd->ndim];
    int ndstf[FCMND];
    // iterators.
    int ifc, inf, ind, icl, inc;
    int idm, it, jt;

    // compute face center coordinate.
    pfcnds = msd->fcnds;
    pfccnd = msd->fccnd;
    for (ifc=0; ifc<msd->nface; ifc++) {
        for (idm=0; idm<msd->ndim; idm++) {
            pfccnd[idm] = 0.0;
        };
        nnd = pfcnds[0];
        for (inf=1; inf<=nnd; inf++) {
            ind = pfcnds[inf];
            pndcrd = msd->ndcrd + ind*msd->ndim;
            for (idm=0; idm<msd->ndim; idm++) {
                pfccnd[idm] += pndcrd[idm];
            };
        };
        for (idm=0; idm<msd->ndim; idm++) {
            pfccnd[idm] /= nnd;
        };
        // advance pointers.
        pfcnds += FCMND+1;
        pfccnd += msd->ndim;
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

    // compute center point coordinate for each cell.
    pclnds = msd->clnds;
    pclcnd = msd->clcnd;
    for (icl=0; icl<msd->ncell; icl++) {
        for (idm=0; idm<msd->ndim; idm++) {
            pclcnd[idm] = 0.0;
        };
        nnd = pclnds[0];
        for (inc=1; inc<=nnd; inc++) {
            ind = pclnds[inc];
            pndcrd = msd->ndcrd + ind*msd->ndim;
            for (idm=0; idm<msd->ndim; idm++) {
                pclcnd[idm] += pndcrd[idm];
            };
        };
        for (idm=0; idm<msd->ndim; idm++) {
            pclcnd[idm] /= nnd;
        };
        // advance pointers.
        pclnds += CLMND+1;
        pclcnd += msd->ndim;
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
// vim: set ts=4 et:
