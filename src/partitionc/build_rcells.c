// Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of
// usage.
#include "solvcon.h"
int build_rcells(MeshData *msd, int *rcells, int *rcellno) {
    // pointers.
    int *pclfcs, *pfccls;
    int *prcells;
    // iterators.
    int icl, ifl, ifl1, ifc;

    // initialize.
    prcells = rcells;
    for (icl=0; icl<msd->ncell; icl++) {
        for (ifl=0; ifl<CLMFC; ifl++) {
            prcells[ifl] = -1;
        };
        rcellno[icl] = 0;
        // advance pointers.
        prcells += CLMFC;
    };
    
    // count.
    pclfcs = msd->clfcs;
    prcells = rcells;
    for (icl=0; icl<msd->ncell; icl++) {
        for (ifl=1; ifl<=pclfcs[0]; ifl++) {
            ifl1 = ifl-1;
            ifc = pclfcs[ifl];
            pfccls = msd->fccls + ifc*FCREL;
            if (ifc == -1) {    // NOT A FACE!? SHOULDN'T HAPPEN.
                prcells[ifl1] = -1;
                continue;
            } else if (pfccls[0] == icl) {
                if (pfccls[2] != -1) {  // has neighboring block.
                    prcells[ifl1] = -1;
                } else {    // is interior.
                    prcells[ifl1] = pfccls[1];
                };
            } else if (pfccls[1] == icl) {  // I am the neighboring cell.
                prcells[ifl1] = pfccls[0];
            };
            // count rcell number.
            if (prcells[ifl1] >= 0) {
                rcellno[icl] += 1;
            } else {
                prcells[ifl1] = -1;
            };
        };
        // advance pointers.
        pclfcs += CLMFC+1;
        prcells += CLMFC;
    };

    return 0;
};
// vim: set ts=4 et:
