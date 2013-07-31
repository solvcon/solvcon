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

#include "mesh.h"

int sc_mesh_build_rcells(sc_mesh_t *msd, int *rcells, int *rcellno) {
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
