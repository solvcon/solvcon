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
int count_max_nodeinblock(MeshData *msd, int *part, int nblk, int *pmndcnt) {
    // pointers.
    int *pclnds;
    // buffers.
    int *blkit, *ndcnt, *ndlcnt;
    // iterators.
    int iblk, icl, ind, inl;
    int it;

    // sweep 1.
    blkit = (int *)malloc((size_t)nblk*2*sizeof(int));
    for (iblk=0; iblk<nblk; iblk++) {   // initialize.
        blkit[iblk*2] = blkit[iblk*2+1] = 0;
    };
    for (icl=0; icl<msd->ncell; icl++) {
        iblk = part[icl];
        blkit[iblk*2+1] += 1;
    };
    for (iblk=1; iblk<nblk; iblk++) {
        blkit[iblk*2+1] += blkit[(iblk-1)*2+1];
    };

    // sweep 2.
    ndcnt = (int*)malloc((size_t)msd->nnode*sizeof(int));
    ndlcnt = (int*)malloc((size_t)msd->nnode*sizeof(int));
    for (ind=0; ind<msd->nnode; ind++) {    // initialize.
        ndcnt[ind] = 0;
    };
    blkit[0] = 0;
    for (iblk=1; iblk<nblk; iblk++) {
        blkit[iblk*2] = blkit[(iblk-1)*2+1];
    };
    for (iblk=0; iblk<nblk; iblk++) {
        for (ind=0; ind<msd->nnode; ind++) {    // initialize.
            ndlcnt[ind] = 0;
        };
        for (icl=blkit[iblk*2]; icl<blkit[iblk*2+1]; icl++) {
            pclnds = msd->clnds + icl*(CLMND+1);
            for (inl=1; inl<=pclnds[0]; inl++) {
                ind = pclnds[inl];
                ndlcnt[ind] = 1;
            };
        };
        for (ind=0; ind<msd->nnode; ind++) {
            ndcnt[ind] += ndlcnt[ind];
        };
    };
    pmndcnt[0] = 0;
    for (ind=0; ind<msd->nnode; ind++) {
        if (pmndcnt[0] < ndcnt[ind]) {
            pmndcnt[0] = ndcnt[ind];
        };
    };

    free(ndlcnt);
    free(ndcnt);
    free(blkit);

    return 0;
};
// vim: set ts=4 et:
