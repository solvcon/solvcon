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
 * subroutine get_faces_from_cells: Extract interier faces from node list of
 * cells.  Subroutine is designed to handle all types of cell.  See block.py
 * for the types to be supported.
 */
int get_faces_from_cells(MeshData *msd, int mface,
        int *pnface, int *clfcs, int *fctpn, int *fcnds, int *fccls) {
    // pointers.
    int *pcltpn, *pclnds, *pclfcs, *pfctpn, *pfcnds, *pfccls;
    int *pifctpn, *pjfctpn, *pifcnds, *pjfcnds;
    int *pndfcs;
    // buffers.
    int *ndnfc, *ndfcs, *map, *map2;
    // scalars.
    int tpnicl, ndmfc, cond;
    // iterator.
    int icl, ifc, jfc, inf, jnf, ind, nd1, ifl;
    int it;

    // extract face definition from the node list of cells.
    pcltpn = msd->cltpn;
    pclnds = msd->clnds;
    pclfcs = clfcs;
    pfctpn = fctpn;
    pfcnds = fcnds;
    ifc = 0;
    for (icl=0; icl<msd->ncell; icl++) {
        tpnicl = pcltpn[0];
        // parse each type of cell.
        if (tpnicl == 0) {
        } else if (tpnicl == 1) {   // line.
            // extract 2 points from a line.
            pclfcs[0] = 2;
            for (it=0; it<pclfcs[0]; it++) {
                pfctpn[it] = 0; // face type is point.
                pfcnds[it*(FCMND+1)] = 1;   // number of nodes per face.
            };
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
        } else if (tpnicl == 2) {   // quadrilateral.
            // extract 4 lines from a quadrilateral.
            pclfcs[0] = 4;
            for (it=0; it<pclfcs[0]; it++) {
                pfctpn[it] = 1; // face type is line.
                pfcnds[it*(FCMND+1)] = 2;   // number of nodes per face.
            };
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[4];
            pfcnds[2] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
        } else if (tpnicl == 3) {   // triangle.
            // extract 3 lines from a triangle.
            pclfcs[0] = 3;
            for (it=0; it<pclfcs[0]; it++) {
                pfctpn[it] = 1; // face type is line.
                pfcnds[it*(FCMND+1)] = 2;   // number of nodes per face.
            };
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
        } else if (tpnicl == 4) {   // hexahedron.
            // extract 6 quadrilaterals from a hexahedron.
            pclfcs[0] = 6;
            for (it=0; it<pclfcs[0]; it++) {
                pfctpn[it] = 2; // face type is quadrilateral.
                pfcnds[it*(FCMND+1)] = 4;   // number of nodes per face.
            };
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[3];
            pfcnds[4] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[7];
            pfcnds[4] = pclnds[6];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[5];
            pfcnds[2] = pclnds[6];
            pfcnds[3] = pclnds[7];
            pfcnds[4] = pclnds[8];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[8];
            pfcnds[4] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 5.
            pclfcs[5] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds[3] = pclnds[6];
            pfcnds[4] = pclnds[5];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 6.
            pclfcs[6] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[8];
            pfcnds[4] = pclnds[7];
            pfcnds += FCMND+1;
            ifc += 1;
        } else if (tpnicl == 5) {   // tetrahedron.
            // extract 4 triangles from a tetrahedron.
            pclfcs[0] = 4;
            for (it=0; it<pclfcs[0]; it++) {
                pfctpn[it] = 3; // face type is triangle.
                pfcnds[it*(FCMND+1)] = 3;   // number of nodes per face.
            };
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds[3] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
        } else if (tpnicl == 6) {   // prism.
            // extract 2 triangles and 3 quadrilaterals from a prism.
            pclfcs[0] = 5;
            for (it=0; it<2; it++) {
                pfctpn[it] = 3; // face type is triangle.
                pfcnds[it*(FCMND+1)] = 3;   // number of nodes per face.
            };
            for (it=2; it<pclfcs[0]; it++) {
                pfctpn[it] = 2; // face type is quadrilateral.
                pfcnds[it*(FCMND+1)] = 4;   // number of nodes per face.
            };
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[2];
            pfcnds[3] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[4];
            pfcnds[2] = pclnds[6];
            pfcnds[3] = pclnds[5];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[5];
            pfcnds[4] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[3];
            pfcnds[3] = pclnds[6];
            pfcnds[4] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 5.
            pclfcs[5] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[6];
            pfcnds[4] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
        } else if (tpnicl == 7) {   // pyramid.
            // extract 4 triangles and 1 quadrilaterals from a pyramid.
            pclfcs[0] = 5;
            for (it=0; it<4; it++) {
                pfctpn[it] = 3; // face type is triangle.
                pfcnds[it*(FCMND+1)] = 3;   // number of nodes per face.
            };
            for (it=4; it<pclfcs[0]; it++) {
                pfctpn[it] = 2; // face type is quadrilateral.
                pfcnds[it*(FCMND+1)] = 4;   // number of nodes per face.
            };
            pfctpn += pclfcs[0];
            // face 1.
            pclfcs[1] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[4];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 2.
            pclfcs[2] = ifc;
            pfcnds[1] = pclnds[2];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[1];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 3.
            pclfcs[3] = ifc;
            pfcnds[1] = pclnds[3];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 4.
            pclfcs[4] = ifc;
            pfcnds[1] = pclnds[4];
            pfcnds[2] = pclnds[5];
            pfcnds[3] = pclnds[3];
            pfcnds += FCMND+1;
            ifc += 1;
            // face 5.
            pclfcs[5] = ifc;
            pfcnds[1] = pclnds[1];
            pfcnds[2] = pclnds[4];
            pfcnds[3] = pclnds[3];
            pfcnds[4] = pclnds[2];
            pfcnds += FCMND+1;
            ifc += 1;
        };
        // advance pointers.
        pcltpn += 1;
        pclnds += CLMND+1;
        pclfcs += CLMFC+1;
    };

    // build the hash table, to know what faces connect to each node.
    /// first pass: get the maximum number of faces.
    ndnfc = (int *)malloc((size_t)msd->nnode*sizeof(int));
    for (ind=0; ind<msd->nnode; ind++) {    // initialize.
        ndnfc[ind] = 0;
    };
    pfcnds = fcnds; // count.
    for (ifc=0; ifc<mface; ifc++) {
        for (inf=1; inf<=pfcnds[0]; inf++) {
            ind = pfcnds[inf];  // node of interest.
            ndnfc[ind] += 1;    // increment counting.
        };
        // advance pointers.
        pfcnds += FCMND+1;
    };
    ndmfc = 0;  // get maximum.
    for (ind=0; ind<msd->nnode; ind++) {
        if (ndnfc[ind] > ndmfc) {
            ndmfc = ndnfc[ind];
        };
    };
    free(ndnfc);
    /// second pass: scan again to build hash table.
    ndfcs = (int *)malloc((size_t)msd->nnode*(ndmfc+1)*sizeof(int));
    pndfcs = ndfcs; // initialize.
    for (ind=0; ind<msd->nnode; ind++) {
        pndfcs[0] = 0;
        for (it=1; it<ndmfc; it++) {
            pndfcs[it] = -1;
        };
        // advance pointers;
        pndfcs += ndmfc+1;
    };
    pfcnds = fcnds; // build hash table mapping from node to face.
    for (ifc=0; ifc<mface; ifc++) {
        for (inf=1; inf<=pfcnds[0]; inf++) {
            ind = pfcnds[inf];  // node of interest.
            pndfcs = ndfcs + ind*(ndmfc+1);
            pndfcs[0] += 1; // increment face count for the node.
            pndfcs[pndfcs[0]] = ifc;
        };
        // advance pointers.
        pfcnds += FCMND+1;
    };

    // scan for duplicated faces and build duplication map.
    map = (int *)malloc((size_t)mface*sizeof(int));
    for (ifc=0; ifc<mface; ifc++) { // initialize.
        map[ifc] = ifc;
    };
    for (ifc=0; ifc<mface; ifc++) {
        if (map[ifc] == ifc) {
            pifcnds = fcnds + ifc*(FCMND+1);
            nd1 = pifcnds[1];    // take only the FIRST node of a face.
            pndfcs = ndfcs + nd1*(ndmfc+1);
            for (it=1; it<=pndfcs[0]; it++) {
                jfc = pndfcs[it];
                // test for duplication.
                if ((jfc != ifc) && (fctpn[jfc] == fctpn[ifc])) {
                    pjfcnds = fcnds + jfc*(FCMND+1);
                    cond = pjfcnds[0];
                    // scan all nodes in ifc and jfc to see if all the same.
                    for (jnf=1; jnf<=pjfcnds[0]; jnf++) {
                        for (inf=1; inf<=pifcnds[0]; inf++) {
                            if (pjfcnds[jnf] == pifcnds[inf]) {
                                cond -= 1;
                                break;
                            };
                        };
                    };
                    if (cond == 0) {
                        map[jfc] = ifc;  // record duplication.
                    };
                };
            };
        };
    };

    // use the duplication map to remap nodes in faces, and build renewed map.
    map2 = (int *)malloc((size_t)mface*sizeof(int));
    pifcnds = fcnds;
    pjfcnds = fcnds;
    pifctpn = fctpn;
    pjfctpn = fctpn;
    jfc = 0;
    for (ifc=0; ifc<mface; ifc++) {
        if (map[ifc] == ifc) {
            for (inf=0; inf<=FCMND; inf++) {
                pjfcnds[inf] = pifcnds[inf];
            };
            pjfctpn[0] = pifctpn[0];
            map2[ifc] = jfc;
            // increment j-face.
            jfc += 1;
            pjfcnds += FCMND+1;
            pjfctpn += 1;
        } else {
            map2[ifc] = map2[map[ifc]];
        };
        // advance pointers;
        pifcnds += FCMND+1;
        pifctpn += 1;
    };
    pnface[0] = jfc;    // record deduplicated number of face.

    // rebuild faces in cells and build face neighboring, according to the
    // renewed face map.
    pfccls = fccls; // initialize.
    for (ifc=0; ifc<mface; ifc++) {
        for (it=0; it<FCREL; it++) {
            pfccls[it] = -1;
        };
        // advance pointers;
        pfccls += FCREL;
    };
    pclfcs = clfcs;
    for (icl=0; icl<msd->ncell; icl++) {
        for (ifl=1; ifl<=pclfcs[0]; ifl++) {
            ifc = pclfcs[ifl];
            jfc = map2[ifc];
            // rebuild faces in cells.
            pclfcs[ifl] = jfc;
            // build face neighboring.
            pfccls = fccls + jfc*FCREL;
            if (pfccls[0] == -1) {
                pfccls[0] = icl;
            } else if (pfccls[1] == -1) {
                pfccls[1] = icl;
            };
        };
        // advance pointers;
        pclfcs += CLMFC+1;
    };

    free(map2);
    free(map);
    free(ndfcs);

    return 0;
};
// vim: set ts=4 et:
