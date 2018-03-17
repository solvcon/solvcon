#pragma once

/*
 * Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cmath>
#include <stdlib.h>

#include "march/mesh/UnstructuredBlock/class.hpp"

namespace march
{

/**
 * Extract interier faces from node list of cells.  Subroutine is designed to
 * handle all types of cells.
 */
template< size_t NDIM >
void UnstructuredBlock< NDIM >::build_faces_from_cells() {
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

    const index_type mface = calc_max_nface(cltpn());
    index_type computed_nface = -1;

    // create temporary tables.
    LookupTable<index_type, CellType::CLNFC_MAX+1> tclfcs(0, ncell());
    LookupTable<index_type,           0> tfctpn(0, mface);
    LookupTable<index_type, CellType::FCNND_MAX+1> tfcnds(0, mface);
    LookupTable<index_type,     FCNCL  > tfccls(0, mface);
    tclfcs.fill(-1); tfcnds.fill(-1); tfccls.fill(-1);
    index_type * lclfcs = reinterpret_cast<index_type *>(tclfcs.row(0));
    index_type * lfctpn = reinterpret_cast<shape_type *>(tfctpn.row(0));
    index_type * lfcnds = reinterpret_cast<index_type *>(tfcnds.row(0));
    index_type * lfccls = reinterpret_cast<index_type *>(tfccls.row(0));

    // extract face definition from the node list of cells.
    pcltpn = reinterpret_cast<index_type *>(cltpn().row(0));
    pclnds = reinterpret_cast<index_type *>(clnds().row(0));
    pclfcs = lclfcs;
    pfctpn = lfctpn;
    pfcnds = lfcnds;
    ifc = 0;
    for (icl=0; icl<ncell(); icl++) {
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
    ndnfc = (int *)malloc((size_t)nnode()*sizeof(int));
    for (ind=0; ind<nnode(); ind++) {    // initialize.
        ndnfc[ind] = 0;
    };
    pfcnds = lfcnds; // count.
    for (ifc=0; ifc<mface; ifc++) {
        for (inf=1; inf<=pfcnds[0]; inf++) {
            ind = pfcnds[inf];  // node of interest.
            ndnfc[ind] += 1;    // increment counting.
        };
        // advance pointers.
        pfcnds += FCMND+1;
    };
    ndmfc = 0;  // get maximum.
    for (ind=0; ind<nnode(); ind++) {
        if (ndnfc[ind] > ndmfc) {
            ndmfc = ndnfc[ind];
        };
    };
    free(ndnfc);
    /// second pass: scan again to build hash table.
    ndfcs = (int *)malloc((size_t)nnode()*(ndmfc+1)*sizeof(int));
    pndfcs = ndfcs; // initialize.
    for (ind=0; ind<nnode(); ind++) {
        pndfcs[0] = 0;
        for (it=1; it<=ndmfc; it++) { // <= or < ??
            pndfcs[it] = -1;
        };
        // advance pointers;
        pndfcs += ndmfc+1;
    };
    pfcnds = lfcnds; // build hash table mapping from node to face.
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
            pifcnds = lfcnds + ifc*(FCMND+1);
            nd1 = pifcnds[1];    // take only the FIRST node of a face.
            pndfcs = ndfcs + nd1*(ndmfc+1);
            for (it=1; it<=pndfcs[0]; it++) {
                jfc = pndfcs[it];
                // test for duplication.
                if ((jfc != ifc) && (lfctpn[jfc] == lfctpn[ifc])) {
                    pjfcnds = lfcnds + jfc*(FCMND+1);
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
    pifcnds = lfcnds;
    pjfcnds = lfcnds;
    pifctpn = lfctpn;
    pjfctpn = lfctpn;
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
    computed_nface = jfc;    // record deduplicated number of face.

    // rebuild faces in cells and build face neighboring, according to the
    // renewed face map.
    pfccls = lfccls; // initialize.
    for (ifc=0; ifc<mface; ifc++) {
        for (it=0; it<FCREL; it++) {
            pfccls[it] = -1;
        };
        // advance pointers;
        pfccls += FCREL;
    };
    pclfcs = lclfcs;
    for (icl=0; icl<ncell(); icl++) {
        for (ifl=1; ifl<=pclfcs[0]; ifl++) {
            ifc = pclfcs[ifl];
            jfc = map2[ifc];
            // rebuild faces in cells.
            pclfcs[ifl] = jfc;
            // build face neighboring.
            pfccls = lfccls + jfc*FCREL;
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

    // recreate member tables.
    set_nface(computed_nface);
    create_fctpn(0, nface());
    create_fcnds(0, nface());
    create_fccls(0, nface());
    for (icl=0; icl < ncell(); ++icl) {
        clfcs().set(icl, tclfcs[icl]);
    }
    for (ifc=0; ifc < nface(); ++ifc) {
        fctpn().set(ifc, tfctpn[ifc]);
        fcnds().set(ifc, tfcnds[ifc]);
        fccls().set(ifc, tfccls[ifc]);
    }
    create_fccnd(0, nface());
    create_fcnml(0, nface());
    create_fcara(0, nface());
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
