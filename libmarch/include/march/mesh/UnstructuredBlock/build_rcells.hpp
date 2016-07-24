#pragma once

/*
 * Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cmath>

#include "march/mesh/UnstructuredBlock/class.hpp"

namespace march
{

namespace mesh
{

/**
 * \param[out] rcells
 * \param[out] rcellno
 */
template< size_t NDIM > void UnstructuredBlock< NDIM >::build_rcells(
    LookupTable<index_type, CLMFC> & rcells
  , LookupTable<index_type, 0> & rcellno)
const {
    // pointers.
    index_type *prcells;
    // iterators.
    index_type icl, ifl, ifl1, ifc;

    // initialize.
    prcells = reinterpret_cast<index_type *>(rcells.row(0));
    for (icl=0; icl<ncell(); icl++) {
        for (ifl=0; ifl<CLMFC; ifl++) {
            prcells[ifl] = -1;
        };
        rcellno[icl] = 0;
        // advance pointers.
        prcells += CLMFC;
    };
    
    // count.
    const index_type * pclfcs = reinterpret_cast<const index_type *>(clfcs().row(0));
    prcells = reinterpret_cast<index_type *>(rcells.row(0));
    for (icl=0; icl<ncell(); icl++) {
        for (ifl=1; ifl<=pclfcs[0]; ifl++) {
            ifl1 = ifl-1;
            ifc = pclfcs[ifl];
            const index_type * pfccls = reinterpret_cast<const index_type *>(fccls().row(0)) + ifc*FCREL;
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
}

} /* end namespace mesh */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
