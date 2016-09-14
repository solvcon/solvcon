#pragma once

/*
 * Copyright (c) 2011, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cmath>

#include "march/mesh/UnstructuredBlock/class.hpp"

namespace march
{

/**
 * \param[in]  rcells
 * \param[out] adjncy
 */
template< size_t NDIM > void UnstructuredBlock< NDIM >::build_csr(
    const LookupTable<index_type, CLMFC> & rcells
  , LookupTable<index_type, 0> & adjncy)
const {
    // iterators.
    int icl, ifl, ieg;

    // fill.
    index_type const * prcells = reinterpret_cast<index_type const *>(rcells.row(0));
    index_type       * padjncy = reinterpret_cast<index_type       *>(adjncy.row(0));
    ieg = 0;
    for (icl=0; icl<ncell(); icl++) {
        for (ifl=0; ifl<CLMFC; ifl++) {
            if (prcells[ifl] != -1) {
                padjncy[ieg] = prcells[ifl];
                ieg += 1;
            };
        };
        // advance pointers.
        prcells += CLMFC;
    };
};

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
