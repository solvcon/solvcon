#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include "march/core/core.hpp"

#include "march/mesh/LookupTable.hpp"

namespace march
{

namespace mesh
{

/**
 * SOLVCON legacy C interface.
 */
struct sc_mesh_t {
    index_type ndim, nnode, nface, ncell, nbound, ngstnode, ngstface, ngstcell;
    // geometry.
    real_type *ndcrd;
    real_type *fccnd;
    real_type *fcnml;
    real_type *fcara;
    real_type *clcnd;
    real_type *clvol;
    // meta.
    shape_type *fctpn;
    shape_type *cltpn;
    index_type *clgrp;
    // connectivity.
    index_type *fcnds;
    index_type *fccls;
    index_type *clnds;
    index_type *clfcs;
}; /* end struct sc_mesh_t */

/**
 * Unstructured mesh of mixed-type elements, optimized for reading.
 */
template< size_t NDIM >
class UnstructuredBlock {

public:

    static constexpr size_t ndim = NDIM;
    static constexpr size_t MAX_FCNND = 4;
    static constexpr size_t FCNCL = 4;
    static constexpr size_t MAX_CLNND = 8;
    static constexpr size_t MAX_CLNFC = 6;

    UnstructuredBlock() = delete;
    UnstructuredBlock(const UnstructuredBlock &) = delete;
    UnstructuredBlock & operator=(const UnstructuredBlock &) = delete;

    ~UnstructuredBlock() {
        /* LookupTable destructor takes care of resource management */
    }

private:

    // dimension.
    index_type nnode;
    index_type nface;
    index_type ncell;
    index_type nbound;
    index_type ngstnode;
    index_type ngstface;
    index_type ngstcell;
    // geometry.
    LookupTable<real_type, NDIM> m_ndcrd;
    LookupTable<real_type, NDIM> m_fccnd;
    LookupTable<real_type, NDIM> m_fcnml;
    LookupTable<real_type, 1> m_fcara;
    LookupTable<real_type, NDIM> m_clcnd;
    LookupTable<real_type, 1> m_clvol;
    // meta.
    LookupTable<shape_type, 1> m_fctpn;
    LookupTable<shape_type, 1> m_cltpn;
    LookupTable<index_type, 1> m_clgrp;
    // connectivity.
    LookupTable<index_type, MAX_FCNND+1> m_fcnds;
    LookupTable<index_type, FCNCL> m_fccls;
    LookupTable<index_type, MAX_CLNND+1> m_clnds;
    LookupTable<index_type, MAX_CLNFC+1> m_clfcs;

}; /* end class UnstructuredBlock */

} /* end namespace mesh */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
