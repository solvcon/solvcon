#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <array>

#include "march/core/core.hpp"

#include "march/mesh/UnstructuredBlock/class.hpp"
#include "march/mesh/BasicCE.hpp"

namespace march {

/**
 * Geometry data for a compound conservation element (CCE).
 */
template< size_t NDIM >
struct CompoundCE {

    typedef UnstructuredBlock<NDIM> block_type;
    typedef Vector<NDIM> vector_type;

    vector_type cnd; // CCE centroid.
    real_type vol; // CCE volume.
    std::array<BasicCE<NDIM>, block_type::FCMND> bces;

    CompoundCE() = default;
    CompoundCE(const block_type & block, index_type icl) {
        const auto & tclfcs = block.clfcs()[icl];
        vol = 0;
        cnd = 0;
        for (index_type ifl=0; ifl<tclfcs[0]; ++ifl) {
            auto & bce = bces[ifl];
            new (&bce) BasicCE<NDIM>(block, icl, ifl);
            vol += bce.vol;
            cnd += bce.cnd * bce.vol;
        }
        cnd /= vol;
    }

    Vector<NDIM> mirror_centroid(const Vector<NDIM> & tfccnd, const Vector<NDIM> & tfcnml) const {
        Vector<NDIM> ret;
        const auto len = (tfccnd - cnd).dot(tfcnml)*2.0;
        ret = cnd + tfcnml * len;
        return ret;
    }

}; /* end struct CompoundCE */

template< size_t NDIM >
struct CompoundCEArray {

};

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
