#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <array>
#include <ostream>

#include "march/core.hpp"

#include "march/mesh/UnstructuredBlock.hpp"
#include "march/mesh/ConservationElement/BasicCE.hpp"
#include "march/mesh/ConservationElement/GradientElement.hpp"

namespace march {

/**
 * Geometry data for a compound conservation element (CCE).
 */
template< size_t NDIM >
struct ConservationElement {

    typedef UnstructuredBlock<NDIM> block_type;
    typedef Vector<NDIM> vector_type;

public: // member data

    vector_type cnd; // CCE centroid.
    real_type vol; // CCE volume.
    std::array<BasicCE<NDIM>, block_type::CLMFC> bces;

public:

    ConservationElement(const block_type & block, index_type icl)
    {
#ifdef MH_DEBUG
        init_sentinel();
#endif // MH_DEBUG
        init_from_block(block, icl);
    }

    ConservationElement(const block_type & block, index_type icl, bool init_sentinel) {
        if (init_sentinel) { this->init_sentinel(); }
        init_from_block(block, icl);
    }

    void init_sentinel() {
        cnd = -std::numeric_limits<real_type>::infinity();
        vol = -std::numeric_limits<real_type>::infinity();
        for (index_type ifl=0; ifl<bces.size(); ++ifl) {
            bces[ifl].init_sentinel();
        }
    }

private:

    void init_from_block(const block_type & block, index_type icl) {
        const auto & tclfcs = block.clfcs()[icl];
        cnd = 0;
        vol = 0;
        for (index_type ifl=0; ifl<tclfcs[0]; ++ifl) {
            auto & bce = bces[ifl];
            new (&bce) BasicCE<NDIM>(block, icl, ifl);
            vol += bce.vol;
            cnd += bce.cnd * bce.vol;
        }
        cnd /= vol;
    }

public:

    Vector<NDIM> mirror_centroid(const Vector<NDIM> & tfccnd, const Vector<NDIM> & tfcnml) const {
        Vector<NDIM> ret;
        const auto len = (tfccnd - cnd).dot(tfcnml)*2.0;
        ret = cnd + tfcnml * len;
        return ret;
    }

    std::string repr(size_t indent=0, size_t precision=0) const;

}; /* end struct ConservationElement */

template< size_t NDIM > std::string ConservationElement<NDIM>::repr(
    size_t indent, size_t precision
) const {
    std::string ret(string::format("ConservationElement%ldD(", NDIM));
    const std::string indented_newline = string::create_indented_newline(indent);
    if (indent) { ret += indented_newline; }
    ret += "cnd=" + cnd.repr(indent, precision) + ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "vol=" + string::from_double(vol, precision);
    for (index_type ifc=0; ifc<block_type::CLMFC; ++ifc) {
        ret += ",";
        ret += indent ? indented_newline : std::string(" ");
        ret += string::format("bces[%d]=", ifc);
        ret += string::replace_all_substrings(bces[ifc].repr(indent, precision), "\n", indented_newline);
    }
    if (indent) { ret += "\n)"; }
    else        { ret += ")"; }
    return ret;
}

} /* end namespace march */

template< size_t NDIM >
std::ostream& operator<< (
    std::ostream& stream, const march::ConservationElement<NDIM> & ce
) {
    using namespace march;
    stream << ce.repr();
    return stream;
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
