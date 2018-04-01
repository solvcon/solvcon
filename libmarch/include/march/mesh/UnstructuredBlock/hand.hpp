#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include "march/core.hpp"

#include "march/mesh/UnstructuredBlock/class.hpp"

namespace march {

template< size_t NDIM >
class BlockHandBase {

public:

    using block_type = UnstructuredBlock<NDIM>;
    using vector_type = Vector<NDIM>;

    BlockHandBase(block_type const & block, index_type index) : m_block(&block), m_index(index) {}

    block_type       & block()       { return *m_block; }
    block_type const & block() const { return *m_block; }

    index_type index() const { return m_index; }
    void set_index(index_type index) { m_index = index; }

protected:

    template< class ROW >
    vector_type const & row_as_vector(ROW const & r) const {
        return *reinterpret_cast<vector_type const *>(&r[0]);
    }

private:

    block_type const * m_block = nullptr;
    index_type m_index = MH_INDEX_SENTINEL;

}; /* end class BlockHandBase */

template< size_t NDIM >
class NodeHand : public BlockHandBase< NDIM > {

public:

    using base_type = BlockHandBase<NDIM>;
    using base_type::base_type;

    using block_type = UnstructuredBlock<NDIM>;
    using vector_type = Vector<NDIM>;

    std::string repr(size_t indent=0, size_t precision=0) const;

    vector_type const & crd() const { return this->row_as_vector(this->block().ndcrd()[this->index()]); }

}; /* end class NodeHand */

template< size_t NDIM >
std::string NodeHand<NDIM>::repr(size_t indent, size_t precision) const {
    std::string ret(string::format("NodeHand%ldD(index=%d", NDIM, this->index()));
    ret += ", crd=" + crd().repr(indent, precision) + ")";
    return ret;
}

template< size_t NDIM > class CellHand;

template< size_t NDIM >
class FaceHand : public BlockHandBase< NDIM > {

public:

    using base_type = BlockHandBase<NDIM>;
    using base_type::base_type;

    using block_type = UnstructuredBlock<NDIM>;
    using vector_type = Vector<NDIM>;

    std::string repr(size_t indent=0, size_t precision=0) const;

    index_type tpn() const { return this->block().fctpn()[this->index()]; }

    CellType const & type() const { return celltype(tpn()); }

    vector_type const & cnd() const { return this->row_as_vector(this->block().fccnd()[this->index()]); }
    vector_type const & nml() const { return this->row_as_vector(this->block().fcnml()[this->index()]); }
    real_type ara() const { return this->block().fcara()[this->index()]; }

    index_type nnd() const { return this->block().fcnds()[this->index()][0]; }

    /// Cell that this face belongs to.
    CellHand<NDIM> clb() const;
    /// Cell that this face is neighbor of.
    CellHand<NDIM> cln() const;

    struct boundcheck {};

    /**
     * Get the @a ind -th node index.  @a ind is 1-based.
     */
    index_type nds(index_type ind) const { return this->block().fcnds()[this->index()][ind]; }
    /**
     * Get the @a ind -th node index.  @a ind is 1-based.  Bound checked.
     */
    index_type nds(index_type ind, boundcheck const &) const {
        auto const & fcnds = this->block().fcnds();
        if (ind >= fcnds.ncolumn()) {
            throw std::out_of_range(string::format(
                "in cell %d, %d-th (1-based) node out of range (%d)",
                this->index(), ind, fcnds.ncolumn()));
        }
        return fcnds.at(this->index())[ind];
    }

}; /* end class FaceHand */

template< size_t NDIM >
class CellHand : public BlockHandBase< NDIM > {

public:

    using base_type = BlockHandBase<NDIM>;
    using base_type::base_type;

    using block_type = UnstructuredBlock<NDIM>;
    using vector_type = Vector<NDIM>;

    std::string repr(size_t indent=0, size_t precision=0) const;

    index_type tpn() const { return this->block().cltpn()[this->index()]; }

    CellType const & type() const { return celltype(tpn()); }

    vector_type const & cnd() const {
        return *reinterpret_cast<vector_type const *>(&(this->block().clcnd()[this->index()][0]));
    }

    real_type vol() const { return this->block().clvol()[this->index()]; }

    index_type nnd() const { return this->block().clnds()[this->index()][0]; }

    struct boundcheck {};

    /**
     * Get the @a ind -th node index.  @a ind is 1-based.
     */
    index_type nds(index_type ind) const { return this->block().clnds()[this->index()][ind]; }
    /**
     * Get the @a ind -th node index.  @a ind is 1-based.  Bound checked.
     */
    index_type nds(index_type ind, boundcheck const &) const {
        auto const & clnds = this->block().clnds();
        if (ind >= clnds.ncolumn()) {
            throw std::out_of_range(string::format(
                "in cell %d, %d-th (1-based) node out of range (%d)",
                this->index(), ind, clnds.ncolumn()));
        }
        return clnds.at(this->index())[ind];
    }

    index_type nfc() const { return this->block().clfcs()[this->index()][0]; }

    /**
     * Get the @a ifc -th face index.  @a ifc is 1-based.
     */
    index_type fcs(index_type ifc) const { return this->block().clfcs()[this->index()][ifc]; }
    /**
     * Get the @a ifc -th face index.  @a ifc is 1-based.  Bound checked.
     */
    index_type fcs(index_type ifc, boundcheck const &) const {
        auto const & clfcs = this->block().clfcs();
        if (ifc >= clfcs.ncolumn()) {
            throw std::out_of_range(string::format(
                "in cell %d, %d-th (1-based) face out of range (%d)",
                this->index(), ifc, clfcs.ncolumn()));
        }
        return clfcs.at(this->index())[ifc];
    }

}; /* end class CellHand */

template< size_t NDIM >
std::string CellHand<NDIM>::repr(size_t indent, size_t precision) const {
    std::string ret(string::format("CellHand%ldD(", NDIM));
    const std::string indented_newline = string::create_indented_newline(indent);
    const std::string indented2_newline = string::create_indented_newline(indent*2);
    if (indent) { ret += indented_newline; }
    ret += string::format("index=%d,", this->index());
    ret += indent ? indented_newline : std::string(" ");
    ret += string::format("type=%d:%s,", tpn(), type().name());
    ret += indent ? indented_newline : std::string(" ");
    ret += "cnd=" + cnd().repr(indent, precision) + ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "vol=" + string::from_double(vol(), precision) + ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "nds=[";
    if (indent) { ret += indented2_newline; }
    for (index_type ind=1; ind<=nnd(); ++ind) {
        ret += NodeHand<NDIM>(this->block(), nds(ind)).repr(indent, precision);
        if (nnd() == ind) {
            if (indent) { ret += indented_newline; }
            ret += "]";
        } else {
            ret += ",";
            ret += indent ? indented2_newline : std::string(" ");
        }
    }
    ret += ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "fcs=[";
    for (index_type ifc=1; ifc<=nfc(); ++ifc) {
        auto const fc = FaceHand<NDIM>(this->block(), fcs(ifc));
        ret += string::format("%d:(", fc.index());
        for (index_type ind=1; ind<=fc.nnd(); ++ind) {
            ret += string::format("%d", fc.nds(ind));
            ret += fc.nnd() == ind ? ")" : ",";
        }
        ret += nfc() == ifc ? "]" : ", ";
    }
    ret += indent ? "\n)" : ")";
    return ret;
}

template< size_t NDIM >
CellHand<NDIM> FaceHand<NDIM>::clb() const {
    auto const & block = this->block();
    return CellHand<NDIM>(block, block.fccls()[this->index()][0]);
}

template< size_t NDIM >
CellHand<NDIM> FaceHand<NDIM>::cln() const {
    auto const & block = this->block();
    return CellHand<NDIM>(block, block.fccls()[this->index()][1]);
}

template< size_t NDIM >
std::string FaceHand<NDIM>::repr(size_t indent, size_t precision) const {
    std::string ret(string::format("FaceHand%ldD(", NDIM));
    const std::string indented_newline = string::create_indented_newline(indent);
    const std::string indented2_newline = string::create_indented_newline(indent*2);
    if (indent) { ret += indented_newline; }
    ret += string::format("index=%d,", this->index());
    ret += indent ? indented_newline : std::string(" ");
    ret += string::format("type=%d:%s,", tpn(), type().name());
    ret += indent ? indented_newline : std::string(" ");
    auto const clb = this->clb();
    ret += string::format("belong_cell=%d;%d:%s,", clb.index(), clb.tpn(), clb.type().name());
    ret += indent ? indented_newline : std::string(" ");
    auto const cln = this->cln();
    ret += string::format("neighbor_cell=%d;%d:%s,", cln.index(), cln.tpn(), cln.type().name());
    ret += indent ? indented_newline : std::string(" ");
    ret += "cnd=" + cnd().repr(indent, precision) + ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "nml=" + nml().repr(indent, precision) + ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "ara=" + string::from_double(ara(), precision) + ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "nds=[";
    if (indent) { ret += indented2_newline; }
    for (index_type ind=1; ind<=nnd(); ++ind) {
        ret += NodeHand<NDIM>(this->block(), nds(ind)).repr(indent, precision);
        if (nnd() == ind) {
            if (indent) { ret += indented_newline; }
            ret += "]";
        } else {
            ret += ",";
            ret += indent ? indented2_newline : std::string(" ");
        }
    }
    ret += indent ? "\n)" : ")";
    return ret;
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
