#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <vector>
#include <type_traits>
#include <functional>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <memory>

#include "march/depend/scotch.hpp"

#include "march/core.hpp"

#include "march/mesh/BoundaryData.hpp"
#include "march/mesh/CellType.hpp"

namespace march {

/**
 * Unstructured mesh of mixed-type elements, optimized for reading.
 *
 * This class is managed using std::shared_ptr.  Consumers should get a const
 * reference when a const object is desired, otherwise get a shared pointer of
 * non-const.
 *
 * Question: Should I have the qualifier "unstructured"?  Will there ever be
 * non-unstructured block?
 */
template< size_t NDIM >
class UnstructuredBlock
  : public std::enable_shared_from_this<UnstructuredBlock<NDIM>>
{

public:

    static_assert(2 == NDIM || 3 == NDIM, "not 2 or 3 dimensional");

    static constexpr index_type FCMND = CellType::FCNND_MAX;
    static constexpr index_type CLMND = CellType::CLNND_MAX;
    static constexpr index_type CLMFC = CellType::CLNFC_MAX;
    static constexpr index_type FCNCL = 4;
    static constexpr index_type FCREL = 4;
    static constexpr index_type BFREL = BoundaryData::BFREL;

    // TODO: move to UnstructuredBlock.
    // @[
    void locate_point(const real_type (& crd)[NDIM]) const;

    // moved to mesh: void prepare_ce();
    // moved to mesh: void prepare_sf();
    // @]

    static index_type calc_max_nface(const LookupTable<index_type, 0> & cltpn) {
        index_type max_nfc = 0;
        for (index_type it=0; it<cltpn.nbody(); ++it) {
            max_nfc += celltype(cltpn[it]).nface();
        }
        return max_nfc;
    }

/* data declaration */
private:

    // shape.
    index_type m_nnode = 0; ///< Number of nodes (interior).
    index_type m_nface = 0; ///< Number of faces (interior).
    index_type m_ncell = 0; ///< Number of cells (interior).
    index_type m_nbound = 0; ///< Number of boundary faces.
    index_type m_ngstnode = 0; ///< Number of ghost nodes.
    index_type m_ngstface = 0; ///< Number of ghost faces.
    index_type m_ngstcell = 0; ///< Number of ghost cells.
    // other block information.
    bool m_use_incenter = false; ///< While true, m_clcnd uses in-center for simplices.
    // geometry arrays.
    LookupTable<real_type, NDIM> m_ndcrd; ///< Nodes' coordinates.
    LookupTable<real_type, NDIM> m_fccnd; ///< Faces' centroids.
    LookupTable<real_type, NDIM> m_fcnml; ///< Faces' unit normal vectors.
    LookupTable<real_type,    0> m_fcara; ///< Faces' area.
    LookupTable<real_type, NDIM> m_clcnd; ///< Cells's centers (centroids or in-centers).
    LookupTable<real_type,    0> m_clvol; ///< Cells' volume.
    // meta arrays.
    LookupTable<shape_type, 0> m_fctpn; ///< Faces' type numbers.
    LookupTable<shape_type, 0> m_cltpn; ///< Cells' type numbers.
    LookupTable<index_type, 0> m_clgrp; ///< Cells' group numbers.
    // connectivity arrays.
    LookupTable<index_type, FCMND+1> m_fcnds; ///< Faces' nodes.
    LookupTable<index_type, FCNCL  > m_fccls; ///< Faces' cells.
    LookupTable<index_type, CLMND+1> m_clnds; ///< Cells' nodes.
    LookupTable<index_type, CLMFC+1> m_clfcs; ///< Cells' faces.
    // boundary information.
    LookupTable<index_type, 2> m_bndfcs;
    std::vector<BoundaryData> m_bndvec;

/* end data declaration */

/* constructors and desctructor */
public:

    class ctor_passkey {
    private:
        ctor_passkey() = default;
        friend UnstructuredBlock<NDIM>;
    };

    UnstructuredBlock(
        const ctor_passkey &
      , index_type nnode, index_type nface, index_type ncell, index_type nbound
      , index_type ngstnode, index_type ngstface, index_type ngstcell
      , bool use_incenter
    ) : std::enable_shared_from_this<UnstructuredBlock<NDIM>>()
      , m_nnode(nnode), m_nface(nface), m_ncell(ncell), m_nbound(nbound)
      , m_ngstnode(ngstnode), m_ngstface(ngstface), m_ngstcell(ngstcell)
      , m_use_incenter(use_incenter)
    {
        build_tables();
    }

    UnstructuredBlock() = delete;
    UnstructuredBlock(UnstructuredBlock const & ) = delete;
    UnstructuredBlock(UnstructuredBlock       &&) = delete;
    UnstructuredBlock & operator=(UnstructuredBlock const & ) = delete;
    UnstructuredBlock & operator=(UnstructuredBlock       &&) = delete;

    ~UnstructuredBlock() { /* LookupTable destructor takes care of resource management */ }

    static std::shared_ptr<UnstructuredBlock> construct(
        index_type nnode, index_type nface, index_type ncell, index_type nbound
      , index_type ngstnode, index_type ngstface, index_type ngstcell
      , bool use_incenter
    ) {
        return std::make_shared<UnstructuredBlock>(
            ctor_passkey(), nnode, nface, ncell, nbound, ngstnode, ngstface, ngstcell, use_incenter
        );
    }

    static std::shared_ptr<UnstructuredBlock> construct(
        index_type nnode, index_type nface, index_type ncell, bool use_incenter
    ) {
        return construct(nnode, nface, ncell, 0, 0, 0, 0, use_incenter);
    }

    static std::shared_ptr<UnstructuredBlock> construct() {
        return construct(0, 0, 0, 0, 0, 0, 0, false);
    }

#undef MARCH_USTBLOCK_TABLE_DECL_SWAP

/* end constructors and desctructor */

/* block shape accessors */
public:

    index_type ndim() const { return NDIM; }

#define MARCH_USTBLOCK_SHAPE_DECL_METHODS(NAME) \
    \
    index_type NAME() const { return m_##NAME; } \
    \
    void set_##NAME(index_type NAME##_in) { m_##NAME = NAME##_in; }
    // end define MARCH_USTBLOCK_SHAPE_DECL_METHODS

    MARCH_USTBLOCK_SHAPE_DECL_METHODS(nnode)
    MARCH_USTBLOCK_SHAPE_DECL_METHODS(nface)
    MARCH_USTBLOCK_SHAPE_DECL_METHODS(ncell)
    MARCH_USTBLOCK_SHAPE_DECL_METHODS(nbound)
    MARCH_USTBLOCK_SHAPE_DECL_METHODS(ngstnode)
    MARCH_USTBLOCK_SHAPE_DECL_METHODS(ngstcell)
    MARCH_USTBLOCK_SHAPE_DECL_METHODS(ngstface)

#undef MARCH_USTBLOCK_SHAPE_DECL_METHODS

/* end block shape accessors */

/* other information */
public:

    bool use_incenter() const { return m_use_incenter; }
    void set_use_incenter(bool use_incenter_in) { m_use_incenter = use_incenter_in; }
/* end other information */

/* table methods */
public:

#define MARCH_USTBLOCK_TABLE_DECL_METHODS(NAME, ELEMTYPE, NDIM) \
    \
public: \
    \
    LookupTable<ELEMTYPE, NDIM> const & NAME() const { return m_##NAME; } \
    \
    LookupTable<ELEMTYPE, NDIM>       & NAME()       { return m_##NAME; } \
    \
private: \
    \
    LookupTable<ELEMTYPE, NDIM> & create_##NAME( \
        index_type nghost, index_type nbody \
    ) { \
        m_##NAME = LookupTable<ELEMTYPE, NDIM>(nghost, nbody); \
        return m_##NAME; \
    }
    // end define MARCH_USTBLOCK_TABLE_DECL_METHODS

    // geometry array accessors.
    MARCH_USTBLOCK_TABLE_DECL_METHODS(ndcrd, real_type, NDIM)
    MARCH_USTBLOCK_TABLE_DECL_METHODS(fccnd, real_type, NDIM)
    MARCH_USTBLOCK_TABLE_DECL_METHODS(fcnml, real_type, NDIM)
    MARCH_USTBLOCK_TABLE_DECL_METHODS(fcara, real_type,    0)
    MARCH_USTBLOCK_TABLE_DECL_METHODS(clcnd, real_type, NDIM)
    MARCH_USTBLOCK_TABLE_DECL_METHODS(clvol, real_type,    0)
    // meta array accessors.
    MARCH_USTBLOCK_TABLE_DECL_METHODS(fctpn, shape_type, 0)
    MARCH_USTBLOCK_TABLE_DECL_METHODS(cltpn, shape_type, 0)
    MARCH_USTBLOCK_TABLE_DECL_METHODS(clgrp, shape_type, 0)
    // connectivity array accessors.
    MARCH_USTBLOCK_TABLE_DECL_METHODS(fcnds, index_type, FCMND+1)
    MARCH_USTBLOCK_TABLE_DECL_METHODS(fccls, index_type, FCNCL  )
    MARCH_USTBLOCK_TABLE_DECL_METHODS(clnds, index_type, CLMND+1)
    MARCH_USTBLOCK_TABLE_DECL_METHODS(clfcs, index_type, CLMFC+1)
    // boundary information.
    MARCH_USTBLOCK_TABLE_DECL_METHODS(bndfcs, index_type, 2)

#undef MARCH_USTBLOCK_TABLE_DECL_METHODS

/* end table methods */

public:

    std::vector<BoundaryData> const & bndvec() const { return m_bndvec; }
    std::vector<BoundaryData>       & bndvec()       { return m_bndvec; }

    /**
     * Get the "self" cell number of the input face by index.  A shorthand of
     * fccls()[ifc][0] .
     *
     * @param[in] ifc index of the face of interest.
     * @return        index of the cell.
     */
    index_type fcicl(index_type ifc) const { return fccls()[ifc][0]; }

    /**
     * Get the "related" cell number of the input face by index.  A shorthand
     * of fccls()[ifc][1] .
     *
     * @param[in] ifc index of the face of interest.
     * @return        index of the cell.
     */
    index_type fcjcl(index_type ifc) const { return fccls()[ifc][1]; }

    /**
     * Get the other cell number related by the input face.
     *
     * @param[in] ifc index of the face of interest.
     * @param[in] icl index of the "self" cell of interest.
     * @return        index of the cell.
     */
    index_type fcrcl(index_type ifc, index_type icl) const {
        const auto & tfccls = fccls()[ifc];
        return tfccls[0] + tfccls[1] - icl;
    }

    /**
     * Get the rotation matrix in the face normal direction.
     *
     * @param[in] ifc index of the face of interest.
     * @return        the rotation matrix.
     */
    Matrix<NDIM> get_normal_matrix(index_type const ifc) const;

/* data_processors */
public:

    void build_tables() {
        // geometry arrays.
        create_ndcrd(ngstnode(), nnode());
        create_fccnd(ngstface(), nface());
        create_fcnml(ngstface(), nface());
        create_fcara(ngstface(), nface());
        create_clcnd(ngstcell(), ncell());
        create_clvol(ngstcell(), ncell());
        // meta arrays.
        create_fctpn(ngstface(), nface());
        create_cltpn(ngstcell(), ncell());
        create_clgrp(ngstcell(), ncell());
        clgrp().fill(-1);
        // connectivity arrays.
        create_fcnds(ngstface(), nface());
        create_fccls(ngstface(), nface());
        create_clnds(ngstcell(), ncell());
        create_clfcs(ngstcell(), ncell());
        fcnds().fill(-1);
        fccls().fill(-1);
        clnds().fill(-1);
        clfcs().fill(-1);
        // boundary information.
        create_bndfcs(0, nbound());
    }

    void calc_metric();

    void build_interior() {
        build_faces_from_cells();
        calc_metric();
    }

    void build_boundary();

    void build_ghost() {
        std::tie(m_ngstnode, m_ngstface, m_ngstcell) = count_ghost();

        // geometry arrays.
        m_ndcrd.resize(ngstnode(), nnode());
        m_fccnd.resize(ngstface(), nface());
        m_fcnml.resize(ngstface(), nface());
        m_fcara.resize(ngstface(), nface());
        m_clcnd.resize(ngstcell(), ncell());
        m_clvol.resize(ngstcell(), ncell());
        // meta arrays.
        m_fctpn.resize(ngstface(), nface());
        m_cltpn.resize(ngstcell(), ncell());
        m_clgrp.resize(ngstcell(), ncell(), -1);
        // connectivity arrays.
        m_fcnds.resize(ngstface(), nface(), -1);
        m_fccls.resize(ngstface(), nface(), -1);
        m_clnds.resize(ngstcell(), ncell(), -1);
        m_clfcs.resize(ngstcell(), ncell(), -1);

        fill_ghost();
    }

    std::tuple<march::depend::scotch::num_type, LookupTable<index_type, 0>>
    partition(index_type npart) const;

/* end data_processors */

/* data_report */
public:

    std::string info_string() const {
        return string::format(
            "UnstructuredBlock<NDIM=%d>(nnode=%d ngstnode=%d, nface=%d ngstface=%d, ncell=%d ngstcell=%d)"
          , NDIM
          , nnode(), ngstnode()
          , nface(), ngstface()
          , ncell(), ngstcell()
        );
    }

    std::string cell_info_string(index_type icl) const {
        return string::format(
            "Cell<NDIM=%d>(%d type=%d ncell=%d ngstcell=%d)"
          , NDIM, icl, cltpn()[icl]
          , ncell(), ngstcell()
        );
    }

//* end data_report */

/* utility */
private:

    void build_faces_from_cells();

    index_type calc_max_nface() const;

    std::tuple<index_type, index_type, index_type> count_ghost() const {
        std::vector<index_type> bcls(nbound());
        index_type ngstface = 0;
        index_type ngstnode = 0;
        for (index_type ibfc=0; ibfc<nbound(); ++ibfc) {
            const index_type ifc = bndfcs()[ibfc][0];
            const index_type icl = fccls()[ifc][0];
            ngstface += celltype(cltpn()[icl]).nface() - 1;
            ngstnode += clnds()[icl][0] - fcnds()[ifc][0];
        }
        return std::make_tuple(ngstnode, ngstface, nbound());
    }

    void fill_ghost();

    void build_rcells(LookupTable<index_type, CLMFC> & rcells, LookupTable<index_type, 0> & rcellno) const;

    void build_csr(const LookupTable<index_type, CLMFC> & rcells, LookupTable<index_type, 0> & adjncy) const;

/* end utility */

}; /* end class UnstructuredBlock */

template< size_t NDIM >
void UnstructuredBlock<NDIM>::build_boundary() {
    assert(0 == m_nbound); // nothing should touch m_nbound beforehand.
    m_nbound = std::count_if(
        &fccls()[0], &fccls()[fccls().nbody()],
        [](const typename decltype(m_fccls)::row_type & row) { return row[1] < 0; });
    m_bndfcs = LookupTable<index_type, 2>(0, m_nbound);

    std::vector<index_type> allfacn(m_nbound);
    index_type ait = 0;
    for (index_type ifc=0; ifc<nface(); ++ifc) {
        if (fcjcl(ifc) < 0) { allfacn.at(ait) = ifc; ++ait; }
    }

    std::vector<bool> specified(m_nbound, false);
    index_type ibfc = 0;
    index_type nleft = m_nbound;
    for (index_type ibnd=0; ibnd<m_bndvec.size(); ++ibnd) {
        BoundaryData & bnd = m_bndvec[ibnd];
        auto & bfacn = bnd.facn();
        for (index_type bfit=0; bfit<bfacn.nbody(); ++bfit) {
            m_bndfcs.set_at(ibfc, bfacn[bfit][0], ibnd);
            bfacn[bfit][1] = ibfc;
            auto found = std::find(allfacn.begin(), allfacn.end(), bfacn[bfit][0]);
            if (allfacn.end() != found) {
                specified.at(found - allfacn.begin()) = true;
                --nleft;
            }
            ++ibfc;
        }
    }
    assert(nleft >= 0);

    if (nleft != 0) {
        BoundaryData bnd(0);
        bnd.facn() = LookupTable<index_type, BoundaryData::BFREL>(0, nleft);
        bnd.values() = LookupTableCore(0, nleft, {nleft, 0}, type_to<real_type>::id);
        auto & bfacn = bnd.facn();
        index_type bfit = 0;
        index_type ibnd = m_bndvec.size();
        for (index_type sit=0; sit<m_nbound; ++sit) { // Specified ITerator.
            if (!specified[sit]) {
                m_bndfcs.set_at(ibfc, allfacn[sit], ibnd);
                bfacn[bfit][0] = allfacn[sit];
                bfacn[bfit][1] = ibfc;
                ++ibfc;
                ++bfit;
            }
        }
        m_bndvec.push_back(std::move(bnd));
        assert(m_bndvec.size() == ibnd+1);
    }
    assert(ibfc == m_nbound);
}

template< size_t NDIM >
std::tuple<march::depend::scotch::num_type, LookupTable<index_type, 0>>
UnstructuredBlock<NDIM>::partition(index_type npart) const {
    using num_type = march::depend::scotch::num_type;

    LookupTable<index_type, CLMFC> rcells(0, ncell());
    LookupTable<index_type, 0> rcellno(0, ncell());
    build_rcells(rcells, rcellno);

    LookupTable<index_type, 0> xadj(0, ncell()+1);
    xadj[0] = 0;
    for (index_type it=1; it<xadj.nbody(); ++it) {
        xadj[it] = rcellno[it-1] + xadj[it-1];
    }

    index_type nitem = std::accumulate(rcellno.data(), rcellno.data()+rcellno.nelem(), (index_type)0);
    LookupTable<index_type, 0> adjncy(0, nitem);
    build_csr(rcells, adjncy);

    static_assert(sizeof(index_type) == sizeof(num_type), "index_type differs from num_type");
    num_type nedge = ncell();
    num_type vwgt = 0;
    num_type adjwgt = 0;
    num_type wgtflag = 0;
    num_type numflag = 0;
    num_type options[5] = {0, 0, 0, 0, 0};
    num_type edgecut = 0;
    LookupTable<index_type, 0> part(0, ncell());

    METIS_PartGraphKway(
        &nedge,
        xadj.data(),
        adjncy.data(),
        &vwgt,
        &adjwgt,
        &wgtflag,
        &numflag,
        &npart,
        options,
        // output.
        &edgecut,
        part.data()
    );

    return std::make_tuple(edgecut, part);
}

template< size_t NDIM >
class CellHand {

public:

    using block_type = UnstructuredBlock<NDIM>;
    using vector_type = Vector<NDIM>;

    CellHand(block_type & block, index_type index) : m_block(&block), m_index(index) {}

    block_type       & block()       { return *m_block; }
    block_type const & block() const { return *m_block; }

    index_type index() const { return m_index; }
    void set_index(index_type index) { m_index = index; }

    std::string repr(size_t indent=0, size_t precision=0) const;

    CellType const & type() const { return celltype(block().cltpn()[index()]); }

    vector_type const & cnd() const {
        return *reinterpret_cast<vector_type const *>(&(block().clcnd()[index()][0]));
    }

    real_type vol() const { return block().clvol()[index()]; }

    index_type nnd() const { return block().clnds()[index()][0]; }

    struct boundcheck {};

    /**
     * Get the @a ind -th node index.  @a ind is 1-based.
     */
    index_type nds(index_type ind) const { return block().clnds()[index()][ind]; }
    /**
     * Get the @a ind -th node index.  @a ind is 1-based.  Bound checked.
     */
    index_type nds(index_type ind, boundcheck const &) const {
        auto const & clnds = block().clnds();
        if (ind >= clnds.ncolumn()) {
            throw std::out_of_range(string::format(
                "in cell %d, %d-th (1-based) node out of range (%d)",
                index(), ind, clnds.ncolumn()));
        }
        return clnds.at(index())[ind];
    }

    index_type nfc() const { return block().clfcs()[index()][0]; }

    /**
     * Get the @a ifc -th face index.  @a ifc is 1-based.
     */
    index_type fcs(index_type ifc) const { return block().clfcs()[index()][ifc]; }
    /**
     * Get the @a ifc -th face index.  @a ifc is 1-based.  Bound checked.
     */
    index_type fcs(index_type ifc, boundcheck const &) const {
        auto const & clfcs = block().clfcs();
        if (ifc >= clfcs.ncolumn()) {
            throw std::out_of_range(string::format(
                "in cell %d, %d-th (1-based) face out of range (%d)",
                index(), ifc, clfcs.ncolumn()));
        }
        return clfcs.at(index())[ifc];
    }

private:

    block_type * m_block = nullptr;
    index_type m_index = MH_INDEX_SENTINEL;

}; /* end class CellHand */

template< size_t NDIM >
std::string CellHand<NDIM>::repr(size_t indent, size_t precision) const {
    std::string ret(string::format("CellHand%ldD(", NDIM));
    const std::string indented_newline = string::create_indented_newline(indent);
    if (indent) { ret += indented_newline; }
    ret += "cnd=" + cnd().repr(indent, precision) + ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "vol=" + string::from_double(vol(), precision) + ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "nds=[";
    for (index_type ind=1; ind<=nnd(); ++ind) {
        ret += string::format("%d", nds(ind));
        ret += nnd() == ind ? "]" : ",";
    }
    ret += ",";
    ret += indent ? indented_newline : std::string(" ");
    ret += "fcs=[";
    for (index_type ifc=1; ifc<=nfc(); ++ifc) {
        ret += string::format("%d", fcs(ifc));
        ret += nfc() == ifc ? "]" : ",";
    }
    if (indent) { ret += "\n)"; }
    else        { ret += ")"; }
    return ret;
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
