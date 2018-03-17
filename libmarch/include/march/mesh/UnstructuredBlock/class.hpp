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

#include "march/core/core.hpp"

#include "march/mesh/BoundaryData.hpp"

namespace march {

struct CellType {

    static constexpr size_t NVALUE = 5;

    CellType(
        index_type const type_in
      , index_type const dim_in
      , index_type const nnode_in
      , index_type const nedge_in
      , index_type const nsurface_in
    ) : m_data{type_in, dim_in, nnode_in, nedge_in, nsurface_in} {}

    CellType() = default;

    index_type const & type    () const { return m_data[0]; }
    index_type const & ndim    () const { return m_data[1]; }
    index_type const & nnode   () const { return m_data[2]; }
    index_type const & nedge   () const { return m_data[3]; }
    index_type const & nsurface() const { return m_data[4]; }

    index_type nface() const { return m_data[ndim()+1] ; }

    index_type const & operator[](size_t it) const { return m_data[it]; }
    constexpr size_t size() const { return NVALUE; }

    /* symbols for type codes */
    static constexpr index_type NONCELLTYPE   = -1; /* not a cell type */
    static constexpr index_type POINT         =  0;
    static constexpr index_type LINE          =  1;
    static constexpr index_type QUADRILATERAL =  2;
    static constexpr index_type TRIANGLE      =  3;
    static constexpr index_type HEXAHEDRON    =  4;
    static constexpr index_type TETRAHEDRON   =  5;
    static constexpr index_type PRISM         =  6;
    static constexpr index_type PYRAMID       =  7;
    /* Number of all types; one larger than the last type code.  Try not to use
     * this from outside, except the alias in CellTypeGroup */
    static constexpr size_t     NTYPE         =  8;

    const char * name() const {
        switch (type()) {
        case POINT         /*  0 */: return "point"         ; break;
        case LINE          /*  1 */: return "line"          ; break;
        case QUADRILATERAL /*  2 */: return "quadrilateral" ; break;
        case TRIANGLE      /*  3 */: return "triangle"      ; break;
        case HEXAHEDRON    /*  4 */: return "hexahedron"    ; break;
        case TETRAHEDRON   /*  5 */: return "tetrahedron"   ; break;
        case PRISM         /*  6 */: return "prism"         ; break;
        case PYRAMID       /*  7 */: return "pyramid"       ; break;
        case NONCELLTYPE   /* -1 */:
        default         /* other */: return "noncelltype"   ; break;
        }
    }

private:

    index_type m_data[NVALUE] = {-1, -1, -1, -1, -1}; /* sentinel */

}; /* end struct CellType */

#define MH_DECL_CELL_TYPE(NAME, TYPE, DIM, NNODE, NEDGE, NSURFACE) \
struct NAME##CellType : public CellType { \
    NAME##CellType() : CellType(TYPE, DIM, NNODE, NEDGE, NSURFACE) {} \
};
//                               type, ndim, nnode, nedge, nsurface
MH_DECL_CELL_TYPE(Point        ,    0,    0,     1,     0,        0 ) // point/node/vertex
MH_DECL_CELL_TYPE(Line         ,    1,    1,     2,     0,        0 ) // line/edge
MH_DECL_CELL_TYPE(Quadrilateral,    2,    2,     4,     4,        0 )
MH_DECL_CELL_TYPE(Triangle     ,    3,    2,     3,     3,        0 )
MH_DECL_CELL_TYPE(Hexahedron   ,    4,    3,     8,    12,        6 ) // hexahedron/brick
MH_DECL_CELL_TYPE(Tetrahedron  ,    5,    3,     4,     6,        4 )
MH_DECL_CELL_TYPE(Prism        ,    6,    3,     6,     9,        5 )
MH_DECL_CELL_TYPE(Pyramid      ,    7,    3,     5,     8,        5 )
#undef MH_DECL_CELL_TYPE

class CellTypeGroup {

public:

    static constexpr size_t NTYPE = CellType::NTYPE;

    CellTypeGroup(CellTypeGroup const & ) = delete;
    CellTypeGroup(CellTypeGroup       &&) = delete;
    CellTypeGroup const &  operator=(CellTypeGroup const & ) = delete;
    CellTypeGroup       && operator=(CellTypeGroup       &&) = delete;

    CellType const & point        () const { return m_cell_types[CellType::POINT        ]; }
    CellType const & line         () const { return m_cell_types[CellType::LINE         ]; }
    CellType const & quadrilateral() const { return m_cell_types[CellType::QUADRILATERAL]; }
    CellType const & triangle     () const { return m_cell_types[CellType::TRIANGLE     ]; }
    CellType const & hexahedron   () const { return m_cell_types[CellType::HEXAHEDRON   ]; }
    CellType const & tetrahedron  () const { return m_cell_types[CellType::TETRAHEDRON  ]; }
    CellType const & prism        () const { return m_cell_types[CellType::PRISM        ]; }
    CellType const & pyramid      () const { return m_cell_types[CellType::PYRAMID      ]; }

    CellType const & operator[](size_t it) const { return m_cell_types[it]; }
    size_t size() const { return sizeof(m_cell_types) / sizeof(CellType); }

    static const CellTypeGroup & get_instance() {
        static CellTypeGroup inst;
        return inst;
    }

private:

    CellTypeGroup()
      : m_cell_types{
            PointCellType()
          , LineCellType()
          , QuadrilateralCellType()
          , TriangleCellType()
          , HexahedronCellType()
          , TetrahedronCellType()
          , PrismCellType()
          , PyramidCellType()
        }
    {}

    CellType m_cell_types[NTYPE];

}; /* end class CellTypeGroup */

inline CellType const & celltype(size_t it) { return CellTypeGroup::get_instance()[it]; }

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

    /// Maximum number of nodes in a face.
    static constexpr index_type MAX_FCNND = 4;
    static constexpr index_type     FCMND = MAX_FCNND; // alias
    /// Maximum number of nodes in a cell.
    static constexpr index_type MAX_CLNND = 8;
    static constexpr index_type     CLMND = MAX_CLNND; // alias
    /// Maximum number of faces in a cell.
    static constexpr index_type MAX_CLNFC = 6;
    static constexpr index_type     CLMFC = MAX_CLNFC; // alias
    static constexpr index_type     FCNCL = 4;
    static constexpr index_type     FCREL = 4;
    static constexpr index_type     BFREL = BoundaryData::BFREL;

    // TODO: move to UnstructuredBlock.
    // @[
    void locate_point(const real_type (& crd)[NDIM]) const;

    // moved to mesh: void prepare_ce();
    // moved to mesh: void prepare_sf();
    // @]

    /**
     * The dual mesh of the conservation element.
     */
    struct CEMesh {
    public:
        LookupTable<real_type, (CLMFC+1)*NDIM> cecnd;
        LookupTable<real_type, CLMFC+1> cevol;
        LookupTable<real_type, CLMFC*FCMND*2*NDIM> sfmrc;
        CEMesh() = delete;
        CEMesh(CEMesh const & ) = delete;
        CEMesh(CEMesh       &&) = delete;
        CEMesh operator=(CEMesh const & ) = delete;
        CEMesh operator=(CEMesh       &&) = delete;
        CEMesh(const UnstructuredBlock<NDIM> & block)
          : cecnd(block.ngstcell(), block.ncell()), cevol(block.ngstcell(), block.ncell()), sfmrc(0, block.ncell())
        {
            calc_ce(block);
            calc_sf(block);
        }
    private:
        void calc_ce(const UnstructuredBlock<NDIM> & block);
        void calc_sf(const UnstructuredBlock<NDIM> & block);
    }; /* end struct CEMesh */

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
        return string_format(
            "UnstructuredBlock<NDIM=%d>(nnode=%d ngstnode=%d, nface=%d ngstface=%d, ncell=%d ngstcell=%d)"
          , NDIM
          , nnode(), ngstnode()
          , nface(), ngstface()
          , ncell(), ngstcell()
        );
    }

    std::string cell_info_string(index_type icl) const {
        return string_format(
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

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
