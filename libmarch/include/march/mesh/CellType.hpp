#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include "march/core/core.hpp"

namespace march {

/**
 * Cell type for unstructured mesh.
 */
struct CellType {

    static constexpr size_t NVALUE = 5;

    CellType(
        index_type const id_in
      , index_type const dim_in
      , index_type const nnode_in
      , index_type const nedge_in
      , index_type const nsurface_in
    ) : m_data{id_in, dim_in, nnode_in, nedge_in, nsurface_in} {}

    CellType() = default;

    index_type id      () const { return m_data[0]; }
    index_type ndim    () const { return m_data[1]; }
    index_type nnode   () const { return m_data[2]; }
    index_type nedge   () const { return m_data[3]; }
    index_type nsurface() const { return m_data[4]; }

    index_type nface() const { return m_data[ndim()+1] ; }

    index_type operator[](size_t it) const { return m_data[it]; }
    constexpr size_t size() const { return NVALUE; }

    /* symbols for type id codes */
    static constexpr index_type NONCELLTYPE   = -1; /* not a cell type */
    static constexpr index_type POINT         =  0;
    static constexpr index_type LINE          =  1;
    static constexpr index_type QUADRILATERAL =  2;
    static constexpr index_type TRIANGLE      =  3;
    static constexpr index_type HEXAHEDRON    =  4;
    static constexpr index_type TETRAHEDRON   =  5;
    static constexpr index_type PRISM         =  6;
    static constexpr index_type PYRAMID       =  7;
    /* number of all types; one larger than the last type id code */
    static constexpr size_t     NTYPE         =  8;

    const char * name() const {
        switch (id()) {
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

    //< Maximum number of nodes in a face.
    static constexpr index_type FCNND_MAX = 4;
    //< Maximum number of nodes in a cell.
    static constexpr index_type CLNND_MAX = 8;
    //< Maximum number of faces in a cell.
    static constexpr index_type CLNFC_MAX = 6;

private:

    index_type m_data[NVALUE] = {-1, -1, -1, -1, -1}; /* sentinel */

}; /* end struct CellType */

#define MH_DECL_CELL_TYPE(NAME, TYPE, DIM, NNODE, NEDGE, NSURFACE) \
struct NAME##CellType : public CellType { \
    NAME##CellType() : CellType(TYPE, DIM, NNODE, NEDGE, NSURFACE) {} \
};
//                                 id, ndim, nnode, nedge, nsurface
MH_DECL_CELL_TYPE(Point        ,    0,    0,     1,     0,        0 ) // point/node/vertex
MH_DECL_CELL_TYPE(Line         ,    1,    1,     2,     0,        0 ) // line/edge
MH_DECL_CELL_TYPE(Quadrilateral,    2,    2,     4,     4,        0 )
MH_DECL_CELL_TYPE(Triangle     ,    3,    2,     3,     3,        0 )
MH_DECL_CELL_TYPE(Hexahedron   ,    4,    3,     8,    12,        6 ) // hexahedron/brick
MH_DECL_CELL_TYPE(Tetrahedron  ,    5,    3,     4,     6,        4 )
MH_DECL_CELL_TYPE(Prism        ,    6,    3,     6,     9,        5 )
MH_DECL_CELL_TYPE(Pyramid      ,    7,    3,     5,     8,        5 )
#undef MH_DECL_CELL_TYPE

namespace detail {

class CellTypeGroup {

public:

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

    CellType const & operator[](size_t id) const { return m_cell_types[id]; }
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

    CellType m_cell_types[CellType::NTYPE];

}; /* end class CellTypeGroup */

} /* end namespace detail */

/**
 * Get the CellType object for the specified type id.
 */
inline CellType const & celltype(size_t id) { return detail::CellTypeGroup::get_instance()[id]; }

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
