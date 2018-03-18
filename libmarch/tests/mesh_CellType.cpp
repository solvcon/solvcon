/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include <gtest/gtest.h>

#include "march/mesh/CellType.hpp"

using namespace march;

/*
 * begin CellTypeTest
 */

TEST(CellTypeTest, name) {
    EXPECT_EQ(CellType             ().name(), std::string("noncelltype"  ));
    EXPECT_EQ(PointCellType        ().name(), std::string("point"        ));
    EXPECT_EQ(LineCellType         ().name(), std::string("line"         ));
    EXPECT_EQ(QuadrilateralCellType().name(), std::string("quadrilateral"));
    EXPECT_EQ(TriangleCellType     ().name(), std::string("triangle"     ));
    EXPECT_EQ(HexahedronCellType   ().name(), std::string("hexahedron"   ));
    EXPECT_EQ(TetrahedronCellType  ().name(), std::string("tetrahedron"  ));
    EXPECT_EQ(PrismCellType        ().name(), std::string("prism"        ));
    EXPECT_EQ(PyramidCellType      ().name(), std::string("pyramid"      ));
}

TEST(CellTypeTest, type_code_symbols) {
    EXPECT_EQ(index_type(CellType::NONCELLTYPE  ), -1);
    EXPECT_EQ(index_type(CellType::POINT        ),  0);
    EXPECT_EQ(index_type(CellType::LINE         ),  1);
    EXPECT_EQ(index_type(CellType::QUADRILATERAL),  2);
    EXPECT_EQ(index_type(CellType::TRIANGLE     ),  3);
    EXPECT_EQ(index_type(CellType::HEXAHEDRON   ),  4);
    EXPECT_EQ(index_type(CellType::TETRAHEDRON  ),  5);
    EXPECT_EQ(index_type(CellType::PRISM        ),  6);
    EXPECT_EQ(index_type(CellType::PYRAMID      ),  7);
}

/*
 * end CellTypeTest
 */

/*
 * begin CellTypeGroupTest
 */

TEST(CellTypeGroupTest, get_instance) {
    using CellTypeGroup = detail::CellTypeGroup;

    auto const & ctg = CellTypeGroup::get_instance();

    EXPECT_EQ(ctg.size(), index_type(CellType::NTYPE));

    // point
    EXPECT_EQ(ctg.point().id()      , 0);
    EXPECT_EQ(ctg.point().ndim()    , 0);
    EXPECT_EQ(ctg.point().nnode()   , 1);
    EXPECT_EQ(ctg.point().nedge()   , 0);
    EXPECT_EQ(ctg.point().nsurface(), 0);
    EXPECT_EQ(ctg.point().nface()   , 0);
    // line
    EXPECT_EQ(ctg.line().id()      , 1);
    EXPECT_EQ(ctg.line().ndim()    , 1);
    EXPECT_EQ(ctg.line().nnode()   , 2);
    EXPECT_EQ(ctg.line().nedge()   , 0);
    EXPECT_EQ(ctg.line().nsurface(), 0);
    EXPECT_EQ(ctg.line().nface()   , 2);
    // quadrilateral
    EXPECT_EQ(ctg.quadrilateral().id()      , 2);
    EXPECT_EQ(ctg.quadrilateral().ndim()    , 2);
    EXPECT_EQ(ctg.quadrilateral().nnode()   , 4);
    EXPECT_EQ(ctg.quadrilateral().nedge()   , 4);
    EXPECT_EQ(ctg.quadrilateral().nsurface(), 0);
    EXPECT_EQ(ctg.quadrilateral().nface()   , 4);
    // triangle
    EXPECT_EQ(ctg.triangle().id()      , 3);
    EXPECT_EQ(ctg.triangle().ndim()    , 2);
    EXPECT_EQ(ctg.triangle().nnode()   , 3);
    EXPECT_EQ(ctg.triangle().nedge()   , 3);
    EXPECT_EQ(ctg.triangle().nsurface(), 0);
    EXPECT_EQ(ctg.triangle().nface()   , 3);
    // hexahedron
    EXPECT_EQ(ctg.hexahedron().id()      , 4 );
    EXPECT_EQ(ctg.hexahedron().ndim()    , 3 );
    EXPECT_EQ(ctg.hexahedron().nnode()   , 8 );
    EXPECT_EQ(ctg.hexahedron().nedge()   , 12);
    EXPECT_EQ(ctg.hexahedron().nsurface(), 6 );
    EXPECT_EQ(ctg.hexahedron().nface()   , 6 );
    // tetrahedron
    EXPECT_EQ(ctg.tetrahedron().id()      , 5);
    EXPECT_EQ(ctg.tetrahedron().ndim()    , 3);
    EXPECT_EQ(ctg.tetrahedron().nnode()   , 4);
    EXPECT_EQ(ctg.tetrahedron().nedge()   , 6);
    EXPECT_EQ(ctg.tetrahedron().nsurface(), 4);
    EXPECT_EQ(ctg.tetrahedron().nface()   , 4);
    // prism
    EXPECT_EQ(ctg.prism().id()      , 6);
    EXPECT_EQ(ctg.prism().ndim()    , 3);
    EXPECT_EQ(ctg.prism().nnode()   , 6);
    EXPECT_EQ(ctg.prism().nedge()   , 9);
    EXPECT_EQ(ctg.prism().nsurface(), 5);
    EXPECT_EQ(ctg.prism().nface()   , 5);
    // pyramid
    EXPECT_EQ(ctg.pyramid().id()      , 7);
    EXPECT_EQ(ctg.pyramid().ndim()    , 3);
    EXPECT_EQ(ctg.pyramid().nnode()   , 5);
    EXPECT_EQ(ctg.pyramid().nedge()   , 8);
    EXPECT_EQ(ctg.pyramid().nsurface(), 5);
    EXPECT_EQ(ctg.pyramid().nface()   , 5);

}

/*
 * end CellTypeGroupTest
 */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
