/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see LICENSE.txt
 */

#include "march/mesh/UnstructuredBlock.hpp"

/**
 * This is a block composes of only triangles.  "N" denotes node, and "C"
 * denotes cell.
 *
 *                                (0,1)N3
 *                              *
 *                             /|\
 *                            / | \
 *                           /  |  \
 *                          /   |   \
 *                         /    |    \
 *                        /     |     \
 *                       /  C2  *  C1  \
 *                      /      /^\      \
 *                     /     / N0  \     \
 *                    /    /  (0,0)  \    \
 *                   /   /             \   \
 *                  /  /                 \  \
 *                 / /         C0          \ \
 *                //                         \\
 *               *-----------------------------*
 *     (-1,-1)N1                                 (1,-1)N2
 */
inline std::shared_ptr<march::UnstructuredBlock<2>> make_triangles() {
    using namespace march;
    auto blk = UnstructuredBlock<2>::construct(/* nnode */4, /* nface */6, /* ncell */3, /* use_incenter */false);
    blk->ndcrd().set_at(0,  0,  0);
    blk->ndcrd().set_at(1, -1, -1);
    blk->ndcrd().set_at(2,  1, -1);
    blk->ndcrd().set_at(3,  0,  1);
    blk->cltpn().fill(3);
    blk->clnds().set_at(0, 3, 0, 1, 2);
    blk->clnds().set_at(1, 3, 0, 2, 3);
    blk->clnds().set_at(2, 3, 0, 3, 1);
    return blk;
}

/**
 * This is a block composes of only tetrahedra.  3D ascii art is beyond my
 * skill.  Use imagination.
 */
inline std::shared_ptr<march::UnstructuredBlock<3>> make_tetrahedra() {
    using namespace march;
    auto blk = UnstructuredBlock<3>::construct(/* nnode */5, /* nface */0, /* ncell */4, /* use_incenter */false);
    blk->ndcrd().set_at(0,   0,  0,  0);
    blk->ndcrd().set_at(1,  10,  0,  0);
    blk->ndcrd().set_at(2,   0, 10,  0);
    blk->ndcrd().set_at(3,   0,  0, 10);
    blk->ndcrd().set_at(4,   1,  1,  1);
    blk->cltpn().fill(5);
    blk->clnds().set_at(0, 4, 0, 1, 2, 4);
    blk->clnds().set_at(1, 4, 0, 2, 3, 4);
    blk->clnds().set_at(2, 4, 0, 3, 1, 4);
    blk->clnds().set_at(3, 4, 1, 2, 3, 4);
    return blk;
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
