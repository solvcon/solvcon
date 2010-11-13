/*
 * Copyright (C) 2008-2010 Yung-Yu Chen <yyc@solvcon.net>.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "cese.h"

// Two-/three-dimensional GGE definition (in c-tau scheme).
const int ggefcs[31][3] = {
    // quadrilaterals.
    {1, 2, -1}, {2, 3, -1}, {3, 4, -1}, {4, 1, -1},
    // triangles.
    {1, 2, -1}, {2, 3, -1}, {3, 1, -1},
    // hexahedra.
    {2, 3, 5}, {6, 3, 2}, {4, 3, 6}, {5, 3, 4},
    {5, 1, 2}, {2, 1, 6}, {6, 1, 4}, {4, 1, 5},
    // tetrahedra.
    {3, 1, 2}, {2, 1, 4}, {4, 1, 3}, {2, 4, 3},
    // prisms.
    {5, 2, 4}, {3, 2, 5}, {4, 2, 3},
    {4, 1, 5}, {5, 1, 3}, {3, 1, 4},
    // pyramids
    {1, 5, 2}, {2, 5, 3}, {3, 5, 4}, {4, 5, 1},
    {1, 3, 4}, {3, 1, 2},
};
const int ggerng[8][2] = {
    {-1, -1}, {-2, -1}, {0, 4}, {4, 7},
    {7, 15}, {15, 19}, {19, 25}, {25, 31},
    //{0, 8}, {8, 12}, {12, 18}, {18, 24},
};

// Two-/three-dimensional face node mappings (in c-omega scheme).
const int sfcs[4][3] = {
    // edges.
    {1, 2, -1},
    // quadrilaterals.
    {1, 2, 3}, {1, 4, 3},
    // triangles.
    {1, 2, 3},
};
const int sfng[4][2] = {
    {-1, -1}, {0, 1}, {1, 3}, {3, 4},
};

const int hvfs[8][2] = {    // vertex to face.
    // triangles.
    {1, 2}, {2, 3}, {3, 1},
    // tetrahedra.
    {1, 4}, {2, 3}, {3, 2}, {4, 1},
    // pyramids.
    {5, 5},
};
const int hrng[8][2] = {
    {-1, -1}, {-2, -1}, {-2, -1}, {0, 3},
    {-2, -1}, {3, 7}, {-2, -1}, {7, 8},
};

const int evts[42][2] = {
    // quadrilaterals.
    {1, 2}, {2, 3}, {3, 4}, {4, 1},
    // triangles.
    {1, 2}, {2, 3}, {3, 1},
    // hexahedra.
    {1, 2}, {2, 3}, {3, 4}, {4, 1},
    {5, 6}, {6, 7}, {7, 8}, {8, 5},
    {1, 5}, {2, 6}, {3, 7}, {4, 8},
    // tetrahedra.
    {1, 2}, {2, 3}, {3, 1}, {1, 4}, {2, 4}, {3, 4},
    // prisms.
    {1, 2}, {2, 3}, {3, 1}, {4, 5}, {5, 6}, {6, 4}, {1, 4}, {2, 5}, {3, 6},
    // pyramids.
    {1, 2}, {2, 3}, {3, 4}, {4, 1}, {1, 5}, {2, 5}, {3, 5}, {4, 5},
};
const int egng[8][2] = {
    {-1, -1}, {-2, -1}, {0, 4}, {4, 7},
    {7, 19}, {19, 25}, {25, 34}, {34, 42},
};
// vim: set ts=4 et:
