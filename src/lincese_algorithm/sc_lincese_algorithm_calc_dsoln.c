/*
 * Copyright (C) 2008-2012 Yung-Yu Chen <yyc@solvcon.net>.
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

#include <Python.h>

#include "mesh.h"
#include "lincese_algorithm.h"

#define NEQ alg->neq
#define MFGE 8
#define ALMOST_ZERO 1.e-200

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

#undef NDIM
#define NDIM 2
#include "sc_lincese_algorithm_calc_dsoln.c_body"
#undef NDIM
#define NDIM 3
#include "sc_lincese_algorithm_calc_dsoln.c_body"

// vim: set ts=4 et:
