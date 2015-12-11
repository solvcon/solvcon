/*
 * Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <Python.h>

#include "mesh.h"
#include "_algorithm.h"
#include "_algorithm_src.h"

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
#include "sc_linear_calc_dsoln.c_body"
#undef NDIM
#define NDIM 3
#include "sc_linear_calc_dsoln.c_body"

// vim: set ts=4 et:
