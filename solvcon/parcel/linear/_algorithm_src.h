#ifndef __SC_LINEAR__ALGORITHM_SRC_H__
#define __SC_LINEAR__ALGORITHM_SRC_H__
/*
 * Copyright (C) 2013 Po-Hsien Lin <lin.880@buckeyemail.osu.edu>.
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
 * - Neither the name of the SOLVCON nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
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

#include "mesh.h"
#include "_algorithm.h"

#define NEQ alg->neq

#undef NDIM
#define NDIM 2
void sc_linear_calc_jaco_2d(sc_mesh_t *msd, sc_linear_algorithm_t *alg,
    int icl, double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM]);
void sc_linear_calc_dif_2d(sc_mesh_t *msd, sc_linear_algorithm_t *alg,
    int icl, double difs[NEQ][NDIM]);
#undef NDIM
#define NDIM 3
void sc_linear_calc_jaco_3d(sc_mesh_t *msd, sc_linear_algorithm_t *alg,
    int icl, double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM]);
void sc_linear_calc_dif_3d(sc_mesh_t *msd, sc_linear_algorithm_t *alg,
    int icl, double difs[NEQ][NDIM]);

// vim: set ft=c ts=4 et:
#endif // __SC_LINEAR__ALGORITHM_SRC_H__
