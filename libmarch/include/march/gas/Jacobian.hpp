#pragma once

/*
 * Copyright (c) 2014, Yung-Yu Chen <yyc@solvcon.net>
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

#include "march/core/core.hpp"

namespace march {

namespace gas {

template< size_t NEQ, size_t NDIM >
struct Jacobian {
    using vector_type = Vector<NDIM>;
    using matrix_type = vector_type[NEQ][NEQ];
    using flux_type = vector_type[NEQ];
    using const_solution_reference = const real_type (&)[NEQ];
    static constexpr real_type TINY=1.e-60;
    void update(real_type gamma, const_solution_reference sol);
    matrix_type jacos;
    flux_type fcn;
}; /* end struct Jacobian */

template<>
inline
void Jacobian<4, 2>::update(real_type gamma, const_solution_reference sol) {
    // scalars.
    real_type ga, ga1, ga3, ga1h;
    real_type u1, u2, u3, u4;
    // accelerating variables.
    real_type rho2, ke2, g1ke2, vs, gretot, getot, pr, v1o2, v2o2, v1, v2;

    // initialize values.
    ga = gamma;
    ga1 = ga-1;
    ga3 = ga-3;
    ga1h = ga1/2;
    u1 = sol[0] + TINY;
    u2 = sol[1];
    u3 = sol[2];
    u4 = sol[3];

    // accelerating variables.
    rho2 = u1*u1;
    v1 = u2/u1; v1o2 = v1*v1;
    v2 = u3/u1; v2o2 = v2*v2;
    ke2 = (u2*u2 + u3*u3)/u1;
    g1ke2 = ga1*ke2;
    vs = ke2/u1;
    gretot = ga * u4;
    getot = gretot/u1;
    pr = ga1 * u4 - ga1h * ke2;

    // flux function.
    fcn[0][0] = u2; fcn[0][1] = u3;
    fcn[1][0] = pr + u2*v1;
    fcn[1][1] = u2*v2;
    fcn[2][0] = u3*v1;
    fcn[2][1] = pr + u3*v2;
    fcn[3][0] = (pr + u4)*v1;
    fcn[3][1] = (pr + u4)*v2;
 
    // Jacobian matrices.
    jacos[0][0][0] = 0; jacos[0][0][1] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1;
    jacos[0][3][0] = 0; jacos[0][3][1] = 0;

    jacos[1][0][0] = -v1o2 + ga1h*vs;
    jacos[1][0][1] = -v1*v2;
    jacos[1][1][0] = -ga3*v1; jacos[1][1][1] = v2;
    jacos[1][2][0] = -ga1*v2; jacos[1][2][1] = v1;
    jacos[1][3][0] = ga1;     jacos[1][3][1] = 0;

    jacos[2][0][0] = -v2*v1;
    jacos[2][0][1] = -v2o2 + ga1h*vs;
    jacos[2][1][0] = v2; jacos[2][1][1] = -ga1*v1;
    jacos[2][2][0] = v1; jacos[2][2][1] = -ga3*v2;
    jacos[2][3][0] = 0;  jacos[2][3][1] = ga1;

    jacos[3][0][0] = (-gretot + g1ke2)*u2/rho2;
    jacos[3][0][1] = (-gretot + g1ke2)*u3/rho2;
    jacos[3][1][0] = getot - ga1h*(vs + 2*v1o2);
    jacos[3][1][1] = -ga1*v1*v2;
    jacos[3][2][0] = -ga1*v2*v1;
    jacos[3][2][1] = getot - ga1h*(vs + 2*v2o2);
    jacos[3][3][0] = ga*v1; jacos[3][3][1] = ga*v2;
}

template<>
inline
void Jacobian<5, 3>::update(real_type gamma, const_solution_reference sol) {
    // scalars.
    real_type ga, ga1, ga3, ga1h;
    real_type u1, u2, u3, u4, u5;
    // accelerating variables.
    real_type rho2, ke2, g1ke2, vs, gretot, getot, pr, v1o2, v2o2, v1, v2;
    real_type v3o2, v3;

    // initialize values.
    ga = gamma;
    ga1 = ga-1;
    ga3 = ga-3;
    ga1h = ga1/2;
    u1 = sol[0] + TINY;
    u2 = sol[1];
    u3 = sol[2];
    u4 = sol[3];
    u5 = sol[4];

    // accelerating variables.
    rho2 = u1*u1;
    v1 = u2/u1; v1o2 = v1*v1;
    v2 = u3/u1; v2o2 = v2*v2;
    v3 = u4/u1; v3o2 = v3*v3;
    ke2 = (u2*u2 + u3*u3 + u4*u4)/u1;
    g1ke2 = ga1*ke2;
    vs = ke2/u1;
    gretot = ga * u5;
    getot = gretot/u1;
    pr = ga1 * u5 - ga1h * ke2;

    // flux function.
    fcn[0][0] = u2; fcn[0][1] = u3; fcn[0][2] = u4;
    fcn[1][0] = pr + u2*v1;
    fcn[1][1] = u2*v2;
    fcn[1][2] = u2*v3;
    fcn[2][0] = u3*v1;
    fcn[2][1] = pr + u3*v2;
    fcn[2][2] = u3*v3;
    fcn[3][0] = u4*v1;
    fcn[3][1] = u4*v2;
    fcn[3][2] = pr + u4*v3;
    fcn[4][0] = (pr + u5)*v1;
    fcn[4][1] = (pr + u5)*v2;
    fcn[4][2] = (pr + u5)*v3;
 
    // Jacobian matrices.
    jacos[0][0][0] = 0; jacos[0][0][1] = 0; jacos[0][0][2] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0; jacos[0][1][2] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1; jacos[0][2][2] = 0;
    jacos[0][3][0] = 0; jacos[0][3][1] = 0; jacos[0][3][2] = 1;
    jacos[0][4][0] = 0; jacos[0][4][1] = 0; jacos[0][4][2] = 0;

    jacos[1][0][0] = -v1o2 + ga1h*vs;
    jacos[1][0][1] = -v1*v2;
    jacos[1][0][2] = -v1*v3;
    jacos[1][1][0] = -ga3*v1; jacos[1][1][1] = v2; jacos[1][1][2] = v3;
    jacos[1][2][0] = -ga1*v2; jacos[1][2][1] = v1; jacos[1][2][2] = 0;
    jacos[1][3][0] = -ga1*v3; jacos[1][3][1] = 0;  jacos[1][3][2] = v1;
    jacos[1][4][0] = ga1;     jacos[1][4][1] = 0;  jacos[1][4][2] = 0;

    jacos[2][0][0] = -v2*v1;
    jacos[2][0][1] = -v2o2 + ga1h*vs;
    jacos[2][0][2] = -v2*v3;
    jacos[2][1][0] = v2; jacos[2][1][1] = -ga1*v1; jacos[2][1][2] = 0;
    jacos[2][2][0] = v1; jacos[2][2][1] = -ga3*v2; jacos[2][2][2] = v3;
    jacos[2][3][0] = 0;  jacos[2][3][1] = -ga1*v3; jacos[2][3][2] = v2;
    jacos[2][4][0] = 0;  jacos[2][4][1] = ga1;     jacos[2][4][2] = 0;

    jacos[3][0][0] = -v3*v1;
    jacos[3][0][1] = -v3*v2;
    jacos[3][0][2] = -v3o2 + ga1h*vs;
    jacos[3][1][0] = v3; jacos[3][1][1] = 0;  jacos[3][1][2] = -ga1*v1;
    jacos[3][2][0] = 0;  jacos[3][2][1] = v3; jacos[3][2][2] = -ga1*v2;
    jacos[3][3][0] = v1; jacos[3][3][1] = v2; jacos[3][3][2] = -ga3*v3;
    jacos[3][4][0] = 0;  jacos[3][4][1] = 0;  jacos[3][4][2] = ga1;

    jacos[4][0][0] = (-gretot + g1ke2)*u2/rho2;
    jacos[4][0][1] = (-gretot + g1ke2)*u3/rho2;
    jacos[4][0][2] = (-gretot + g1ke2)*u4/rho2;
    jacos[4][1][0] = getot - ga1h*(vs + 2*v1o2);
    jacos[4][1][1] = -ga1*v1*v2;
    jacos[4][1][2] = -ga1*v1*v3;
    jacos[4][2][0] = -ga1*v2*v1;
    jacos[4][2][1] = getot - ga1h*(vs + 2*v2o2);
    jacos[4][2][2] = -ga1*v2*v3;
    jacos[4][3][0] = -ga1*v3*v1;
    jacos[4][3][1] = -ga1*v3*v2;
    jacos[4][3][2] = getot - ga1h*(vs + 2*v3o2);
    jacos[4][4][0] = ga*v1; jacos[4][4][1] = ga*v2; jacos[4][4][2] = ga*v3;
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
