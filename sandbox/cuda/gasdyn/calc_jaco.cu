/*
 * Copyright (C) 2010-2011 Yung-Yu Chen <yyc@solvcon.net>.
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

#include "gasdyn.h"

#ifdef __CUDACC__
#define SOLVCON_CUSE_JACO
#include "calc_soltn.cu"
#include "calc_dsoln.cu"
#endif

#ifdef __CUDACC__
__device__ void cuda_calc_jaco(exedata *exd, int icl,
        double (*fcn)[NDIM], double (*ajacos)[NDIM]) {
    double (*jacos)[NEQ][NDIM];
    jacos = (double (*)[NEQ][NDIM])ajacos;
#else
void calc_jaco(exedata *exd, int icl,
        double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM]) {
#endif
    // pointers.
    double *psol;
    // scalars.
    double ga, ga1, ga3, ga1h;
    double u1, u2, u3, u4;
#if NDIM == 3
    double u5;
#endif
    // accelerating variables.
    double rho2, ke2, g1ke2, vs, gretot, getot, pr, v1o2, v2o2, v1, v2;
#if NDIM == 3
    double v3o2, v3;
#endif

    // initialize values.
    ga = exd->amsca[icl*NSCA];
    ga1 = ga-1;
    ga3 = ga-3;
    ga1h = ga1/2;
    psol = exd->sol + icl*NEQ;
    u1 = psol[0] + SOLVCON_TINY;
    u2 = psol[1];
    u3 = psol[2];
    u4 = psol[3];
#if NDIM == 3
    u5 = psol[4];
#endif

    // accelerating variables.
    rho2 = u1*u1;
    v1 = u2/u1; v1o2 = v1*v1;
    v2 = u3/u1; v2o2 = v2*v2;
#if NDIM == 3
    v3 = u4/u1; v3o2 = v3*v3;
#endif
    ke2 = (u2*u2 + u3*u3
#if NDIM == 3
        + u4*u4
#endif
    )/u1;
    g1ke2 = ga1*ke2;
    vs = ke2/u1;
    gretot = ga * 
#if NDIM == 3
        u5
#else
        u4
#endif
    ;
    getot = gretot/u1;
    pr = ga1*
#if NDIM == 3
        u5
#else
        u4
#endif
        - ga1h * ke2;

    // flux function.
#if NDIM == 3
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
#else
    fcn[0][0] = u2; fcn[0][1] = u3;
    fcn[1][0] = pr + u2*v1;
    fcn[1][1] = u2*v2;
    fcn[2][0] = u3*v1;
    fcn[2][1] = pr + u3*v2;
    fcn[3][0] = (pr + u4)*v1;
    fcn[3][1] = (pr + u4)*v2;
#endif
 
    // Jacobian matrices.
#if NDIM == 3
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
#else
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
#endif

    return;
};

// vim: set ts=4 et:
