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

#include "bulk.h"

#ifdef __CUDACC__
__device__ void cuda_calc_jaco(exedata *exd, int icl,
        double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM]) {
#else
void calc_jaco(exedata *exd, int icl,
        double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM]) {
#endif
    // pointers.
    double *psol;
    double *pamsca;
    // scalars.
    double bulk, p0, rho0, eta;
    double u1, u2, u3;
#if NDIM == 3
    double u4;
#endif
    // accelerating variables.
    double kdu1, u2u3, u2u4, u3u4, u2du1, u3du1, u4du1, bulkm;

    pamsca = exd->amsca + icl*NSCA;
    // initialize values.
    bulk = pamsca[0];
    p0 = pamsca[1];
    rho0 = pamsca[2];
    eta = pamsca[3];
    psol = exd->sol + icl*NEQ;
    u1 = psol[0] + SOLVCON_TINY;
    u2 = psol[1];
    u3 = psol[2];
#if NDIM == 3
    u4 = psol[3];
#endif

    // accelerating variables.
    // density base
    /*
    kdu1 = bulk/u1;
    bulkm = p0 + bulk*log(u1/rho0);
    u2u3 = u2*u3;
    u2du1 = u2/u1; u3du1 = u3/u1;
#if NDIM == 3
    u2u4 = u2*u4; u3u4 = u3*u4;
    u4du1 = u4/u1;
#endif
    // flux function.
#if NDIM == 3
    fcn[0][0] = u2; fcn[0][1] = u3; fcn[0][2] = u4;
    fcn[1][0] = u2*u2du1 + bulkm;
    fcn[1][1] = u2*u3du1;
    fcn[1][2] = u2*u4du1;
    fcn[2][0] = u3*u2du1;
    fcn[2][1] = u3*u3du1 + bulkm;
    fcn[2][2] = u3*u4du1;
    fcn[3][0] = u4*u2du1;
    fcn[3][1] = u4*u3du1;
    fcn[3][2] = u4*u4du1 + bulkm;
#else
    fcn[0][0] = u2;               fcn[0][1] = u3;
    fcn[1][0] = u2*u2du1 + bulkm; fcn[1][1] = u2*u3du1;
    fcn[2][0] = u3*u2du1;         fcn[2][1] = u3*u3du1 + bulkm;
#endif
    // Jacobian matrices.
#if NDIM == 3
    jacos[0][0][0] = 0; jacos[0][0][1] = 0; jacos[0][0][2] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0; jacos[0][1][2] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1; jacos[0][2][2] = 0;
    jacos[0][3][0] = 0; jacos[0][3][1] = 0; jacos[0][3][2] = 1;

    jacos[1][0][0] = kdu1-u2du1*u2du1; jacos[1][0][1] = -u2du1*u3du1; jacos[1][0][2] = -u2du1*u4du1;
    jacos[1][1][0] = 2*u2du1;          jacos[1][1][1] = u3du1;        jacos[1][1][2] = u4du1;
    jacos[1][2][0] = 0;                jacos[1][2][1] = u2du1;        jacos[1][2][2] = 0;
    jacos[1][3][0] = 0;                jacos[1][3][1] = 0;            jacos[1][3][2] = u2du1;

    jacos[2][0][0] = -u2du1*u3du1; jacos[2][0][1] = kdu1-u3du1*u3du1; jacos[2][0][2] = -u3du1*u4du1;
    jacos[2][1][0] = u3du1;        jacos[2][1][1] = 0;                jacos[2][1][2] = 0;
    jacos[2][2][0] = u2du1;        jacos[2][2][1] = 2*u3du1;          jacos[2][2][2] = u4du1;
    jacos[2][3][0] = 0;            jacos[2][3][1] = 0;                jacos[2][3][2] = u3du1;

    jacos[3][0][0] = -u2du1*u4du1; jacos[3][0][1] = -u3du1*u4du1;     jacos[3][0][2] = kdu1-u4du1*u4du1;
    jacos[3][1][0] = u4du1;        jacos[3][1][1] = 0;                jacos[3][1][2] = 0;
    jacos[3][2][0] = 0;            jacos[3][2][1] = u4du1;            jacos[3][2][2] = 0;
    jacos[3][3][0] = u2du1;        jacos[3][3][1] = u3du1;            jacos[3][3][2] = 2*u4du1;
#else
    jacos[0][0][0] = 0; jacos[0][0][1] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1;

    jacos[1][0][0] = kdu1-u2du1*u2du1; jacos[1][0][1] = -u2du1*u3du1;
    jacos[1][1][0] = 2*u2du1;          jacos[1][1][1] = u3du1;
    jacos[1][2][0] = 0;                jacos[1][2][1] = u2du1;

    jacos[2][0][0] = -u2du1*u3du1;     jacos[2][0][1] = kdu1-u3du1*u3du1;
    jacos[2][1][0] = u3du1;            jacos[2][1][1] = 0;
    jacos[2][2][0] = u2du1;            jacos[2][2][1] = 2*u3du1;
#endif
    */


    // pressure base
    //
    kdu1 = bulk/(u1*eta);
    bulkm = bulk*log(u1)/eta;
    u2u3 = u2*u3;
    u2du1 = u2/u1; u3du1 = u3/u1;
#if NDIM == 3
    u2u4 = u2*u4; u3u4 = u3*u4;
    u4du1 = u4/u1;
#endif
#if NDIM == 3
    fcn[0][0] = u2;               fcn[0][1] = u3;               fcn[0][2] = u4;
    fcn[1][0] = u2*u2du1 + bulkm; fcn[1][1] = u2*u3du1;         fcn[1][2] = u2*u4du1;
    fcn[2][0] = u3*u2du1;         fcn[2][1] = u3*u3du1 + bulkm; fcn[2][2] = u3*u4du1;
    fcn[3][0] = u4*u2du1;         fcn[3][1] = u4*u3du1;         fcn[3][2] = u4*u4du1 + bulkm;
#else
    fcn[0][0] = u2;               fcn[0][1] = u3;
    fcn[1][0] = u2*u2du1 + bulkm; fcn[1][1] = u2*u3du1;
    fcn[2][0] = u3*u2du1;         fcn[2][1] = u3*u3du1 + bulkm;
#endif   
#if NDIM == 3
    jacos[0][0][0] = 0; jacos[0][0][1] = 0; jacos[0][0][2] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0; jacos[0][1][2] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1; jacos[0][2][2] = 0;
    jacos[0][3][0] = 0; jacos[0][3][1] = 0; jacos[0][3][2] = 1;

    jacos[1][0][0] = kdu1/eta-u2du1*u2du1; jacos[1][0][1] = -u2du1*u3du1; jacos[1][0][2] = -u2du1*u4du1;
    jacos[1][1][0] = 2*u2du1;              jacos[1][1][1] = u3du1;        jacos[1][1][2] = u4du1;
    jacos[1][2][0] = 0;                    jacos[1][2][1] = u2du1;        jacos[1][2][2] = 0;
    jacos[1][3][0] = 0;                    jacos[1][3][1] = 0;            jacos[1][3][2] = u2du1;

    jacos[2][0][0] = -u2du1*u3du1; jacos[2][0][1] = kdu1/eta-u3du1*u3du1; jacos[2][0][2] = -u3du1*u4du1;
    jacos[2][1][0] = u3du1;        jacos[2][1][1] = 0;                    jacos[2][1][2] = 0;
    jacos[2][2][0] = u2du1;        jacos[2][2][1] = 2*u3du1;              jacos[2][2][2] = u4du1;
    jacos[2][3][0] = 0;            jacos[2][3][1] = 0;                    jacos[2][3][2] = u3du1;

    jacos[3][0][0] = -u2du1*u4du1; jacos[3][0][1] = -u3du1*u4du1;     jacos[3][0][2] = kdu1/eta-u4du1*u4du1;
    jacos[3][1][0] = u4du1;        jacos[3][1][1] = 0;                jacos[3][1][2] = 0;
    jacos[3][2][0] = 0;            jacos[3][2][1] = u4du1;            jacos[3][2][2] = 0;
    jacos[3][3][0] = u2du1;        jacos[3][3][1] = u3du1;            jacos[3][3][2] = 2*u4du1;
#else
    jacos[0][0][0] = 0; jacos[0][0][1] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1;

    jacos[1][0][0] = kdu1/eta-u2du1*u2du1; jacos[1][0][1] = -u2du1*u3du1;
    jacos[1][1][0] = 2*u2du1;              jacos[1][1][1] = u3du1;
    jacos[1][2][0] = 0;                    jacos[1][2][1] = u2du1;

    jacos[2][0][0] = -u2du1*u3du1;     jacos[2][0][1] = kdu1/eta-u3du1*u3du1;
    jacos[2][1][0] = u3du1;            jacos[2][1][1] = 0;
    jacos[2][2][0] = u2du1;            jacos[2][2][1] = 2*u3du1;
#endif
    //

    return;
};

#ifdef __CUDACC__
#include "calc_soltn.c"
#include "calc_dsoln_w1.c"
#include "calc_dsoln_w3.c"
#endif

// vim: set ft=cuda ts=4 et:
