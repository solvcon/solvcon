/*
 * Copyright (C) 2010 Yung-Yu Chen <yyc@solvcon.net>.
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

#include "euler.h"

void calc_jaco_slow(exedata *exd, int icl,
        double fcn[NEQ][NDIM], double jacos[NEQ][NEQ][NDIM]) {
    // pointers.
    double *psol;
    // scalars.
    double ga, ga1, ga3, ga1h, ga3h;
    double u1, u2, u3, u4;
#if NDIM == 3
    double u5;
#endif

    // initialize values.
    ga = exd->amsca[icl*NSCA];
    ga1 = ga-1; ga1h = ga1/2;
    ga3 = ga-3; ga3h = ga3/2;
    psol = exd->sol + icl*NEQ;
    u1 = psol[0];
    u2 = psol[1];
    u3 = psol[2];
    u4 = psol[3];
#if NDIM == 3
    u5 = psol[4];
#endif

    // flux function.
#if NDIM == 3
    fcn[0][0] = u2;
    fcn[1][0] = ga1*u5 - ga3h*u2*u2/u1 - ga1h*u3*u3/u1 - ga1h*u4*u4/u1;
    fcn[2][0] = u2*u3/u1;
    fcn[3][0] = u2*u4/u1;
    fcn[4][0] = ga*u2*u5/u1 - ga1h*(u2*u2+u3*u3+u4*u4)*u2/u1/u1;
    fcn[0][1] = u3;
    fcn[1][1] = u2*u3/u1;
    fcn[2][1] = ga1*u5 - ga1h*u2*u2/u1 - ga3h*u3*u3/u1 - ga1h*u4*u4/u1;
    fcn[3][1] = u3*u4/u1;
    fcn[4][1] = ga*u3*u5/u1 - ga1h*(u2*u2+u3*u3+u4*u4)*u3/u1/u1;
    fcn[0][2] = u4;
    fcn[1][2] = u2*u4/u1;
    fcn[2][2] = u3*u4/u1;
    fcn[3][2] = ga1*u5 - ga1h*u2*u2/u1 - ga1h*u3*u3/u1 - ga3h*u4*u4/u1;
    fcn[4][2] = ga*u4*u5/u1 - ga1h*(u2*u2+u3*u3+u4*u4)*u4/u1/u1;
#else
    fcn[0][0] = u2;
    fcn[1][0] = ga1*u4 - ga3h*u2*u2/u1 - ga1h*u3*u3/u1;
    fcn[2][0] = u2*u3/u1;
    fcn[3][0] = ga*u2*u4/u1 - ga1h*(u2*u2+u3*u3)*u2/u1/u1;
    fcn[0][1] = u3;
    fcn[1][1] = u2*u3/u1;
    fcn[2][1] = ga1*u4 - ga1h*u2*u2/u1 - ga3h*u3*u3/u1;
    fcn[3][1] = ga*u3*u4/u1 - ga1h*(u2*u2+u3*u3)*u3/u1/u1;
#endif
 
    // Jacobian matrices.
#if NDIM == 3
    jacos[0][0][0] = 0; jacos[0][0][1] = 0; jacos[0][0][2] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0; jacos[0][1][2] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1; jacos[0][2][2] = 0;
    jacos[0][3][0] = 0; jacos[0][3][1] = 0; jacos[0][3][2] = 1;
    jacos[0][4][0] = 0; jacos[0][4][1] = 0; jacos[0][4][2] = 0;

    jacos[1][0][0] = (ga3h*u2*u2 + ga1h*u3*u3 + ga1h*u4*u4)/(u1*u1);
    jacos[1][1][0] = -ga3*u2/u1;
    jacos[1][2][0] = -ga1*u3/u1;
    jacos[1][3][0] = -ga1*u4/u1;
    jacos[1][4][0] = ga1;
    jacos[2][0][0] = -u2*u3/(u1*u1);
    jacos[2][1][0] = u3/u1;
    jacos[2][2][0] = u2/u1;
    jacos[2][3][0] = jacos[2][4][0] = 0;
    jacos[3][0][0] = -u2*u4/(u1*u1);
    jacos[3][1][0] = u4/u1;
    jacos[3][3][0] = u2/u1;
    jacos[3][2][0] = jacos[3][4][0] = 0;
    jacos[4][0][0] = -ga*u2*u5/(u1*u1) + ga1*(u2*u2+u3*u3+u4*u4)*u2/(u1*u1*u1);
    jacos[4][1][0] = ga*u5/u1 - ga1h*(3*u2*u2 + u3*u3 + u4*u4)/(u1*u1);
    jacos[4][2][0] = -ga1*u2*u3/(u1*u1);
    jacos[4][3][0] = -ga1*u2*u4/(u1*u1);
    jacos[4][4][0] = ga*u2/u1;

    jacos[1][0][1] = -u2*u3/(u1*u1);
    jacos[1][1][1] = u3/u1;
    jacos[1][2][1] = u2/u1;
    jacos[1][3][1] = jacos[1][4][1] = 0;
    jacos[2][0][1] = (ga1h*u2*u2 + ga3h*u3*u3 + ga1h*u4*u4)/(u1*u1);
    jacos[2][1][1] = -ga1*u2/u1;
    jacos[2][2][1] = -ga3*u3/u1;
    jacos[2][3][1] = -ga1*u4/u1;
    jacos[2][4][1] = ga1;
    jacos[3][0][1] = -u3*u4/(u1*u1);
    jacos[3][2][1] = u4/u1;
    jacos[3][3][1] = u3/u1;
    jacos[3][1][1] = jacos[3][4][1] = 0;
    jacos[4][0][1] = -ga*u3*u5/(u1*u1) + ga1*(u2*u2+u3*u3+u4*u4)*u3/(u1*u1*u1);
    jacos[4][1][1] = -ga1*u2*u3/(u1*u1);
    jacos[4][2][1] = ga*u5/u1 - ga1h*(u2*u2 + 3*u3*u3 + u4*u4)/(u1*u1);
    jacos[4][3][1] = -ga1*u3*u4/(u1*u1);
    jacos[4][4][1] = ga*u3/u1;

    jacos[1][0][2] = -u2*u4/(u1*u1);
    jacos[1][1][2] = u4/u1;
    jacos[1][3][2] = u2/u1;
    jacos[1][2][2] = jacos[1][4][2] = 0;
    jacos[2][0][2] = -u3*u4/(u1*u1);
    jacos[2][2][2] = u4/u1;
    jacos[2][3][2] = u3/u1;
    jacos[2][1][2] = jacos[2][4][2] = 0;
    jacos[3][0][2] = (ga1h*u2*u2 + ga1h*u3*u3 + ga3h*u4*u4)/(u1*u1);
    jacos[3][1][2] = -ga1*u2/u1;
    jacos[3][2][2] = -ga1*u3/u1;
    jacos[3][3][2] = -ga3*u4/u1;
    jacos[3][4][2] = ga1;
    jacos[4][0][2] = -ga*u4*u5/(u1*u1) + ga1*(u2*u2+u3*u3+u4*u4)*u4/(u1*u1*u1);
    jacos[4][1][2] = -ga1*u2*u4/(u1*u1);
    jacos[4][2][2] = -ga1*u3*u4/(u1*u1);
    jacos[4][3][2] = ga*u5/u1 - ga1h*(u2*u2 + u3*u3 + 3*u4*u4)/(u1*u1);
    jacos[4][4][2] = ga*u4/u1;
#else
    jacos[0][0][0] = 0; jacos[0][0][1] = 0;
    jacos[0][1][0] = 1; jacos[0][1][1] = 0;
    jacos[0][2][0] = 0; jacos[0][2][1] = 1;
    jacos[0][3][0] = 0; jacos[0][3][1] = 0;

    jacos[1][0][0] = (ga3h*u2*u2 + ga1h*u3*u3)/(u1*u1);
    jacos[1][1][0] = -ga3*u2/u1;
    jacos[1][2][0] = -ga1*u3/u1;
    jacos[1][3][0] = ga1;
    jacos[2][0][0] = -u2*u3/(u1*u1);
    jacos[2][1][0] = u3/u1;
    jacos[2][2][0] = u2/u1;
    jacos[2][3][0] = 0;
    jacos[3][0][0] = -ga*u2*u4/(u1*u1) + ga1*(u2*u2+u3*u3)*u2/(u1*u1*u1);
    jacos[3][1][0] = ga*u4/u1 - ga1h*(3*u2*u2 + u3*u3)/(u1*u1);
    jacos[3][2][0] = -ga1*u2*u3/(u1*u1);
    jacos[3][3][0] = ga*u2/u1;

    jacos[1][0][1] = -u2*u3/(u1*u1);
    jacos[1][1][1] = u3/u1;
    jacos[1][2][1] = u2/u1;
    jacos[1][3][1] = 0;
    jacos[2][0][1] = (ga1h*u2*u2 + ga3h*u3*u3)/(u1*u1);
    jacos[2][1][1] = -ga1*u2/u1;
    jacos[2][2][1] = -ga3*u3/u1;
    jacos[2][3][1] = ga1;
    jacos[3][0][1] = -ga*u3*u4/(u1*u1) + ga1*(u2*u2+u3*u3)*u3/(u1*u1*u1);
    jacos[3][1][1] = -ga1*u2*u3/(u1*u1);
    jacos[3][2][1] = ga*u4/u1 - ga1h*(u2*u2 + 3*u3*u3)/(u1*u1);
    jacos[3][3][1] = ga*u3/u1;
#endif

    return;
};
// vim: set ts=4 et:
