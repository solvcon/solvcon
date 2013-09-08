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
__device__ void cuda_calc_viscous(exedata *exd, int icl,
        double visc[NEQ][NDIM]) {
#else
void calc_viscous(exedata *exd, int icl,
        double visc[NEQ][NDIM]) {
#endif
    // pointers.
    double *psol, *pdsol;
    double *pamsca;
    // scalars.
    double bulk, p0, rho0, eta, mu, d, Re, vel, rho;
    double u1, u2, u3;
#if NDIM == 3
    double u4;
#endif
    // accelerating variables.
    double dudx, dudy, dvdx, dvdy;
    pamsca = exd->amsca + icl*NSCA;
    // initialize values.
    bulk = pamsca[0];
    p0 = pamsca[1];
    rho0 = pamsca[2];
    eta = pamsca[3];
    mu = pamsca[6];
    d = pamsca[7];
    psol = exd->sol + icl*NEQ;
    pdsol = exd->dsol + icl*NEQ*NDIM;
    u1 = psol[0] + SOLVCON_TINY;
    u2 = psol[1];
    u3 = psol[2];
    vel = u2*u2 + u3*u3;
#if NDIM == 3
    u4 = psol[3];
    vel += u4*u4;
#endif
    vel = sqrt(vel/(u1*u1));

    dudx = u2/u1*pdsol[0];
    dudy = u2/u1*pdsol[1];
    dvdx = u3/u1*pdsol[0];
    dvdy = u3/u1*pdsol[1];
    pdsol += NDIM;
    dudx = (pdsol[0]-dudx)/u1;
    dudy = (pdsol[1]-dudy)/u1;
    pdsol += NDIM;
    dvdx = (pdsol[0]-dvdx)/u1;
    dvdy = (pdsol[1]-dvdy)/u1;

    visc[0][0] = 0;                           visc[0][1] = 0;
    visc[1][0] = mu*(2*dudx-2/3*(dudx+dvdy)); visc[1][1] = mu*(dudy+dvdx);
    visc[2][0] = mu*(dudy+dvdx);              visc[2][1] = mu*(2*dvdy-2/3*(dudx+dvdy));

    return;
};

#ifdef __CUDACC__
#include "calc_soltn.c"
#include "calc_dsoln_w1.c"
#include "calc_dsoln_w3.c"
#endif

// vim: set ft=cuda ts=4 et:
