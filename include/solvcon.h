/*
 * Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
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

#ifndef _SOLVCON
#define _SOLVCON

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <fenv.h>
#include <stddef.h>
#include <sys/times.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * quantities.
 */
#define PI 3.14159265358979311600
#define SOLVCON_ALMOST_ZERO 1.e-200
#define SOLVCON_TINY 1.e-60
#define SOLVCON_SMALL 1.e-30

/*
 * mesh constants.
 */
#define FCMND 4
#define CLMND 8
#define CLMFC 6
#define FCREL 4
#define BFREL 3

/*
 * mesh structure.
 */
typedef struct {
    int ndim;
    int nnode, nface, ncell, nbound;
    int ngstnode, ngstface, ngstcell;
    // metric.
    FPTYPE *ndcrd, *fccnd, *fcnml, *fcara, *clcnd, *clvol;
    // meta.
    int *fctpn, *cltpn, *clgrp;
    // connectivity.
    int *fcnds, *fccls, *clnds, *clfcs;
} MeshData;

/*
 * example execution data.
 */
typedef struct {
	int ncore, neq;
	double time, time_increment;
} ExecutionData;

/*
 * Debugging.
 */
// floating point exception.
#ifdef SOLVCON_DEBUG
//#define SOLVCESE_FE FE_ALL_EXCEPT
#define SOLVCON_FE FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW
//#define SOLVCON_FE FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW
#endif

#endif
