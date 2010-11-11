/*
 * Copyright (C) 2008-2010 Yung-Yu Chen.
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

typedef struct {
	int ncore, neq;
	double time, time_increment;
} ExecutionData;

#define FCMND 4
#define CLMND 8
#define CLMFC 6
#define FCREL 4
#define BFREL 3

#endif
