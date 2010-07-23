// Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of
// usage.
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
