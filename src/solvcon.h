#ifndef _SOLVCON
#define _SOLVCON

#include <stdio.h>
#include <math.h>

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

#define FCMND 4
#define CLMND 8
#define CLMFC 6
#define FCREL 4
#define BFREL 3

#endif
