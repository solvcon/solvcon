#ifndef _SOLVCON
#define _SOLVCON

#include <math.h>

typedef struct {
    int ndim;
    int fcmnd, clmnd, clmfc;
    int nnode, nface, ncell, nbound;
    int ngstnode, ngstface, ngstcell;
    // metric.
    FPTYPE *ndcrd, *fccnd, *fcnml, *fcara, *clcnd, *clvol;
    // meta.
    int *fctpn, *cltpn, *clgrp;
    // connectivity.
    int *fcnds, *fccls, *clnds, *clfcs;
} MeshData;

#endif
