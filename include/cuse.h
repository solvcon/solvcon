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

#ifndef SOLVCON_CUSE_H
#define SOLVCON_CUSE_H

#define FPTYPE double
#include "solvcon.h"

#ifndef __CUDACC__
/*
 * shorthand for min/max.
 */
#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif
#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif
#endif

/*
 * Generic definitions and data structure.
 */
#define NEQ exd->neq
#define NSCA exd->nsca
#define NVEC exd->nvec
typedef struct {
    // inherited.
    int ncore, neq;
    double time, time_increment;
    // mesh shape.
    int ndim, nnode, nface, ncell, nbound;
    int ngstnode, ngstface, ngstcell;
    // group shape.
    int ngroup, gdlen;
    // parameter shape.
    int nsca, nvec;
    // function pointer.
    void (*jacofunc)(void *exd, int icl, double *fcn, double *jacos);
    double (*taufunc)(void *exd, int icl);
    double (*omegafunc)(void *exd, int icl);
    // scheme.
    int alpha;
    double taylor, cnbfac, sftfac;
    double taumin, taumax, tauscale;
    double omegamin, omegascale;
    // meta array.
    int *fctpn, *cltpn, *clgrp;
    double *grpda;
    // geometry array.
    double *ndcrd, *fccnd, *fcnml, *clcnd, *clvol, *cecnd, *cevol;
    // connectivity array.
    int *fcnds, *fccls, *clnds, *clfcs;
    // solutions array.
    double *sol, *dsol, *solt, *soln, *dsoln, *cfl, *ocfl, *amsca, *amvec;
} exedata;

#ifndef __CUDACC__
/*
 * mapping arrays.
 */
extern const int ggefcs[31][3];
extern const int ggerng[8][2];
extern const int sfcs[4][3];
extern const int sfng[4][2];
extern const int hvfs[8][2];
extern const int hrng[8][2];
extern const int evts[42][2];
extern const int egng[8][2];
#endif

#endif
// vim: set ts=4 et:

