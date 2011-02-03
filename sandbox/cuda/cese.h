/*
 * Copyright (C) 2008-2010 Yung-Yu Chen <yyc@solvcon.net>.
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

#ifndef SOLVCON_CESE_H
#define SOLVCON_CESE_H
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>
#include <sys/times.h>

/*
 * shorthand for min/max.
 */
#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif
#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

/*
 * quantities.
 */
#define PI 3.14159265358979311600
#define SOLVCESE_ALMOST_ZERO 1.e-200
#define SOLVCESE_TINY 1.e-60
#define SOLVCESE_SMALL 1.e-30

/*
 * Generic definitions and data structure.
 */
#define FCMND 4
#define CLMND 8
#define CLMFC 6
#define FCREL 4
#define MFGE 8
#define BFREL 3
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
    double mqmin, mqscale;
    // meta array.
    int *fctpn, *cltpn, *clgrp;
    double *grpda;
    // geometry array.
    double *ndcrd, *fccnd, *fcnml, *clcnd, *clvol, *cecnd, *cevol, *mqlty;
    // connectivity array.
    int *fcnds, *fccls, *clnds, *clfcs;
    // solutions array.
    double *sol, *dsol, *solt, *soln, *dsoln, *cfl, *ocfl, *amsca, *amvec;
} exedata;

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

/*
 * Debugging.
 */
//#define SOLVCESE_DEBUG
// floating point exception.
#ifdef SOLVCESE_DEBUG
//#define SOLVCESE_FE FE_ALL_EXCEPT
#define SOLVCESE_FE FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW
//#define SOLVCESE_FE FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW
#endif

#endif
// vim: set ts=4 et:
