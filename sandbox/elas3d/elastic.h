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

#ifndef SOLVCON_ELASTIC_H
#define SOLVCON_ELASTIC_H
#include <stdio.h>
#include <fenv.h>
#include "cese.h"

#undef NEQ
#define NEQ 9

int get_transformation(double *nvec, double *svec,
        double rotm[][3], double bondm[][6], double bondminv[][6]);

// lapack.
#define lapack_dgeev dgeev_
extern void lapack_dgeev(char *jobvl, char *jobvr, int *n, double *a, int *lda,
        double *wr, double *wi, double *vl, int *ldvl, double *vr, int *ldvr,
        double *work, int *lwork, int *info);

#endif
// vim: set ts=4 et:
