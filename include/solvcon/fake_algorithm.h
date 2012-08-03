/*
 * Copyright (C) 2012 Yung-Yu Chen <yyc@solvcon.net>.
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

#ifndef _SOLVCON_FAKE_ALGORITHM_H_
#define _SOLVCON_FAKE_ALGORITHM_H_

#include "solvcon/mesh.h"

typedef struct {
	int ncore, neq;
	double time, time_increment;
    double *sol, *soln, *dsol, *dsoln;
    double *cecnd, *cevol;
} sc_fake_algorithm_t;

int sc_fake_algorithm_calc_soln(sc_mesh_t *msd, sc_fake_algorithm_t *exd);
int calc_dsoln(sc_mesh_t *msd, sc_fake_algorithm_t *exd);

#endif

// vim: fenc=utf8 ff=unix ft=c ai et sw=4 ts=4 tw=79:
