/*
 * Copyright (C) 2008-2013 Yung-Yu Chen <yyc@solvcon.net>.
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

#include <Python.h>

#include "mesh.h"
#include "lincese_algorithm.h"

#define ALMOST_ZERO 1.e-200

#undef NDIM
#define NDIM 2
#include "sc_lincese_algorithm_prepare_sf.c_body"
#undef NDIM
#define NDIM 3
#include "sc_lincese_algorithm_prepare_sf.c_body"

// vim: set ft=cuda ts=4 et:
