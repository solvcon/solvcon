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

#ifndef SOLVCON_CUEULER_H
#define SOLVCON_CUEULER_H
#include <stdio.h>
#include <fenv.h>
#include "cese.h"

// use CUDA max/min.
#undef max
#undef min

#undef NEQ
#if NDIM == 2
#define NEQ 4
#elif NDIM == 3
#define NEQ 5
#endif

#undef NSCA
#define NSCA 1

#endif
// vim: set ts=4 et:
