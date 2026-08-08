#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * The interface master header file for the linear algebraic code.
 */

#include <solvcon/linalg/factorization.hpp>
#include <solvcon/linalg/lu_factorization.hpp>
#include <solvcon/linalg/kalman_filter.hpp>
#ifdef SC_HAS_VENDOR_LAPACK
#include <solvcon/linalg/EigenSystem.hpp>
#endif

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
