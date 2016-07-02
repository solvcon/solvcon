#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * \file
 * Utilities.
 */

#include <cstring>

namespace march {

namespace detail {

inline static constexpr size_t log2(size_t n, int k = 0) { return (n <= 1) ? k : log2(n >> 1, k + 1); }

} /* end namespace detail */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
