#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>

namespace march
{

typedef int32_t index_type;
static constexpr index_type INVALID_INDEX = INT32_MAX;

/**
 * The primitive data type for element shape type.  May use only a single byte
 * but now take 4 bytes for legacy compatibility.
 */
typedef int32_t shape_type;

typedef double real_type;

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
