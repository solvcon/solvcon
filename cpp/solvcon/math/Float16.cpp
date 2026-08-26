/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/Float16.hpp>

#ifdef _MSC_VER
// Tell MSVC that calls in this translation unit observe the floating-point
// environment changed by std::fesetround().
#pragma fenv_access(on)

namespace solvcon
{

// Prevent LTCG from inlining this fenv-aware query into a caller.
__declspec(noinline) int Float16::rounding_mode()
{
    return std::fegetround();
}

} /* end namespace solvcon */
#endif

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
