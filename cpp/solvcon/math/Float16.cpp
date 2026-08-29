/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/Float16.hpp>

#ifdef _MSC_VER
// Tell MSVC that calls in this translation unit observe the floating-point
// environment changed by std::fesetround().
#pragma fenv_access(on)
#endif

// Keep fenv-aware arithmetic out of non-strict callers so it observes the rounding mode selected by std::fesetround().
#ifdef _MSC_VER
#define SC_DECL_FLOAT16_NOINLINE __declspec(noinline)
#elif defined(__clang__) || defined(__GNUC__)
#define SC_DECL_FLOAT16_NOINLINE __attribute__((noinline))
#else
#define SC_DECL_FLOAT16_NOINLINE
#endif

namespace solvcon
{

#ifdef _MSC_VER
// Prevent LTCG from inlining this fenv-aware query into a caller.
__declspec(noinline) int Float16::rounding_mode()
{
    return std::fegetround();
}
#endif

SC_DECL_FLOAT16_NOINLINE Float16 Float16::add_runtime(float lhs, float rhs)
{
    return from_bits(encode_fallback(lhs + rhs));
}

SC_DECL_FLOAT16_NOINLINE Float16 Float16::sub_runtime(float lhs, float rhs)
{
    return from_bits(encode_fallback(lhs - rhs));
}

SC_DECL_FLOAT16_NOINLINE Float16 Float16::mul_runtime(float lhs, float rhs)
{
    return from_bits(encode_fallback(lhs * rhs));
}

SC_DECL_FLOAT16_NOINLINE Float16 Float16::div_runtime(float lhs, float rhs)
{
    return from_bits(encode_fallback(lhs / rhs));
}

} /* end namespace solvcon */

#undef SC_DECL_FLOAT16_NOINLINE

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
