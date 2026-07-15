#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <functional>

#include <solvcon/simd/neon/neon_alias.hpp>
#include <solvcon/simd/neon/neon_type.hpp>
#include <solvcon/simd/simd_generic.hpp>

#ifdef __aarch64__
#include <arm_neon.h>
#endif /* defined(__aarch64__) */

namespace solvcon
{

namespace simd
{

namespace neon
{

namespace detail
{

#ifndef NDEBUG
template <typename T>
bool is_aligned(T const * pointer, size_t alignment)
{
    return (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

template <typename T>
void check_alignment(T const * pointer, size_t required_alignment, const char * name)
{
    if (!is_aligned(pointer, required_alignment))
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,modernize-use-std-print,cert-err33-c)
        std::fprintf(stderr,
                     "Warning: %s pointer %p is not aligned to %zu bytes. "
                     "SIMD performance may be degraded.\n",
                     name,
                     static_cast<const void *>(pointer),
                     required_alignment);
    }
}
#endif // NDEBUG

// Get the recommended memory alignment for SIMD operations based on the detected SIMD instruction set.
inline constexpr size_t get_recommended_alignment()
{
#if defined(__aarch64__) || defined(__arm__)
    return 16;
// TODO: The non-NEON conditional should be factored out elsewhere in the future.
#elifdef __AVX512F__
    return 64;
#elif defined(__AVX__) || defined(__AVX2__)
    return 32;
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__)
    return 16;
#else
    return 0;
#endif
}

} /* end namespace detail */

#ifdef __aarch64__
// SFINAE helpers for vectorized operations.
// The trailing decltype return types are required for the SFINAE used by
// transform_binary below, so fuchsia-trailing-return does not apply here.
// NOLINTBEGIN(fuchsia-trailing-return)
struct vec_add
{
    template <typename V>
    static auto operator()(V a, V b) -> decltype(vaddq(a, b)) { return vaddq(a, b); }
}; /* end struct vec_add */
struct vec_sub
{
    template <typename V>
    static auto operator()(V a, V b) -> decltype(vsubq(a, b)) { return vsubq(a, b); }
}; /* end struct vec_sub */
struct vec_mul
{
    template <typename V>
    static auto operator()(V a, V b) -> decltype(vmulq(a, b)) { return vmulq(a, b); }
}; /* end struct vec_mul */
struct vec_div
{
    template <typename V>
    static auto operator()(V a, V b) -> decltype(vdivq(a, b)) { return vdivq(a, b); }
}; /* end struct vec_div */
// NOLINTEND(fuchsia-trailing-return)

template <typename T, std::invocable<T, T> ScalarOp, typename VecOp>
void transform_binary(T * dest, T const * dest_end, T const * src1, T const * src2, ScalarOp scalar_op, VecOp vec_op)
{
    if constexpr (!type::has_vectype<T>)
    {
        generic::transform_binary<T>(dest, dest_end, src1, src2, scalar_op);
    }
    else
    {
        using vec_t = type::vector_t<T>;
        if constexpr (!std::invocable<VecOp, vec_t, vec_t>)
        {
            generic::transform_binary<T>(dest, dest_end, src1, src2, scalar_op);
        }
        else
        {
            constexpr size_t N_lane = type::vector_lane<T>;

#ifndef NDEBUG
            constexpr size_t alignment = detail::get_recommended_alignment();
            detail::check_alignment(dest, alignment, "transform_binary dest");
            detail::check_alignment(src1, alignment, "transform_binary src1");
            detail::check_alignment(src2, alignment, "transform_binary src2");
#endif

            // Counted trip form. `ptr <= dest_end - N_lane` is UB on sub-lane
            // inputs (forms a pointer before the buffer); `dest_end - ptr >=
            // N_lane` is safe but adds a non-fusable `sub` per iteration
            // (~20-25% hit on cache-resident NEON loops). Hoisting the block
            // count keeps both safety and lets the loop fold to `subs/b.ne`.
            size_t const blocks = static_cast<size_t>(dest_end - dest) / N_lane;
            T * ptr = dest;
            for (size_t i = 0; i < blocks; ++i)
            {
                vec_t const v1 = vld1q(src1);
                vec_t const v2 = vld1q(src2);
                vst1q(ptr, vec_op(v1, v2));
                ptr += N_lane;
                src1 += N_lane;
                src2 += N_lane;
            }
            while (ptr < dest_end)
            {
                *ptr = scalar_op(*src1, *src2);
                ++ptr;
                ++src1;
                ++src2;
            }
        }
    }
}

template <typename T>
inline void add(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::plus<T>{}, vec_add{});
}

template <typename T>
inline void sub(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::minus<T>{}, vec_sub{});
}

template <typename T>
inline void mul(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::multiplies<T>{}, vec_mul{});
}

template <typename T>
inline void div(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    transform_binary<T>(dest, dest_end, src1, src2, std::divides<T>{}, vec_div{});
}

template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    if constexpr (!type::has_vectype<T>)
    {
        return generic::check_between<T>(start, end, min_val, max_val);
    }
    else
    {
        using vec_t = type::vector_t<T>;
        using cmpvec_t = type::vector_t<uint64_t>;
        constexpr size_t N_lane = type::vector_lane<T>;

#ifndef NDEBUG
        constexpr size_t alignment = detail::get_recommended_alignment();
        detail::check_alignment(start, alignment, "check_between start");
#endif

        vec_t const max_vec = vdupq(max_val);
        vec_t const min_vec = vdupq(min_val);

        // Vector loop runs while a full lane still fits. Counted trip form
        // for the same reason as transform_binary above: avoids UB on
        // sub-lane inputs and the per-iter `sub` overhead.
        size_t const blocks = static_cast<size_t>(end - start) / N_lane;
        T const * ptr = start;
        for (size_t block = 0; block < blocks; ++block)
        {
            vec_t const data_vec = vld1q(ptr);

            // Inspect both bounds in one pass so the lowest-index failing lane
            // wins; callers report this pointer as the first out-of-range
            // element.
            auto const ge_vec = (cmpvec_t)vcgeq(data_vec, max_vec); // NOLINT(modernize-avoid-c-style-cast)
            auto const lt_vec = (cmpvec_t)vcltq(data_vec, min_vec); // NOLINT(modernize-avoid-c-style-cast)
            bool const out_of_range = vgetq<0>(ge_vec) || vgetq<1>(ge_vec) || vgetq<0>(lt_vec) || vgetq<1>(lt_vec);

            if (out_of_range)
            {
                T ge_val[N_lane] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
                T lt_val[N_lane] = {}; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
                vst1q(ge_val, ge_vec);
                vst1q(lt_val, lt_vec);
                for (size_t i = 0; i < N_lane; ++i)
                {
                    if (ge_val[i] || lt_val[i])
                    {
                        return ptr + i;
                    }
                }
                return ptr;
            }

            ptr += N_lane;
        }

        // Tail scalar loop for remaining elements
        while (ptr < end)
        {
            if (*ptr < min_val || *ptr > max_val)
            {
                return ptr;
            }
            ++ptr;
        }
        return nullptr;
    }
}

#else
template <typename T>
const T * check_between(T const * start, T const * end, T const & min_val, T const & max_val)
{
    return generic::check_between<T>(start, end, min_val, max_val);
}

template <typename T>
void add(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    generic::add<T>(dest, dest_end, src1, src2);
}

template <typename T>
void sub(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    generic::sub<T>(dest, dest_end, src1, src2);
}

template <typename T>
void mul(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    generic::mul<T>(dest, dest_end, src1, src2);
}

template <typename T>
void div(T * dest, T const * dest_end, T const * src1, T const * src2)
{
    generic::div<T>(dest, dest_end, src1, src2);
}

#endif /* defined(__aarch64__) */

} /* end namespace neon */

} /* end namespace simd */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
