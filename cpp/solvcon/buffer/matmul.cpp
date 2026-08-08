/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/matmul.hpp>

#include <algorithm>
#include <array>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

template <typename T, size_t Side, size_t ColumnStart, size_t ColumnCount, bool LhsInnerIsUnitStride>
static void run_fixed_ikj_block(
    T * output,
    T const * lhs,
    T const * rhs,
    ssize_t lhs_row_stride,
    ssize_t lhs_inner_stride,
    ssize_t rhs_inner_stride)
{
    ssize_t constexpr side = static_cast<ssize_t>(Side);
    for (ssize_t row = 0; row < side; ++row)
    {
        std::array<T, ColumnCount> sums{};
        T const * lhs_value_ptr = lhs + row * lhs_row_stride;
        ssize_t rhs_row_base = 0;
        for (ssize_t inner = 0; inner < side; ++inner)
        {
            T const lhs_value = *lhs_value_ptr;
            T const * rhs_row = rhs + rhs_row_base + ColumnStart;
            for (size_t column = 0; column < ColumnCount; ++column)
            {
                sums[column] += lhs_value * rhs_row[column];
            }
            lhs_value_ptr += LhsInnerIsUnitStride ? 1 : lhs_inner_stride;
            rhs_row_base += rhs_inner_stride;
        }
        std::copy(sums.begin(), sums.end(), output + row * side + ColumnStart);
    }
}

template <typename T, size_t Side, bool LhsInnerIsUnitStride>
static void run_fixed_ikj_loop(
    T * output,
    T const * lhs,
    T const * rhs,
    ssize_t lhs_row_stride,
    ssize_t lhs_inner_stride,
    ssize_t rhs_inner_stride)
{
    size_t constexpr block_size = std::min<size_t>(Side, 8);
    run_fixed_ikj_block<T, Side, 0, block_size, LhsInnerIsUnitStride>(
        output, lhs, rhs, lhs_row_stride, lhs_inner_stride, rhs_inner_stride);
    if constexpr (Side > block_size)
    {
        run_fixed_ikj_block<T, Side, block_size, Side - block_size, LhsInnerIsUnitStride>(
            output, lhs, rhs, lhs_row_stride, lhs_inner_stride, rhs_inner_stride);
    }
}

template <typename T, size_t Side>
static void run_fixed_ikj(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan,
    bool lhs_inner_is_unit_stride)
{
    if (lhs_inner_is_unit_stride)
    {
        run_fixed_ikj_loop<T, Side, true>(
            output, lhs, rhs, plan.lhs_row_stride(), plan.lhs_inner_stride(), plan.rhs_inner_stride());
    }
    else
    {
        run_fixed_ikj_loop<T, Side, false>(
            output, lhs, rhs, plan.lhs_row_stride(), plan.lhs_inner_stride(), plan.rhs_inner_stride());
    }
}

template <typename T, size_t Side, size_t RowStart, size_t RowCount>
static void run_fixed_jki_block(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    ssize_t constexpr side = static_cast<ssize_t>(Side);
    for (ssize_t column = 0; column < side; ++column)
    {
        std::array<T, RowCount> sums{};
        T const * rhs_value = rhs + column * plan.rhs_column_stride();
        ssize_t lhs_inner_base = 0;
        for (ssize_t inner = 0; inner < side; ++inner)
        {
            T const rhs_scalar = *rhs_value;
            for (size_t row = 0; row < RowCount; ++row)
            {
                ssize_t const lhs_offset = static_cast<ssize_t>(RowStart + row) * plan.lhs_row_stride();
                sums[row] += lhs[lhs_offset + lhs_inner_base] * rhs_scalar;
            }
            lhs_inner_base += plan.lhs_inner_stride();
            rhs_value += plan.rhs_inner_stride();
        }
        for (size_t row = 0; row < RowCount; ++row)
        {
            output[(RowStart + row) * Side + column] = sums[row];
        }
    }
}

template <typename T, size_t Side>
static void run_fixed_jki(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    size_t constexpr block_size = std::min<size_t>(Side, 8);
    run_fixed_jki_block<T, Side, 0, block_size>(output, lhs, rhs, plan);
    if constexpr (Side > block_size)
    {
        run_fixed_jki_block<T, Side, block_size, Side - block_size>(output, lhs, rhs, plan);
    }
}

template <typename T>
static void dispatch_fixed_ikj(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan,
    bool lhs_inner_is_unit_stride)
{
    switch (plan.rows())
    {
    case 8:
        run_fixed_ikj<T, 8>(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
        return;
    case 9:
        run_fixed_ikj<T, 9>(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
        return;
    case 10:
        run_fixed_ikj<T, 10>(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
        return;
    case 11:
        run_fixed_ikj<T, 11>(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
        return;
    case 12:
        run_fixed_ikj<T, 12>(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
        return;
    case 13:
        run_fixed_ikj<T, 13>(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
        return;
    case 14:
        run_fixed_ikj<T, 14>(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
        return;
    case 15:
        run_fixed_ikj<T, 15>(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
        return;
    default: throw std::logic_error("execute_fixed_ikj(): unsupported matrix side");
    }
}

template <typename T>
static void dispatch_fixed_jki(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    switch (plan.rows())
    {
    case 8:
        run_fixed_jki<T, 8>(output, lhs, rhs, plan);
        return;
    case 9:
        run_fixed_jki<T, 9>(output, lhs, rhs, plan);
        return;
    case 10:
        run_fixed_jki<T, 10>(output, lhs, rhs, plan);
        return;
    case 11:
        run_fixed_jki<T, 11>(output, lhs, rhs, plan);
        return;
    case 12:
        run_fixed_jki<T, 12>(output, lhs, rhs, plan);
        return;
    case 13:
        run_fixed_jki<T, 13>(output, lhs, rhs, plan);
        return;
    case 14:
        run_fixed_jki<T, 14>(output, lhs, rhs, plan);
        return;
    case 15:
        run_fixed_jki<T, 15>(output, lhs, rhs, plan);
        return;
    default: throw std::logic_error("execute_fixed_jki(): unsupported matrix side");
    }
}

void execute_fixed_ikj(
    float * output,
    float const * lhs,
    float const * rhs,
    MatmulPlan const & plan,
    bool lhs_inner_is_unit_stride)
{
    dispatch_fixed_ikj(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
}

void execute_fixed_ikj(
    double * output,
    double const * lhs,
    double const * rhs,
    MatmulPlan const & plan,
    bool lhs_inner_is_unit_stride)
{
    dispatch_fixed_ikj(output, lhs, rhs, plan, lhs_inner_is_unit_stride);
}

void execute_fixed_jki(
    float * output,
    float const * lhs,
    float const * rhs,
    MatmulPlan const & plan)
{
    dispatch_fixed_jki(output, lhs, rhs, plan);
}

void execute_fixed_jki(
    double * output,
    double const * lhs,
    double const * rhs,
    MatmulPlan const & plan)
{
    dispatch_fixed_jki(output, lhs, rhs, plan);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
