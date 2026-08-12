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

/**
 * @brief Multiply one output-column block in IKJ order.
 *
 * @code
 * for (row) { for (inner) { for (column) { C[row][column] += A[row][inner] * B[inner][column]; } } }
 * @endcode
 */
template <typename T, size_t Side, size_t ColumnStart, size_t ColumnCount, bool LhsInnerIsUnitStride>
static void multiply_ikj_block(
    T * output,
    T const * lhs,
    T const * rhs,
    ssize_t lhs_row_stride,
    ssize_t lhs_inner_stride,
    ssize_t rhs_inner_stride)
{
    auto constexpr side = static_cast<ssize_t>(Side);
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
static void multiply_fixed_ikj(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    size_t constexpr block_size = std::min<size_t>(Side, 8);
    multiply_ikj_block<T, Side, 0, block_size, LhsInnerIsUnitStride>(
        output,
        lhs,
        rhs,
        plan.lhs_row_stride(),
        plan.lhs_inner_stride(),
        plan.rhs_inner_stride());
    if constexpr (Side > block_size)
    {
        multiply_ikj_block<T, Side, block_size, Side - block_size, LhsInnerIsUnitStride>(
            output,
            lhs,
            rhs,
            plan.lhs_row_stride(),
            plan.lhs_inner_stride(),
            plan.rhs_inner_stride());
    }
}

/**
 * @brief Multiply one output-row block in JKI order.
 *
 * @code
 * for (column) { for (inner) { for (row) { C[row][column] += A[row][inner] * B[inner][column]; } } }
 * @endcode
 */
template <typename T, size_t Side, size_t RowStart, size_t RowCount>
static void multiply_jki_block(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    auto constexpr side = static_cast<ssize_t>(Side);
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
static void multiply_fixed_jki(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    size_t constexpr block_size = std::min<size_t>(Side, 8);
    multiply_jki_block<T, Side, 0, block_size>(output, lhs, rhs, plan);
    if constexpr (Side > block_size)
    {
        multiply_jki_block<T, Side, block_size, Side - block_size>(output, lhs, rhs, plan);
    }
}

template <MatmulKernel Kernel, typename T, size_t Side, bool LhsInnerIsUnitStride = false>
static void multiply_fixed_matrix(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    if constexpr (Kernel == MatmulKernel::FixedIkj)
    {
        multiply_fixed_ikj<T, Side, LhsInnerIsUnitStride>(output, lhs, rhs, plan);
    }
    else
    {
        static_assert(Kernel == MatmulKernel::FixedJki);
        multiply_fixed_jki<T, Side>(output, lhs, rhs, plan);
    }
}

template <MatmulKernel Kernel, typename T, size_t Side, bool LhsInnerIsUnitStride = false>
static void traverse_fixed_batch(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    if (!plan.has_batch_axes())
    {
        multiply_fixed_matrix<Kernel, T, Side, LhsInnerIsUnitStride>(output, lhs, rhs, plan);
        return;
    }

    for (MappedOffsetCursor cursor = plan.batch_cursor(); cursor; cursor.advance())
    {
        multiply_fixed_matrix<Kernel, T, Side, LhsInnerIsUnitStride>(
            output + cursor.offset(MatmulPlan::Operand::Output),
            lhs + cursor.offset(MatmulPlan::Operand::Lhs),
            rhs + cursor.offset(MatmulPlan::Operand::Rhs),
            plan);
    }
}

template <MatmulKernel Kernel, typename T, size_t Side>
static void execute_fixed_side(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    if constexpr (Kernel == MatmulKernel::FixedIkj)
    {
        if (plan.lhs_inner_stride() == 1)
        {
            traverse_fixed_batch<Kernel, T, Side, true>(output, lhs, rhs, plan);
        }
        else
        {
            traverse_fixed_batch<Kernel, T, Side, false>(output, lhs, rhs, plan);
        }
    }
    else
    {
        static_assert(Kernel == MatmulKernel::FixedJki);
        traverse_fixed_batch<Kernel, T, Side>(output, lhs, rhs, plan);
    }
}

template <MatmulKernel Kernel, typename T, size_t Side>
static void dispatch_fixed_side(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    if (plan.rows() == static_cast<ssize_t>(Side))
    {
        execute_fixed_side<Kernel, T, Side>(output, lhs, rhs, plan);
    }
    else if constexpr (Side < static_cast<size_t>(FIXED_GEMM_COMPILED_MAX_SIDE))
    {
        dispatch_fixed_side<Kernel, T, Side + 1>(output, lhs, rhs, plan);
    }
    else
    {
        throw std::logic_error("dispatch_fixed_side(): unsupported matrix side");
    }
}

template <typename T>
void execute_fixed_gemm(
    MatmulKernel kernel,
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan)
{
    switch (kernel)
    {
    case MatmulKernel::FixedIkj:
        dispatch_fixed_side<MatmulKernel::FixedIkj, T, static_cast<size_t>(FIXED_GEMM_COMPILED_MIN_SIDE)>(
            output, lhs, rhs, plan);
        return;
    case MatmulKernel::FixedJki:
        dispatch_fixed_side<MatmulKernel::FixedJki, T, static_cast<size_t>(FIXED_GEMM_COMPILED_MIN_SIDE)>(
            output, lhs, rhs, plan);
        return;
    default:
        throw std::logic_error("execute_fixed_gemm(): invalid kernel");
    }
}

template void execute_fixed_gemm<float>(MatmulKernel, float *, float const *, float const *, MatmulPlan const &);
template void execute_fixed_gemm<double>(MatmulKernel, double *, double const *, double const *, MatmulPlan const &);
template void execute_fixed_gemm<Complex<float>>(
    MatmulKernel, Complex<float> *, Complex<float> const *, Complex<float> const *, MatmulPlan const &);

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
