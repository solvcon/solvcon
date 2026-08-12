/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/matmul.hpp>

#include <solvcon/buffer/ConcreteBuffer.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
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

template <typename T>
static void copy_matrix_to_row_major(
    T const * source,
    T * destination,
    ssize_t rows,
    ssize_t columns,
    ssize_t row_stride,
    ssize_t column_stride)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T const * source_row = source + row * row_stride;
        T * destination_row = destination + row * columns;
        if (column_stride == 1)
        {
            std::copy_n(source_row, static_cast<size_t>(columns), destination_row);
            continue;
        }
        for (ssize_t column = 0; column < columns; ++column)
        {
            destination_row[column] = source_row[column * column_stride];
        }
    }
}

template <typename T>
static size_t checked_matrix_nbytes(ssize_t rows, ssize_t columns)
{
    if (rows < 0 || columns < 0)
    {
        throw std::length_error(
            "execute_streamed_gemm(): scratch size overflows size_t");
    }

    auto const unsigned_rows = static_cast<size_t>(rows);
    auto const unsigned_columns = static_cast<size_t>(columns);
    size_t constexpr maximum = std::numeric_limits<size_t>::max();
    if (unsigned_columns != 0 && unsigned_rows > maximum / unsigned_columns)
    {
        throw std::length_error(
            "execute_streamed_gemm(): scratch size overflows size_t");
    }

    size_t const elements = unsigned_rows * unsigned_columns;
    if (elements > maximum / sizeof(T))
    {
        throw std::length_error(
            "execute_streamed_gemm(): scratch size overflows size_t");
    }
    return elements * sizeof(T);
}

template <typename T>
void execute_streamed_gemm(
    T * output,
    T const * lhs,
    T const * rhs,
    MatmulPlan const & plan,
    std::optional<BlasMatrixView<T>> lhs_view,
    std::optional<BlasMatrixView<T>> rhs_view)
{
    bool const pack_lhs = !lhs_view;
    bool const pack_rhs = !rhs_view;
    if (!pack_lhs && !pack_rhs)
    {
        throw std::logic_error("execute_streamed_gemm(): packing is not required");
    }

    std::shared_ptr<ConcreteBuffer> lhs_scratch;
    std::shared_ptr<ConcreteBuffer> rhs_scratch;
    T * packed_lhs = nullptr;
    T * packed_rhs = nullptr;
    if (pack_lhs)
    {
        size_t const nbytes = checked_matrix_nbytes<T>(plan.rows(), plan.inner_size());
        lhs_scratch = ConcreteBuffer::construct(nbytes);
        packed_lhs = lhs_scratch->data<T>();
        lhs_view = BlasMatrixView<T>{
            .m_data = packed_lhs,
            .m_leading_dimension = plan.inner_size(),
            .m_transpose = BlasTranspose::None,
        };
    }
    if (pack_rhs)
    {
        size_t const nbytes = checked_matrix_nbytes<T>(plan.inner_size(), plan.columns());
        rhs_scratch = ConcreteBuffer::construct(nbytes);
        packed_rhs = rhs_scratch->data<T>();
        rhs_view = BlasMatrixView<T>{
            .m_data = packed_rhs,
            .m_leading_dimension = plan.columns(),
            .m_transpose = BlasTranspose::None,
        };
    }

    BlasGemmOperation<T> operation{
        .rows = plan.rows(),
        .columns = plan.columns(),
        .inner_size = plan.inner_size(),
        .lhs = *lhs_view,
        .rhs = *rhs_view,
        .output = {
            .m_data = output,
            .m_leading_dimension = plan.columns(),
        },
        .alpha = T{1},
        .beta = T{},
    };
    for (MappedOffsetCursor cursor = plan.batch_cursor(); cursor; cursor.advance())
    {
        T const * lhs_matrix = lhs + cursor.offset(MatmulPlan::Operand::Lhs);
        T const * rhs_matrix = rhs + cursor.offset(MatmulPlan::Operand::Rhs);
        if (pack_lhs)
        {
            copy_matrix_to_row_major(
                lhs_matrix,
                packed_lhs,
                plan.rows(),
                plan.inner_size(),
                plan.lhs_row_stride(),
                plan.lhs_inner_stride());
            lhs_matrix = packed_lhs;
        }
        if (pack_rhs)
        {
            copy_matrix_to_row_major(
                rhs_matrix,
                packed_rhs,
                plan.inner_size(),
                plan.columns(),
                plan.rhs_inner_stride(),
                plan.rhs_column_stride());
            rhs_matrix = packed_rhs;
        }

        operation.lhs.m_data = lhs_matrix;
        operation.rhs.m_data = rhs_matrix;
        operation.output.m_data = output + cursor.offset(MatmulPlan::Operand::Output);
        gemm_blas(operation);
    }
}

template void execute_streamed_gemm<float>(
    float *,
    float const *,
    float const *,
    MatmulPlan const &,
    std::optional<BlasMatrixView<float>>,
    std::optional<BlasMatrixView<float>>);
template void execute_streamed_gemm<double>(
    double *,
    double const *,
    double const *,
    MatmulPlan const &,
    std::optional<BlasMatrixView<double>>,
    std::optional<BlasMatrixView<double>>);
template void execute_streamed_gemm<Complex<float>>(
    Complex<float> *,
    Complex<float> const *,
    Complex<float> const *,
    MatmulPlan const &,
    std::optional<BlasMatrixView<Complex<float>>>,
    std::optional<BlasMatrixView<Complex<float>>>);
template void execute_streamed_gemm<Complex<double>>(
    Complex<double> *,
    Complex<double> const *,
    Complex<double> const *,
    MatmulPlan const &,
    std::optional<BlasMatrixView<Complex<double>>>,
    std::optional<BlasMatrixView<Complex<double>>>);

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
