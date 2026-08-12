#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Implement one-level Winograd GEMM.
 *
 * This is Winograd's variant of Strassen's seven-product method. The
 * execution schedule reuses two scratch blocks and accumulates products
 * directly into output quadrants.
 *
 * For C = A B, the operands are divided into four blocks and evaluated with
 * seven products:
 *
 * S1 = A21 + A22, S2 = S1 - A11, S3 = A11 - A21, S4 = A12 - S2
 * T1 = B12 - B11, T2 = B22 - T1, T3 = B22 - B12, T4 = T2 - B21
 * P1 = A11 B11, P2 = A12 B21, P3 = S4 B22, P4 = A22 T4
 * P5 = S1 T1, P6 = S2 T2, P7 = S3 T3
 *
 * C11 = P1 + P2, C12 = P1 + P6 + P5 + P3
 * C21 = P1 + P6 + P7 - P4, C22 = P1 + P6 + P7 + P5
 *
 * Every product writes or accumulates directly into an output quadrant. The
 * only scratch storage holds one left and one right operand block.
 *
 * Inputs are row-major non-transposed views. The output is a row-major view
 * whose leading dimension is at least its column count, and must not overlap
 * either input. Dimensions must be positive and even. The top-level operation
 * requires alpha=1 and beta=0.
 *
 * @see S. Winograd, "On multiplication of 2 x 2 matrices."
 *      https://doi.org/10.1016/0024-3795(71)90009-7
 * @see B. Boyer et al., "Memory efficient scheduling of Strassen-Winograd's
 *      matrix multiplication algorithm." https://arxiv.org/abs/0707.2347
 * @ingroup group_core
 */

#include <solvcon/base.hpp>
#include <solvcon/math/blas_compat.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

namespace winograd
{

template <typename T>
void validate(BlasGemmOperation<T> const & gemm)
{
    if (gemm.rows <= 0 || gemm.columns <= 0 || gemm.inner_size <= 0)
    {
        throw std::invalid_argument("Winograd GEMM dimensions must be positive");
    }
    if (gemm.rows % 2 != 0 || gemm.columns % 2 != 0 || gemm.inner_size % 2 != 0)
    {
        throw std::invalid_argument("Winograd GEMM dimensions must be even");
    }
    if (gemm.lhs.m_transpose != BlasTranspose::None || gemm.rhs.m_transpose != BlasTranspose::None)
    {
        throw std::invalid_argument("Winograd GEMM does not support transposed input views");
    }
    if (gemm.lhs.m_leading_dimension < gemm.inner_size ||
        gemm.rhs.m_leading_dimension < gemm.columns)
    {
        throw std::invalid_argument("Winograd GEMM input leading dimensions are too small");
    }
    if (gemm.output.m_leading_dimension < gemm.columns)
    {
        throw std::invalid_argument("Winograd GEMM output leading dimension is too small");
    }
    if (gemm.alpha != T{1} || gemm.beta != T{0})
    {
        throw std::invalid_argument("Winograd GEMM requires alpha=1 and beta=0");
    }
}

template <typename T>
BlasMatrixView<T> make_subview(BlasMatrixView<T> matrix, ssize_t row, ssize_t column)
{
    return {
        .m_data = matrix.m_data + row * matrix.m_leading_dimension + column,
        .m_leading_dimension = matrix.m_leading_dimension,
        .m_transpose = BlasTranspose::None,
    };
}

template <typename T>
BlasOutputView<T> make_subview(BlasOutputView<T> matrix, ssize_t row, ssize_t column)
{
    return {
        .m_data = matrix.m_data + row * matrix.m_leading_dimension + column,
        .m_leading_dimension = matrix.m_leading_dimension,
    };
}

template <typename T>
void combine_block(
    BlasMatrixView<T> lhs,
    BlasMatrixView<T> rhs,
    T * output,
    ssize_t rows,
    ssize_t columns,
    T rhs_scale)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T const * lhs_row = lhs.m_data + row * lhs.m_leading_dimension;
        T const * rhs_row = rhs.m_data + row * rhs.m_leading_dimension;
        T * output_row = output + row * columns;
        for (ssize_t column = 0; column < columns; ++column)
        {
            output_row[column] = lhs_row[column] + rhs_scale * rhs_row[column];
        }
    }
}

template <typename T, typename Multiply>
void multiply(BlasGemmOperation<T> const & gemm, Multiply const & multiply_product);

/**
 * @brief Evaluate one Winograd decomposition.
 *
 * Step divides one GEMM into quadrants, forms the Winograd input blocks in two
 * reusable scratch regions, and sends seven product descriptors to the
 * caller-provided multiplication callback. The products write or accumulate
 * directly into the four output quadrants. The callback must consume each
 * product synchronously because later products reuse the same scratch.
 *
 * For an `8 x 12 x 16` contraction, Step forms seven `4 x 6 x 8` products
 * and uses scratch for one `4 x 8` lhs block and one `8 x 6` rhs block.
 *
 * @tparam T Element type.
 */
template <typename T>
class Step
{
private:
    template <typename U, typename Multiply>
    friend void multiply(BlasGemmOperation<U> const & gemm, Multiply const & multiply_product);

    explicit Step(BlasGemmOperation<T> const & gemm);

    template <typename Multiply>
    void evaluate(Multiply const & multiply);

    size_t lhs_scratch_size() const
    {
        return static_cast<size_t>(m_block_rows) *
               static_cast<size_t>(m_block_inner_size);
    }
    size_t rhs_scratch_size() const
    {
        return static_cast<size_t>(m_block_inner_size) *
               static_cast<size_t>(m_block_columns);
    }
    T * lhs_scratch() { return m_scratch.get(); }
    T * rhs_scratch() { return m_scratch.get() + lhs_scratch_size(); }
    BlasMatrixView<T> lhs_block();
    BlasMatrixView<T> rhs_block();

    void form_lhs_scratch(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T rhs_scale);
    void form_rhs_scratch(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T rhs_scale);

    template <typename Multiply>
    void multiply_into(
        BlasMatrixView<T> lhs,
        BlasMatrixView<T> rhs,
        BlasOutputView<T> output,
        T alpha,
        T beta,
        Multiply const & multiply);

    void form_output_intermediates();

    ssize_t m_block_rows;
    ssize_t m_block_columns;
    ssize_t m_block_inner_size;
    BlasMatrixView<T> m_a11;
    BlasMatrixView<T> m_a12;
    BlasMatrixView<T> m_a21;
    BlasMatrixView<T> m_a22;
    BlasMatrixView<T> m_b11;
    BlasMatrixView<T> m_b12;
    BlasMatrixView<T> m_b21;
    BlasMatrixView<T> m_b22;
    BlasOutputView<T> m_c11;
    BlasOutputView<T> m_c12;
    BlasOutputView<T> m_c21;
    BlasOutputView<T> m_c22;
    std::unique_ptr<T[]> m_scratch; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
}; /* end class Step */

template <typename T>
Step<T>::Step(BlasGemmOperation<T> const & gemm)
    : m_block_rows(gemm.rows / 2)
    , m_block_columns(gemm.columns / 2)
    , m_block_inner_size(gemm.inner_size / 2)
    , m_a11(make_subview(gemm.lhs, 0, 0))
    , m_a12(make_subview(gemm.lhs, 0, m_block_inner_size))
    , m_a21(make_subview(gemm.lhs, m_block_rows, 0))
    , m_a22(make_subview(gemm.lhs, m_block_rows, m_block_inner_size))
    , m_b11(make_subview(gemm.rhs, 0, 0))
    , m_b12(make_subview(gemm.rhs, 0, m_block_columns))
    , m_b21(make_subview(gemm.rhs, m_block_inner_size, 0))
    , m_b22(make_subview(gemm.rhs, m_block_inner_size, m_block_columns))
    , m_c11(make_subview(gemm.output, 0, 0))
    , m_c12(make_subview(gemm.output, 0, m_block_columns))
    , m_c21(make_subview(gemm.output, m_block_rows, 0))
    , m_c22(make_subview(gemm.output, m_block_rows, m_block_columns))
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    , m_scratch(std::make_unique_for_overwrite<T[]>(
          lhs_scratch_size() + rhs_scratch_size()))
{
}

template <typename T>
template <typename Multiply>
void Step<T>::evaluate(Multiply const & multiply)
{
    // P7 = (A11 - A21)(B22 - B12); initialize C21.
    form_lhs_scratch(m_a11, m_a21, T{-1});
    form_rhs_scratch(m_b22, m_b12, T{-1});
    multiply_into(lhs_block(), rhs_block(), m_c21, T{1}, T{0}, multiply);

    // P5 = (A21 + A22)(B12 - B11); initialize C22.
    form_lhs_scratch(m_a21, m_a22, T{1});
    form_rhs_scratch(m_b12, m_b11, T{-1});
    multiply_into(lhs_block(), rhs_block(), m_c22, T{1}, T{0}, multiply);

    // P6 = (A21 + A22 - A11)(B22 - B12 + B11); initialize C12.
    form_lhs_scratch(lhs_block(), m_a11, T{-1});
    form_rhs_scratch(m_b22, rhs_block(), T{-1});
    multiply_into(lhs_block(), rhs_block(), m_c12, T{1}, T{0}, multiply);

    // P1 initializes C11; U2, U3, U4, and U7 share one output pass.
    multiply_into(m_a11, m_b11, m_c11, T{1}, T{0}, multiply);
    form_output_intermediates();

    // P3 accumulates into U4 to form C12 = U5.
    form_lhs_scratch(m_a12, lhs_block(), T{-1});
    multiply_into(lhs_block(), m_b22, m_c12, T{1}, T{1}, multiply);

    // P4 and P2 accumulate into U3 and P1 to form C21 = U6 and C11 = U1.
    form_rhs_scratch(rhs_block(), m_b21, T{-1});
    multiply_into(m_a22, rhs_block(), m_c21, T{-1}, T{1}, multiply);
    multiply_into(m_a12, m_b21, m_c11, T{1}, T{1}, multiply);
}

template <typename T>
BlasMatrixView<T> Step<T>::lhs_block()
{
    return {
        .m_data = lhs_scratch(),
        .m_leading_dimension = m_block_inner_size,
        .m_transpose = BlasTranspose::None,
    };
}

template <typename T>
BlasMatrixView<T> Step<T>::rhs_block()
{
    return {
        .m_data = rhs_scratch(),
        .m_leading_dimension = m_block_columns,
        .m_transpose = BlasTranspose::None,
    };
}

template <typename T>
void Step<T>::form_lhs_scratch(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T rhs_scale)
{
    combine_block(lhs, rhs, lhs_scratch(), m_block_rows, m_block_inner_size, rhs_scale);
}

template <typename T>
void Step<T>::form_rhs_scratch(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T rhs_scale)
{
    combine_block(lhs, rhs, rhs_scratch(), m_block_inner_size, m_block_columns, rhs_scale);
}

template <typename T>
template <typename Multiply>
void Step<T>::multiply_into(
    BlasMatrixView<T> lhs,
    BlasMatrixView<T> rhs,
    BlasOutputView<T> output,
    T alpha,
    T beta,
    Multiply const & multiply)
{
    BlasGemmOperation<T> const product{
        .rows = m_block_rows,
        .columns = m_block_columns,
        .inner_size = m_block_inner_size,
        .lhs = lhs,
        .rhs = rhs,
        .output = output,
        .alpha = alpha,
        .beta = beta,
    };
    multiply(product);
}

template <typename T>
void Step<T>::form_output_intermediates()
{
    for (ssize_t row = 0; row < m_block_rows; ++row)
    {
        T const * c11_row = m_c11.m_data + row * m_c11.m_leading_dimension;
        T * c12_row = m_c12.m_data + row * m_c12.m_leading_dimension;
        T * c21_row = m_c21.m_data + row * m_c21.m_leading_dimension;
        T * c22_row = m_c22.m_data + row * m_c22.m_leading_dimension;
        for (ssize_t column = 0; column < m_block_columns; ++column)
        {
            T const u2 = c11_row[column] + c12_row[column];
            T const u3 = u2 + c21_row[column];
            T const u4 = u2 + c22_row[column];
            c22_row[column] += u3;
            c21_row[column] = u3;
            c12_row[column] = u4;
        }
    }
}

template <typename T, typename Multiply>
void multiply(BlasGemmOperation<T> const & gemm, Multiply const & multiply_product)
{
    validate(gemm);
    Step<T> step(gemm);
    step.evaluate(multiply_product);
}

} /* end namespace winograd */

template <typename T>
void gemm_winograd(
    ssize_t rows,
    ssize_t columns,
    ssize_t inner_size,
    BlasMatrixView<T> lhs,
    BlasMatrixView<T> rhs,
    BlasOutputView<T> output)
{
    BlasGemmOperation<T> const gemm{
        .rows = rows,
        .columns = columns,
        .inner_size = inner_size,
        .lhs = lhs,
        .rhs = rhs,
        .output = output,
        .alpha = T{1},
        .beta = T{0},
    };
    auto const multiply_product = [](BlasGemmOperation<T> const & product)
    { gemm_blas(product); };
    winograd::multiply(gemm, multiply_product);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
