#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Implement a fixed-depth rectangular Strassen GEMM kernel.
 *
 * For C = A B, each recursion splits the operands into four blocks and
 * evaluates Strassen's seven products:
 *
 * P1 = (A11 + A22)(B11 + B22), P2 = (A21 + A22)B11
 * P3 = A11(B12 - B22), P4 = A22(B21 - B11)
 * P5 = (A11 + A12)B22, P6 = (A21 - A11)(B11 + B12)
 * P7 = (A12 - A22)(B21 + B22)
 *
 * C11 = P1 + P4 - P5 + P7, C12 = P3 + P5
 * C21 = P2 + P4, C22 = P1 - P2 + P3 + P6
 *
 * Inputs are row-major non-transposed views. The output is compact row-major
 * storage and must not overlap either input.
 *
 * Reference:
 * Volker Strassen, "Gaussian elimination is not optimal," Numerische
 * Mathematik 13(4), 354-356 (1969).
 *
 * @see https://doi.org/10.1007/BF02165411
 * @ingroup group_core
 */

#include <solvcon/base.hpp>
#include <solvcon/math/blas_compat.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

namespace strassen
{

template <typename T>
struct Gemm
{
    ssize_t rows;
    ssize_t columns;
    ssize_t inner_size;
    BlasMatrixView<T> lhs;
    BlasMatrixView<T> rhs;
    T * output;
}; /* end struct Gemm */

/**
 * @brief Reuse scratch storage along one Strassen recursion path.
 *
 * Workspace owns a grow-only buffer and provides stack-like allocation.
 * prepare() resets the cursor for one multiplication, while mark() and
 * rewind() bound the storage used by each recursion level. Because each Step
 * evaluates P1-P7 sequentially, sibling products reuse the same child storage
 * instead of reserving space for the full recursion tree.
 *
 * At depth 2, the buffer holds scratch blocks for two simultaneously active
 * Steps, rather than all eight non-leaf Steps or 49 leaf products. Its
 * capacity remains available for later multiplications.
 */
template <typename T>
class Workspace
{
public:
    void prepare(size_t required_size);
    size_t mark() const noexcept { return m_offset; }
    T * allocate(size_t count);
    void rewind(size_t mark) noexcept { m_offset = mark; }
    size_t capacity() const noexcept { return m_capacity; }

private:
    std::unique_ptr<T[]> m_storage; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    size_t m_capacity = 0;
    size_t m_limit = 0;
    size_t m_offset = 0;
}; /* end class Workspace */

template <typename T>
void Workspace<T>::prepare(size_t required_size)
{
    if (required_size > m_capacity)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        m_storage = std::make_unique_for_overwrite<T[]>(required_size);
        m_capacity = required_size;
    }
    m_limit = required_size;
    m_offset = 0;
}

template <typename T>
T * Workspace<T>::allocate(size_t count)
{
    if (count > m_limit - m_offset)
    {
        throw std::runtime_error("Strassen workspace is too small");
    }
    T * block = m_storage.get() + m_offset;
    m_offset += count;
    return block;
}

template <size_t Depth>
size_t workspace_size(ssize_t rows, ssize_t columns, ssize_t inner_size)
{
    if constexpr (Depth == 0)
    {
        return 0;
    }
    else
    {
        rows /= 2;
        columns /= 2;
        inner_size /= 2;
        auto const block_rows = static_cast<size_t>(rows);
        auto const block_columns = static_cast<size_t>(columns);
        auto const block_inner_size = static_cast<size_t>(inner_size);
        size_t const current_size = block_rows * block_inner_size + block_inner_size * block_columns +
                                    block_rows * block_columns;
        return current_size + workspace_size<Depth - 1>(rows, columns, inner_size);
    }
}

template <size_t Depth, typename T>
void validate(Gemm<T> const & gemm)
{
    static_assert(Depth <= 2, "Strassen GEMM supports recursion depths 0, 1, and 2");
    if (gemm.rows <= 0 || gemm.columns <= 0 || gemm.inner_size <= 0)
    {
        throw std::invalid_argument("Strassen GEMM dimensions must be positive");
    }
    size_t const divisor = size_t{1} << Depth;
    if (gemm.rows % divisor != 0 || gemm.columns % divisor != 0 || gemm.inner_size % divisor != 0)
    {
        throw std::invalid_argument("Strassen GEMM dimensions must be divisible by 2^depth");
    }
    if (gemm.lhs.m_transpose != BlasTranspose::None || gemm.rhs.m_transpose != BlasTranspose::None)
    {
        throw std::invalid_argument("Strassen GEMM does not support transposed input views");
    }
    if (gemm.lhs.m_leading_dimension < gemm.inner_size ||
        gemm.rhs.m_leading_dimension < gemm.columns)
    {
        throw std::invalid_argument("Strassen GEMM input leading dimensions are too small");
    }
}

template <typename T>
BlasMatrixView<T> make_subview(BlasMatrixView<T> matrix, ssize_t row, ssize_t column)
{
    return {matrix.m_data + row * matrix.m_leading_dimension + column,
            matrix.m_leading_dimension,
            BlasTranspose::None};
}

template <typename T>
void combine_block(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T * output, ssize_t rows, ssize_t columns, T rhs_scale)
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

template <typename T>
void copy_block(T * output, ssize_t output_stride, T const * input, ssize_t rows, ssize_t columns)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T * output_row = output + row * output_stride;
        T const * input_row = input + row * columns;
        std::copy_n(input_row, columns, output_row);
    }
}

template <typename T>
void add_block(T * output, ssize_t output_stride, T const * input, ssize_t rows, ssize_t columns, T scale)
{
    for (ssize_t row = 0; row < rows; ++row)
    {
        T * output_row = output + row * output_stride;
        T const * input_row = input + row * columns;
        for (ssize_t column = 0; column < columns; ++column)
        {
            output_row[column] += scale * input_row[column];
        }
    }
}

/**
 * @brief Evaluate one level of Strassen matrix multiplication.
 *
 * Step divides one Gemm descriptor into lhs, rhs, and output quadrants. It
 * borrows three Workspace blocks to form the next lhs and rhs operands and to
 * store one product. evaluate() computes P1-P7 sequentially through a
 * caller-provided multiplication callback and assembles the four output
 * quadrants. The callback, rather than Step, decides whether a product recurses
 * or reaches the leaf backend.
 *
 * For an `8 x 12 x 16` contraction, one Step forms seven `4 x 6 x 8`
 * product descriptors while reusing the same three scratch blocks.
 */
template <typename T>
class Step
{
public:
    Step(Gemm<T> const & gemm, Workspace<T> & workspace);

    template <typename Multiply>
    void evaluate(Multiply const & multiply) const;

private:
    struct Product
    {
        ssize_t rows;
        ssize_t columns;
        ssize_t inner_size;
        T * lhs;
        T * rhs;
        T * output;
    }; /* end struct Product */

    BlasMatrixView<T> lhs_block() const { return {m_product.lhs, m_product.inner_size, BlasTranspose::None}; }
    BlasMatrixView<T> rhs_block() const { return {m_product.rhs, m_product.columns, BlasTranspose::None}; }
    void form_lhs(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T rhs_scale) const
    {
        combine_block(lhs, rhs, m_product.lhs, m_product.rows, m_product.inner_size, rhs_scale);
    }
    void form_rhs(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, T rhs_scale) const
    {
        combine_block(lhs, rhs, m_product.rhs, m_product.inner_size, m_product.columns, rhs_scale);
    }

    template <typename Multiply>
    void multiply_product(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, Multiply const & multiply) const;

    void initialize_output(T * destination) const
    {
        copy_block(destination, m_output_stride, m_product.output, m_product.rows, m_product.columns);
    }
    void accumulate_output(T * destination, T scale) const
    {
        add_block(destination, m_output_stride, m_product.output, m_product.rows, m_product.columns, scale);
    }

    Product m_product;
    ssize_t m_output_stride;
    BlasMatrixView<T> m_a11;
    BlasMatrixView<T> m_a12;
    BlasMatrixView<T> m_a21;
    BlasMatrixView<T> m_a22;
    BlasMatrixView<T> m_b11;
    BlasMatrixView<T> m_b12;
    BlasMatrixView<T> m_b21;
    BlasMatrixView<T> m_b22;
    T * m_c11;
    T * m_c12;
    T * m_c21;
    T * m_c22;
}; /* end class Step */

template <typename T>
Step<T>::Step(Gemm<T> const & gemm, Workspace<T> & workspace)
    : m_product{
          gemm.rows / 2,
          gemm.columns / 2,
          gemm.inner_size / 2,
          workspace.allocate(static_cast<size_t>((gemm.rows / 2) * (gemm.inner_size / 2))),
          workspace.allocate(static_cast<size_t>((gemm.inner_size / 2) * (gemm.columns / 2))),
          workspace.allocate(static_cast<size_t>((gemm.rows / 2) * (gemm.columns / 2))),
      }
    , m_output_stride(gemm.columns)
    , m_a11(make_subview(gemm.lhs, 0, 0))
    , m_a12(make_subview(gemm.lhs, 0, m_product.inner_size))
    , m_a21(make_subview(gemm.lhs, m_product.rows, 0))
    , m_a22(make_subview(gemm.lhs, m_product.rows, m_product.inner_size))
    , m_b11(make_subview(gemm.rhs, 0, 0))
    , m_b12(make_subview(gemm.rhs, 0, m_product.columns))
    , m_b21(make_subview(gemm.rhs, m_product.inner_size, 0))
    , m_b22(make_subview(gemm.rhs, m_product.inner_size, m_product.columns))
    , m_c11(gemm.output)
    , m_c12(gemm.output + m_product.columns)
    , m_c21(gemm.output + m_product.rows * m_output_stride)
    , m_c22(m_c21 + m_product.columns)
{
}

template <typename T>
template <typename Multiply>
void Step<T>::multiply_product(BlasMatrixView<T> lhs, BlasMatrixView<T> rhs, Multiply const & multiply) const
{
    Gemm<T> const product{
        m_product.rows,
        m_product.columns,
        m_product.inner_size,
        lhs,
        rhs,
        m_product.output,
    };
    multiply(product);
}

template <typename T>
template <typename Multiply>
void Step<T>::evaluate(Multiply const & multiply) const
{
    // P1 = (A11 + A22)(B11 + B22); initialize C11 and C22.
    form_lhs(m_a11, m_a22, T{1});
    form_rhs(m_b11, m_b22, T{1});
    multiply_product(lhs_block(), rhs_block(), multiply);
    initialize_output(m_c11);
    initialize_output(m_c22);

    // P2 = (A21 + A22)B11; initialize C21 and subtract from C22.
    form_lhs(m_a21, m_a22, T{1});
    multiply_product(lhs_block(), m_b11, multiply);
    initialize_output(m_c21);
    accumulate_output(m_c22, T{-1});

    // P3 = A11(B12 - B22); initialize C12 and add to C22.
    form_rhs(m_b12, m_b22, T{-1});
    multiply_product(m_a11, rhs_block(), multiply);
    initialize_output(m_c12);
    accumulate_output(m_c22, T{1});

    // P4 = A22(B21 - B11); add to C11 and C21.
    form_rhs(m_b21, m_b11, T{-1});
    multiply_product(m_a22, rhs_block(), multiply);
    accumulate_output(m_c11, T{1});
    accumulate_output(m_c21, T{1});

    // P5 = (A11 + A12)B22; subtract from C11 and add to C12.
    form_lhs(m_a11, m_a12, T{1});
    multiply_product(lhs_block(), m_b22, multiply);
    accumulate_output(m_c11, T{-1});
    accumulate_output(m_c12, T{1});

    // P6 = (A21 - A11)(B11 + B12); add to C22.
    form_lhs(m_a21, m_a11, T{-1});
    form_rhs(m_b11, m_b12, T{1});
    multiply_product(lhs_block(), rhs_block(), multiply);
    accumulate_output(m_c22, T{1});

    // P7 = (A12 - A22)(B21 + B22); add to C11.
    form_lhs(m_a12, m_a22, T{-1});
    form_rhs(m_b21, m_b22, T{1});
    multiply_product(lhs_block(), rhs_block(), multiply);
    accumulate_output(m_c11, T{1});
}

template <typename T, typename Leaf>
class Kernel
{
public:
    Kernel(Workspace<T> & workspace, Leaf const & leaf)
        : m_workspace(workspace)
        , m_leaf(leaf)
    {
    }

    template <size_t Depth>
    void multiply(Gemm<T> const & gemm);

private:
    template <size_t Depth>
    void recurse(Gemm<T> const & gemm);

    Workspace<T> & m_workspace;
    Leaf const & m_leaf;
}; /* end class Kernel */

template <typename T, typename Leaf>
template <size_t Depth>
void Kernel<T, Leaf>::multiply(Gemm<T> const & gemm)
{
    validate<Depth>(gemm);
    m_workspace.prepare(workspace_size<Depth>(gemm.rows, gemm.columns, gemm.inner_size));
    recurse<Depth>(gemm);
}

template <typename T, typename Leaf>
template <size_t Depth>
void Kernel<T, Leaf>::recurse(Gemm<T> const & gemm)
{
    if constexpr (Depth == 0)
    {
        m_leaf(gemm);
    }
    else
    {
        size_t const mark = m_workspace.mark();
        Step<T> const step(gemm, m_workspace);
        auto const recurse_product = [this](Gemm<T> const & product)
        { this->template recurse<Depth - 1>(product); };
        step.evaluate(recurse_product);
        m_workspace.rewind(mark);
    }
}

template <size_t Depth, typename T, typename Leaf>
void multiply(Gemm<T> const & gemm, Workspace<T> & workspace, Leaf const & leaf)
{
    Kernel<T, Leaf> kernel(workspace, leaf);
    kernel.template multiply<Depth>(gemm);
}

} /* end namespace strassen */

template <size_t Depth, typename T>
void gemm_strassen(strassen::Gemm<T> const & gemm, strassen::Workspace<T> & workspace)
{
    auto const leaf = [](strassen::Gemm<T> const & leaf_gemm)
    {
        gemm_blas(
            leaf_gemm.rows,
            leaf_gemm.columns,
            leaf_gemm.inner_size,
            leaf_gemm.lhs,
            leaf_gemm.rhs,
            leaf_gemm.output);
    };
    strassen::multiply<Depth>(gemm, workspace, leaf);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
