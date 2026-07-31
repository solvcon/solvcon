#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/loop.hpp>
#include <solvcon/buffer/small_vector.hpp>
#include <solvcon/math/math.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace solvcon
{

namespace detail
{

template <typename T>
inline constexpr bool can_matmul_blas_v = std::is_same_v<T, float> ||
                                          std::is_same_v<T, double> ||
                                          std::is_same_v<T, Complex<float>> ||
                                          std::is_same_v<T, Complex<double>>;

/**
 * @brief Describe matmul operands as an execution-independent contraction.
 *
 * MatmulPlan interprets trailing axes as vector or matrix roles and leading
 * axes as the batch domain. It validates the contracted dimension, derives
 * the result shape, and records the signed-stride mappings used to locate
 * operands in the result batch domain. Broadcast batch axes are represented
 * by zero strides, so an executor can advance operand offsets without
 * reinterpreting ranks or broadcasting rules.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the plan records batch shape `(2,5)`, M=3,
 * N=6, K=4, output shape `(2,5,3,6)`, and zero-stride batch mappings for
 * the broadcast axes. It does not allocate the output or evaluate the ten
 * matrix pairs.
 *
 * @note This implementation plans matrix-matrix operands with equal ranks
 * and equal leading batch shapes. It records the batch domain and mappings,
 * M, N, K, the output shape, and signed matrix strides. Vector roles and
 * broadcast batch axes are follow-up work.
 */
class MatmulPlan
{
public:
    using shape_type = small_vector<ssize_t>;

    template <typename Array>
    static MatmulPlan make(Array const & lhs, Array const & rhs);

    shape_type const & output_shape() const noexcept { return m_output_shape; }
    ssize_t rows() const noexcept { return m_contraction.m_rows; }
    ssize_t columns() const noexcept { return m_contraction.m_columns; }
    ssize_t inner_size() const noexcept { return m_contraction.m_inner_size; }
    ssize_t lhs_row_stride() const noexcept { return m_strides.m_lhs_row_stride; }
    ssize_t lhs_inner_stride() const noexcept { return m_strides.m_lhs_inner_stride; }
    ssize_t rhs_inner_stride() const noexcept { return m_strides.m_rhs_inner_stride; }
    ssize_t rhs_column_stride() const noexcept { return m_strides.m_rhs_column_stride; }

    bool has_batch_axes() const noexcept { return m_batch.m_domain.rank() != 0; }
    MappedOffsetCursor batch_cursor() const & { return MappedOffsetCursor(m_batch.m_domain, m_batch.m_mappings); }
    MappedOffsetCursor batch_cursor() const && = delete;

private:
    using batch_stride_type = OperandMapping::stride_type;
    using mapping_type = MappedOffsetCursor::mapping_type;

    struct Contraction
    {
        ssize_t m_rows;
        ssize_t m_columns;
        ssize_t m_inner_size;
    }; /* end struct Contraction */

    struct MatrixStrides
    {
        ssize_t m_lhs_row_stride;
        ssize_t m_lhs_inner_stride;
        ssize_t m_rhs_inner_stride;
        ssize_t m_rhs_column_stride;
    }; /* end struct MatrixStrides */

    struct BatchMappings
    {
        LoopDomain m_domain;
        mapping_type m_mappings;
    }; /* end struct BatchMappings */

    MatmulPlan(shape_type output_shape, Contraction contraction, MatrixStrides strides, BatchMappings batch);

    template <typename Array>
    static Contraction make_contraction(Array const & lhs, Array const & rhs);

    template <typename Array>
    static BatchMappings make_batch_mappings(Array const & lhs, Array const & rhs, ssize_t output_matrix_size);

    static shape_type make_output_shape(BatchMappings const & batch, Contraction const & contraction);

    template <typename Array>
    static std::string shape_string(Array const & array);

    shape_type m_output_shape;
    Contraction m_contraction;
    MatrixStrides m_strides;
    BatchMappings m_batch;
}; /* end class MatmulPlan */

/**
 * @brief Execute a MatmulPlan with a layout-appropriate contraction route.
 *
 * MatmulExecutor maps vector and matrix roles to DOT, GEMV, or GEMM, then
 * selects generic, tiled, direct BLAS, or pack-once BLAS execution. It owns
 * BLAS eligibility, packing reuse, and size thresholds; these decisions do
 * not change the plan. The constructor only binds a plan and caller-owned
 * arrays; execute() evaluates the plan.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the executor visits ten batch offsets and
 * evaluates one `(3,4) @ (4,6)` contraction at each offset. The results are
 * written into the allocated `(2,5,3,6)` output.
 *
 * @note This implementation provides the generic signed-stride
 * matrix-matrix route for equal batch shapes. Broadcast, optimized, and
 * vector routes are follow-up work.
 */
template <typename Array>
class MatmulExecutor
{
public:
    MatmulExecutor(MatmulPlan const & plan, Array & output, Array const & lhs, Array const & rhs);

    void execute();

private:
    using value_type = typename Array::value_type;

    enum class MappingSlot : std::uint8_t
    {
        Output,
        Lhs,
        Rhs,
    };

    void execute_generic(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base);

    MatmulPlan const & m_plan;
    value_type * m_output_data;
    value_type const * m_lhs_data;
    value_type const * m_rhs_data;
}; /* end class MatmulExecutor */

inline MatmulPlan::MatmulPlan(
    shape_type output_shape,
    Contraction contraction,
    MatrixStrides strides,
    BatchMappings batch)
    : m_output_shape(std::move(output_shape))
    , m_contraction(contraction)
    , m_strides(strides)
    , m_batch(std::move(batch))
{
}

template <typename Array>
MatmulPlan MatmulPlan::make(Array const & lhs, Array const & rhs)
{
    Contraction const contraction = make_contraction(lhs, rhs);
    size_t const lhs_matrix_axis = lhs.ndim() - 2;
    size_t const rhs_matrix_axis = rhs.ndim() - 2;
    MatrixStrides const strides{
        .m_lhs_row_stride = lhs.stride(lhs_matrix_axis),
        .m_lhs_inner_stride = lhs.stride(lhs_matrix_axis + 1),
        .m_rhs_inner_stride = rhs.stride(rhs_matrix_axis),
        .m_rhs_column_stride = rhs.stride(rhs_matrix_axis + 1),
    };
    BatchMappings batch = make_batch_mappings(lhs, rhs, contraction.m_rows * contraction.m_columns);
    shape_type output_shape = make_output_shape(batch, contraction);
    return MatmulPlan{
        std::move(output_shape),
        contraction,
        strides,
        std::move(batch),
    };
}

template <typename Array>
MatmulPlan::Contraction MatmulPlan::make_contraction(Array const & lhs, Array const & rhs)
{
    if (lhs.ndim() < 2 || rhs.ndim() < 2)
    {
        throw std::invalid_argument(
            "planned matrix-matrix matmul requires rank-2 or greater operands");
    }

    size_t const lhs_matrix_axis = lhs.ndim() - 2;
    size_t const rhs_matrix_axis = rhs.ndim() - 2;
    if (lhs.shape(lhs_matrix_axis + 1) != rhs.shape(rhs_matrix_axis))
    {
        throw std::invalid_argument(
            std::format("SimpleArray::matmul_planned(): shape mismatch: "
                        "this={} other={}",
                        shape_string(lhs),
                        shape_string(rhs)));
    }
    return Contraction{
        .m_rows = lhs.shape(lhs_matrix_axis),
        .m_columns = rhs.shape(rhs_matrix_axis + 1),
        .m_inner_size = lhs.shape(lhs_matrix_axis + 1),
    };
}

template <typename Array>
MatmulPlan::BatchMappings MatmulPlan::make_batch_mappings(
    Array const & lhs,
    Array const & rhs,
    ssize_t output_matrix_size)
{
    if (lhs.ndim() != rhs.ndim())
    {
        throw std::invalid_argument(
            std::format("SimpleArray::matmul_planned(): batch shape "
                        "mismatch: this={} other={}",
                        shape_string(lhs),
                        shape_string(rhs)));
    }

    size_t const batch_rank = lhs.ndim() - 2;
    shape_type batch_shape(lhs.shape().begin(), lhs.shape().begin() + batch_rank);
    if (!std::equal(batch_shape.begin(), batch_shape.end(), rhs.shape().begin()))
    {
        throw std::invalid_argument(
            std::format("SimpleArray::matmul_planned(): batch shape "
                        "mismatch: this={} other={}",
                        shape_string(lhs),
                        shape_string(rhs)));
    }

    LoopDomain domain{std::move(batch_shape)};
    batch_stride_type lhs_batch_strides(lhs.stride().begin(), lhs.stride().begin() + batch_rank);
    batch_stride_type rhs_batch_strides(rhs.stride().begin(), rhs.stride().begin() + batch_rank);
    mapping_type mappings{
        OperandMapping::contiguous_blocks(domain, output_matrix_size),
        OperandMapping(std::move(lhs_batch_strides)),
        OperandMapping(std::move(rhs_batch_strides)),
    };
    return BatchMappings{
        .m_domain = std::move(domain),
        .m_mappings = std::move(mappings),
    };
}

inline MatmulPlan::shape_type MatmulPlan::make_output_shape(
    BatchMappings const & batch,
    Contraction const & contraction)
{
    size_t const batch_rank = batch.m_domain.rank();
    shape_type output_shape(batch_rank + 2);
    std::ranges::copy(batch.m_domain.shape(), output_shape.begin());
    output_shape[batch_rank] = contraction.m_rows;
    output_shape[batch_rank + 1] = contraction.m_columns;
    return output_shape;
}

template <typename Array>
std::string MatmulPlan::shape_string(Array const & array)
{
    std::string result = "(";
    for (size_t axis = 0; axis < array.ndim(); ++axis)
    {
        if (axis > 0)
        {
            result += ",";
        }
        result += std::to_string(array.shape(axis));
    }
    result += ")";
    return result;
}

template <typename Array>
MatmulExecutor<Array>::MatmulExecutor(MatmulPlan const & plan, Array & output, Array const & lhs, Array const & rhs)
    : m_plan(plan)
    , m_output_data(output.logical_data())
    , m_lhs_data(lhs.logical_data())
    , m_rhs_data(rhs.logical_data())
{
}

template <typename Array>
void MatmulExecutor<Array>::execute()
{
    if (!m_plan.has_batch_axes())
    {
        execute_generic(0, 0, 0);
        return;
    }

    for (MappedOffsetCursor cursor = m_plan.batch_cursor(); cursor; cursor.advance())
    {
        execute_generic(
            cursor.offset(MappingSlot::Output),
            cursor.offset(MappingSlot::Lhs),
            cursor.offset(MappingSlot::Rhs));
    }
}

template <typename Array>
void MatmulExecutor<Array>::execute_generic(ssize_t output_base, ssize_t lhs_base, ssize_t rhs_base)
{
    for (ssize_t row = 0; row < m_plan.rows(); ++row)
    {
        ssize_t const lhs_row_base = lhs_base + row * m_plan.lhs_row_stride();
        ssize_t const output_row_base = output_base + row * m_plan.columns();
        for (ssize_t column = 0; column < m_plan.columns(); ++column)
        {
            value_type total{};
            ssize_t lhs_offset = lhs_row_base;
            ssize_t rhs_offset = rhs_base + column * m_plan.rhs_column_stride();
            for (ssize_t inner = 0; inner < m_plan.inner_size(); ++inner)
            {
                total += m_lhs_data[lhs_offset] * m_rhs_data[rhs_offset];
                lhs_offset += m_plan.lhs_inner_stride();
                rhs_offset += m_plan.rhs_inner_stride();
            }
            m_output_data[output_row_base + column] = total;
        }
    }
}

template <typename A, typename T>
class SimpleArrayMatmulHelper
{

public:

    using value_type = T;
    using shape_type = typename A::shape_type;

    SimpleArrayMatmulHelper() = delete;
    SimpleArrayMatmulHelper(A const & lhs, A const & rhs);
    SimpleArrayMatmulHelper(A const & lhs,
                            A const & rhs,
                            ssize_t tile_x,
                            ssize_t tile_y,
                            ssize_t tile_z);
    ~SimpleArrayMatmulHelper() = default;

    SimpleArrayMatmulHelper(SimpleArrayMatmulHelper const &) = delete;
    SimpleArrayMatmulHelper(SimpleArrayMatmulHelper &&) = delete;
    SimpleArrayMatmulHelper & operator=(SimpleArrayMatmulHelper const &) = delete;
    SimpleArrayMatmulHelper & operator=(SimpleArrayMatmulHelper &&) = delete;

    A matmul();
    A matmul_fast();
    A matmul_blas();

private:

    static std::string shape_str(A const & arr);
    void check_dims() const;
    void check_inner(size_t lhs_idx, size_t rhs_idx) const;
    void check_tiles() const;
    A matmul_vec_vec();
    A matmul_vec_vec_blas();
    A matmul_vec_mat();
    A matmul_vec_mat_blas();
    A matmul_mat_vec();
    A matmul_mat_vec_blas();
    A matmul_mat_mat();
    A matmul_mat_mat_blas();
    A pack_rhs(ssize_t n, ssize_t k);
    void accumulate_tile(A const & packed_rhs,
                         ssize_t row_begin,
                         ssize_t row_end,
                         ssize_t col_begin,
                         ssize_t col_end,
                         ssize_t inner_begin,
                         ssize_t inner_end);
    A matmul_mat_mat_tiled();

    A const & m_lhs;
    A const & m_rhs;
    A m_result;
    ssize_t m_tile_x;
    ssize_t m_tile_y;
    ssize_t m_tile_z;

}; /* end class SimpleArrayMatmulHelper */

template <typename A, typename T>
SimpleArrayMatmulHelper<A, T>::SimpleArrayMatmulHelper(A const & lhs, A const & rhs)
    : SimpleArrayMatmulHelper(lhs, rhs, 0, 0, 0)
{
}

template <typename A, typename T>
SimpleArrayMatmulHelper<A, T>::SimpleArrayMatmulHelper(A const & lhs,
                                                       A const & rhs,
                                                       ssize_t tile_x,
                                                       ssize_t tile_y,
                                                       ssize_t tile_z)
    : m_lhs(lhs)
    , m_rhs(rhs)
    , m_tile_x(tile_x)
    , m_tile_y(tile_y)
    , m_tile_z(tile_z)
{
    check_dims();

    size_t const lhs_ndim = m_lhs.ndim();
    size_t const rhs_ndim = m_rhs.ndim();

    if (lhs_ndim == 1 && rhs_ndim == 1)
    {
        check_inner(0, 0);
        m_result = A(1);
        return;
    }

    if (lhs_ndim == 1)
    {
        check_inner(0, 0);
        m_result = A(m_rhs.shape(1));
        return;
    }

    if (rhs_ndim == 1)
    {
        check_inner(1, 0);
        m_result = A(m_lhs.shape(0));
        return;
    }

    check_inner(1, 0);
    shape_type const result_shape{m_lhs.shape(0), m_rhs.shape(1)};
    m_result = A(result_shape);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul()
{
    if (m_lhs.ndim() == 1 && m_rhs.ndim() == 1)
    {
        return matmul_vec_vec();
    }
    if (m_lhs.ndim() == 1)
    {
        return matmul_vec_mat();
    }
    if (m_rhs.ndim() == 1)
    {
        return matmul_mat_vec();
    }

    return matmul_mat_mat();
}

/**
 * Perform fast matrix multiplication for SimpleArrays.
 * This implementation currently uses tiling for 2D x 2D matrix multiplication.
 * Future optimizations may add other techniques such as SIMD kernels.
 */
template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_fast()
{
    check_tiles();

    if (m_lhs.ndim() == 1 && m_rhs.ndim() == 1)
    {
        return matmul_vec_vec();
    }
    if (m_lhs.ndim() == 1)
    {
        return matmul_vec_mat();
    }
    if (m_rhs.ndim() == 1)
    {
        return matmul_mat_vec();
    }

    return matmul_mat_mat_tiled();
}

/**
 * Perform matrix multiplication using vendor BLAS when available.
 */
template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_blas()
{
    if (m_lhs.ndim() == 1 && m_rhs.ndim() == 1)
    {
        return matmul_vec_vec_blas();
    }
    if (m_lhs.ndim() == 1)
    {
        return matmul_vec_mat_blas();
    }
    if (m_rhs.ndim() == 1)
    {
        return matmul_mat_vec_blas();
    }

    return matmul_mat_mat_blas();
}

/**
 * Format shape for matrix multiplication diagnostics.
 */
template <typename A, typename T>
std::string SimpleArrayMatmulHelper<A, T>::shape_str(A const & arr)
{
    if (arr.ndim() == 0)
    {
        return "()";
    }

    std::string result = "(";
    for (size_t i = 0; i < arr.ndim(); ++i)
    {
        if (i > 0)
        {
            result += ",";
        }
        result += std::to_string(arr.shape(i));
    }
    result += ")";
    return result;
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::check_dims() const
{
    bool const lhs_is_supported = m_lhs.ndim() == 1 || m_lhs.ndim() == 2;
    bool const rhs_is_supported = m_rhs.ndim() == 1 || m_rhs.ndim() == 2;
    if (lhs_is_supported && rhs_is_supported)
    {
        return;
    }

    std::string const err = std::format("SimpleArray::matmul(): unsupported dimensions: "
                                        "this={} other={}. SimpleArray must be 1D or 2D.",
                                        shape_str(m_lhs),
                                        shape_str(m_rhs));
    throw std::out_of_range(err);
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::check_inner(size_t lhs_idx, size_t rhs_idx) const
{
    if (m_lhs.shape(lhs_idx) == m_rhs.shape(rhs_idx))
    {
        return;
    }

    throw std::out_of_range(
        std::format("SimpleArray::matmul(): shape mismatch: this={} other={}",
                    shape_str(m_lhs),
                    shape_str(m_rhs)));
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::check_tiles() const
{
    if (m_tile_x > 0 && m_tile_y > 0 && m_tile_z > 0)
    {
        return;
    }

    throw std::out_of_range(
        std::format("SimpleArray::matmul_fast(): tile sizes must be positive: "
                    "tile_x={} tile_y={} tile_z={}",
                    m_tile_x,
                    m_tile_y,
                    m_tile_z));
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_vec()
{
    ssize_t const k = m_lhs.shape(0);
    value_type v = 0;
    for (ssize_t i = 0; i < k; ++i)
    {
        v += m_lhs(i) * m_rhs(i);
    }
    m_result.data(0) = v;
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_vec_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_vec_vec();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const k = m_lhs.shape(0);
        m_result.data(0) = dot_blas(k, m_lhs.data(), m_rhs.data());
        return std::move(m_result);
    }
    else
    {
        return matmul_vec_vec();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_mat()
{
    ssize_t const n = m_result.shape(0);
    ssize_t const k = m_lhs.shape(0);
    for (ssize_t j = 0; j < n; ++j)
    {
        value_type v = 0;
        for (ssize_t l = 0; l < k; ++l)
        {
            v += m_lhs(l) * m_rhs(l, j);
        }
        m_result.data(j) = v;
    }
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_vec_mat_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_vec_mat();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const k = m_rhs.shape(0);
        ssize_t const n = m_rhs.shape(1);
        bool const transpose_matrix = true;
        gemv_blas(k,
                  n,
                  m_rhs.data(),
                  m_lhs.data(),
                  m_result.data(),
                  transpose_matrix);
        return std::move(m_result);
    }
    else
    {
        return matmul_vec_mat();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_vec()
{
    ssize_t const m = m_result.shape(0);
    ssize_t const k = m_lhs.shape(1);
    for (ssize_t i = 0; i < m; ++i)
    {
        value_type v = 0;
        for (ssize_t l = 0; l < k; ++l)
        {
            v += m_lhs(i, l) * m_rhs(l);
        }
        m_result.data(i) = v;
    }
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_vec_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_mat_vec();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const m = m_lhs.shape(0);
        ssize_t const k = m_lhs.shape(1);
        bool const transpose_matrix = false;
        gemv_blas(m,
                  k,
                  m_lhs.data(),
                  m_rhs.data(),
                  m_result.data(),
                  transpose_matrix);
        return std::move(m_result);
    }
    else
    {
        return matmul_mat_vec();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_mat()
{
    ssize_t const m = m_result.shape(0);
    ssize_t const n = m_result.shape(1);
    ssize_t const k = m_lhs.shape(1);
    for (ssize_t i = 0; i < m; ++i)
    {
        for (ssize_t j = 0; j < n; ++j)
        {
            value_type v = 0;
            for (ssize_t l = 0; l < k; ++l)
            {
                v += m_lhs(i, l) * m_rhs(l, j);
            }
            m_result(i, j) = v;
        }
    }
    return std::move(m_result);
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_mat_blas()
{
    if (!m_lhs.is_c_contiguous() || !m_rhs.is_c_contiguous())
    {
        return matmul_mat_mat();
    }

    if constexpr (can_matmul_blas_v<value_type>)
    {
        ssize_t const m = m_result.shape(0);
        ssize_t const n = m_result.shape(1);
        ssize_t const k = m_lhs.shape(1);
        gemm_blas(m, n, k, m_lhs.data(), m_rhs.data(), m_result.data());
        return std::move(m_result);
    }
    else
    {
        return matmul_mat_mat();
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::pack_rhs(ssize_t n, ssize_t k)
{
    shape_type const packing_shape{n, k};
    A packing(packing_shape);
    for (ssize_t i = 0; i < n; ++i)
    {
        for (ssize_t j = 0; j < k; ++j)
        {
            packing(i, j) = m_rhs(j, i);
        }
    }
    return packing;
}

template <typename A, typename T>
void SimpleArrayMatmulHelper<A, T>::accumulate_tile(A const & packed_rhs,
                                                    ssize_t row_begin,
                                                    ssize_t row_end,
                                                    ssize_t col_begin,
                                                    ssize_t col_end,
                                                    ssize_t inner_begin,
                                                    ssize_t inner_end)
{
    for (ssize_t i = row_begin; i < row_end; ++i)
    {
        for (ssize_t j = col_begin; j < col_end; ++j)
        {
            value_type v = m_result(i, j);
            for (ssize_t l = inner_begin; l < inner_end; ++l)
            {
                v += m_lhs(i, l) * packed_rhs(j, l);
            }
            m_result(i, j) = v;
        }
    }
}

template <typename A, typename T>
A SimpleArrayMatmulHelper<A, T>::matmul_mat_mat_tiled()
{
    ssize_t const m = m_result.shape(0);
    ssize_t const n = m_result.shape(1);
    ssize_t const k = m_lhs.shape(1);
    A packed_rhs = pack_rhs(n, k);
    for (size_t i = 0; i < m_result.size(); ++i)
    {
        m_result.data(i) = value_type{0};
    }
    for (ssize_t row = 0; row < m; row += m_tile_x)
    {
        ssize_t const row_end = std::min(row + m_tile_x, m);
        for (ssize_t col = 0; col < n; col += m_tile_y)
        {
            ssize_t const col_end = std::min(col + m_tile_y, n);
            for (ssize_t inner = 0; inner < k; inner += m_tile_z)
            {
                ssize_t const inner_end = std::min(inner + m_tile_z, k);
                accumulate_tile(packed_rhs, row, row_end, col, col_end, inner, inner_end);
            }
        }
    }
    return std::move(m_result);
}

} /* end namespace detail */

} /* end namespace solvcon */
