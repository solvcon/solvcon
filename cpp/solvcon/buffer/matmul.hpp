#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/small_vector.hpp>
#include <solvcon/math/math.hpp>

#include <algorithm>
#include <cstddef>
#include <format>
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
        v += m_lhs(i) * m_rhs.data(i);
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
