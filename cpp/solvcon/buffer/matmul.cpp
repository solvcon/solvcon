/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/matmul.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace solvcon
{

namespace detail
{

namespace
{

constexpr std::array<std::string_view, 6> MATMUL_KERNEL_NAMES{
    "naive",
    "blas_dot",
    "blas_gevm",
    "blas_gemv",
    "blas_gemm",
    "winograd",
};

} /* end namespace */

std::string_view matmul_kernel_name(MatmulKernel kernel) noexcept
{
    auto const index = static_cast<size_t>(kernel);
    return index < MATMUL_KERNEL_NAMES.size() ? MATMUL_KERNEL_NAMES[index] : "unknown";
}

std::optional<MatmulKernel> matmul_kernel_from_name(std::string_view name) noexcept
{
    for (size_t index = 0; index < MATMUL_KERNEL_NAMES.size(); ++index)
    {
        if (MATMUL_KERNEL_NAMES[index] == name)
        {
            return static_cast<MatmulKernel>(index);
        }
    }
    return std::nullopt;
}

MatmulLayout::MatmulLayout(shape_type shape, shape_type strides)
    : m_shape(std::move(shape))
    , m_strides(std::move(strides))
{
    if (m_shape.empty() || m_shape.size() != m_strides.size())
    {
        throw std::invalid_argument("shape and strides must have the same nonzero rank");
    }
    constexpr ssize_t MAX_ELEMENTS = std::numeric_limits<ssize_t>::max();
    ssize_t size = 1;
    for (ssize_t const extent : m_shape)
    {
        if (extent < 0 || std::max(extent, ssize_t{1}) > MAX_ELEMENTS / size)
        {
            throw std::invalid_argument("shape exceeds the supported element count");
        }
        size *= std::max(extent, ssize_t{1});
    }
}

MatmulPlan MatmulPlan::make(MatmulLayout const & lhs, MatmulLayout const & rhs)
{
    constexpr ssize_t MAX_ELEMENTS = std::numeric_limits<ssize_t>::max();
    ssize_t const rows = lhs.ndim() == 1 ? 1 : lhs.shape(lhs.ndim() - 2);
    ssize_t const columns = rhs.ndim() == 1 ? 1 : rhs.shape(rhs.ndim() - 1);
    if (columns && rows > MAX_ELEMENTS / columns)
    {
        throw std::invalid_argument("output shape exceeds the supported element count");
    }
    ssize_t output_size = std::max(rows, ssize_t{1}) * std::max(columns, ssize_t{1});
    for (ssize_t offset = 3; offset <= std::max(lhs.ndim(), rhs.ndim()); ++offset)
    {
        ssize_t const lhs_extent = offset <= lhs.ndim() ? lhs.shape(lhs.ndim() - offset) : 1;
        ssize_t const rhs_extent = offset <= rhs.ndim() ? rhs.shape(rhs.ndim() - offset) : 1;
        ssize_t const extent = std::max({lhs_extent, rhs_extent, ssize_t{1}});
        if (extent > MAX_ELEMENTS / output_size)
        {
            throw std::invalid_argument("output shape exceeds the supported element count");
        }
        output_size *= extent;
    }

    return make<MatmulLayout>(lhs, rhs);
}

MatmulPlan::MatmulPlan(
    shape_type output_shape,
    Contraction contraction,
    ContractionStrides strides,
    BatchMappings batch)
    : m_output_shape(std::move(output_shape))
    , m_contraction(contraction)
    , m_strides(strides)
    , m_batch(std::move(batch))
{
}

MatmulPlan::shape_type MatmulPlan::make_output_shape(
    BatchMappings const & batch,
    Contraction const & contraction)
{
    size_t const batch_axis_count = batch.m_domain.rank();
    if (contraction.m_lhs_vector && contraction.m_rhs_vector)
    {
        return shape_type{1};
    }

    size_t output_axis_count = batch_axis_count;
    if (!contraction.m_lhs_vector)
    {
        ++output_axis_count;
    }
    if (!contraction.m_rhs_vector)
    {
        ++output_axis_count;
    }
    shape_type output_shape(output_axis_count);
    std::ranges::copy(batch.m_domain.shape(), output_shape.begin());
    size_t output_axis = batch_axis_count;
    if (!contraction.m_lhs_vector)
    {
        output_shape[output_axis++] = contraction.m_rows;
    }
    if (!contraction.m_rhs_vector)
    {
        output_shape[output_axis] = contraction.m_columns;
    }
    return output_shape;
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
