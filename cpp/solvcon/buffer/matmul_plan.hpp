#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Validate and plan SimpleArray matrix multiplication.
 *
 * @ingroup group_core
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/loop.hpp>
#include <solvcon/buffer/small_vector.hpp>

#include <algorithm>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace solvcon
{

namespace detail
{

/**
 * @brief Describe matmul operands as an execution-independent contraction.
 *
 * MatmulPlan interprets trailing axes as vector or matrix roles and leading
 * axes as the batch domain. It validates the contracted dimension, derives
 * the result shape, and records the signed-stride mappings used to locate
 * operands in the result batch domain. Broadcast batch axes are represented
 * by zero strides, so an executor can advance operand offsets without
 * reinterpreting operand dimensions or broadcasting rules.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the plan records batch shape `(2,5)`, M=3,
 * N=6, K=4, output shape `(2,5,3,6)`, and zero-stride batch mappings for
 * the broadcast axes.
 *
 * For `(K,) @ (B,K,N)`, the plan keeps a synthetic M=1 for execution but
 * omits that axis from the `(B,N)` output. Matrix-vector products similarly
 * omit synthetic N=1, while vector-vector products retain SimpleArray's
 * existing `(1,)` dot-result convention.
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

    bool lhs_is_vector() const noexcept { return m_contraction.m_lhs_vector; }
    bool rhs_is_vector() const noexcept { return m_contraction.m_rhs_vector; }
    bool lhs_is_broadcast() const noexcept { return m_batch.m_lhs_is_broadcast; }
    bool rhs_is_broadcast() const noexcept { return m_batch.m_rhs_is_broadcast; }
    bool lhs_has_zero_batch_stride() const noexcept { return m_batch.m_lhs_has_zero_stride; }
    bool rhs_has_zero_batch_stride() const noexcept { return m_batch.m_rhs_has_zero_stride; }
    bool has_batch_axes() const noexcept { return m_batch.m_domain.rank() != 0; }
    size_t batch_size() const noexcept { return m_batch.m_domain.size(); }
    MappedOffsetCursor batch_cursor() const & { return MappedOffsetCursor(m_batch.m_domain, m_batch.m_mappings); }
    MappedOffsetCursor batch_cursor() const && = delete;

private:
    using batch_stride_type = OperandMapping::stride_type;
    using mapping_type = MappedOffsetCursor::mapping_type;

    struct Contraction
    {
        bool m_lhs_vector;
        bool m_rhs_vector;
        ssize_t m_rows;
        ssize_t m_columns;
        ssize_t m_inner_size;
    }; /* end struct Contraction */

    struct ContractionStrides
    {
        ssize_t m_lhs_row_stride;
        ssize_t m_lhs_inner_stride;
        ssize_t m_rhs_inner_stride;
        ssize_t m_rhs_column_stride;
    }; /* end struct ContractionStrides */

    struct BatchOperand
    {
        OperandMapping m_mapping;
        bool m_is_broadcast = false;
        bool m_has_zero_stride = false;
    }; /* end struct BatchOperand */

    struct BatchMappings
    {
        LoopDomain m_domain;
        mapping_type m_mappings;
        bool m_lhs_is_broadcast;
        bool m_rhs_is_broadcast;
        bool m_lhs_has_zero_stride;
        bool m_rhs_has_zero_stride;
    }; /* end struct BatchMappings */

    MatmulPlan(shape_type output_shape, Contraction contraction, ContractionStrides strides, BatchMappings batch);

    template <typename Array>
    static Contraction make_contraction(Array const & lhs, Array const & rhs);

    template <typename Array>
    static ContractionStrides make_contraction_strides(
        Array const & lhs,
        Array const & rhs,
        Contraction const & contraction);

    template <typename Array>
    static BatchMappings make_batch_mappings(Array const & lhs, Array const & rhs, ssize_t output_block_size);

    template <typename Array>
    static shape_type make_batch_shape(Array const & lhs, Array const & rhs);

    template <typename Array>
    static BatchOperand make_batch_operand(Array const & operand, LoopDomain const & domain);

    static shape_type make_output_shape(BatchMappings const & batch, Contraction const & contraction);

    template <typename Array>
    static std::string shape_string(Array const & array);

    shape_type m_output_shape;
    Contraction m_contraction;
    ContractionStrides m_strides;
    BatchMappings m_batch;
}; /* end class MatmulPlan */

template <typename Array>
MatmulPlan MatmulPlan::make(Array const & lhs, Array const & rhs)
{
    Contraction const contraction = make_contraction(lhs, rhs);
    ContractionStrides const strides = make_contraction_strides(lhs, rhs, contraction);
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
    if (lhs.ndim() == 0 || rhs.ndim() == 0)
    {
        throw std::invalid_argument(
            "matmul requires non-scalar operands");
    }

    bool const lhs_vector = lhs.ndim() == 1;
    bool const rhs_vector = rhs.ndim() == 1;
    size_t const lhs_inner_axis = lhs.ndim() - 1;
    size_t rhs_inner_axis = rhs.ndim() - 1;
    if (!rhs_vector)
    {
        --rhs_inner_axis;
    }
    if (lhs.shape(lhs_inner_axis) != rhs.shape(rhs_inner_axis))
    {
        throw std::invalid_argument(
            std::format("SimpleArray::matmul(): shape mismatch: "
                        "this={} other={}",
                        shape_string(lhs),
                        shape_string(rhs)));
    }

    ssize_t rows = 1;
    if (!lhs_vector)
    {
        rows = lhs.shape(lhs.ndim() - 2);
    }
    ssize_t columns = 1;
    if (!rhs_vector)
    {
        columns = rhs.shape(rhs.ndim() - 1);
    }
    return Contraction{
        .m_lhs_vector = lhs_vector,
        .m_rhs_vector = rhs_vector,
        .m_rows = rows,
        .m_columns = columns,
        .m_inner_size = lhs.shape(lhs_inner_axis),
    };
}

template <typename Array>
MatmulPlan::ContractionStrides MatmulPlan::make_contraction_strides(
    Array const & lhs,
    Array const & rhs,
    Contraction const & contraction)
{
    ssize_t lhs_row_stride = 0;
    if (!contraction.m_lhs_vector)
    {
        lhs_row_stride = lhs.stride(lhs.ndim() - 2);
    }

    size_t rhs_inner_axis = rhs.ndim() - 1;
    ssize_t rhs_column_stride = 0;
    if (!contraction.m_rhs_vector)
    {
        --rhs_inner_axis;
        rhs_column_stride = rhs.stride(rhs.ndim() - 1);
    }
    return ContractionStrides{
        .m_lhs_row_stride = lhs_row_stride,
        .m_lhs_inner_stride = lhs.stride(lhs.ndim() - 1),
        .m_rhs_inner_stride = rhs.stride(rhs_inner_axis),
        .m_rhs_column_stride = rhs_column_stride,
    };
}

template <typename Array>
MatmulPlan::BatchMappings MatmulPlan::make_batch_mappings(
    Array const & lhs,
    Array const & rhs,
    ssize_t output_block_size)
{
    LoopDomain domain{make_batch_shape(lhs, rhs)};
    BatchOperand lhs_operand = make_batch_operand(lhs, domain);
    BatchOperand rhs_operand = make_batch_operand(rhs, domain);
    mapping_type mappings{
        OperandMapping::contiguous_blocks(domain, output_block_size),
        std::move(lhs_operand.m_mapping),
        std::move(rhs_operand.m_mapping),
    };
    return BatchMappings{
        .m_domain = std::move(domain),
        .m_mappings = std::move(mappings),
        .m_lhs_is_broadcast = lhs_operand.m_is_broadcast,
        .m_rhs_is_broadcast = rhs_operand.m_is_broadcast,
        .m_lhs_has_zero_stride = lhs_operand.m_has_zero_stride,
        .m_rhs_has_zero_stride = rhs_operand.m_has_zero_stride,
    };
}

template <typename Array>
MatmulPlan::shape_type MatmulPlan::make_batch_shape(Array const & lhs, Array const & rhs)
{
    size_t lhs_batch_axis_count = 0;
    if (lhs.ndim() > 1)
    {
        lhs_batch_axis_count = lhs.ndim() - 2;
    }
    size_t rhs_batch_axis_count = 0;
    if (rhs.ndim() > 1)
    {
        rhs_batch_axis_count = rhs.ndim() - 2;
    }
    size_t const batch_axis_count = std::max(lhs_batch_axis_count, rhs_batch_axis_count);
    shape_type batch_shape(batch_axis_count, 1);
    for (size_t offset = 0; offset < batch_axis_count; ++offset)
    {
        ssize_t const lhs_extent =
            offset < lhs_batch_axis_count ? lhs.shape(lhs_batch_axis_count - offset - 1) : 1;
        ssize_t const rhs_extent =
            offset < rhs_batch_axis_count ? rhs.shape(rhs_batch_axis_count - offset - 1) : 1;
        if (lhs_extent != rhs_extent && lhs_extent != 1 && rhs_extent != 1)
        {
            throw std::invalid_argument(
                std::format("SimpleArray::matmul(): batch shape "
                            "mismatch: this={} other={}",
                            shape_string(lhs),
                            shape_string(rhs)));
        }
        batch_shape[batch_axis_count - offset - 1] = lhs_extent == 1 ? rhs_extent : lhs_extent;
    }
    return batch_shape;
}

template <typename Array>
MatmulPlan::BatchOperand MatmulPlan::make_batch_operand(Array const & operand, LoopDomain const & domain)
{
    size_t batch_axis_count = 0;
    if (operand.ndim() > 1)
    {
        batch_axis_count = operand.ndim() - 2;
    }
    size_t const batch_axis_offset = domain.rank() - batch_axis_count;
    batch_stride_type strides(domain.rank(), 0);
    bool is_broadcast = false;
    bool has_zero_stride = false;
    for (size_t domain_axis = 0; domain_axis < domain.rank(); ++domain_axis)
    {
        if (domain_axis < batch_axis_offset)
        {
            is_broadcast = is_broadcast || domain.extent(domain_axis) > 1;
            continue;
        }

        size_t const operand_axis = domain_axis - batch_axis_offset;
        ssize_t const operand_extent = operand.shape(operand_axis);
        ssize_t const operand_stride = operand.stride(operand_axis);
        if (operand_extent == domain.extent(domain_axis))
        {
            strides[domain_axis] = operand_stride;
        }
        else if (domain.extent(domain_axis) > 1)
        {
            is_broadcast = true;
        }
        has_zero_stride = has_zero_stride || (operand_extent > 1 && operand_stride == 0);
    }
    return BatchOperand{
        .m_mapping = OperandMapping(std::move(strides)),
        .m_is_broadcast = is_broadcast,
        .m_has_zero_stride = has_zero_stride,
    };
}

template <typename Array>
std::string MatmulPlan::shape_string(Array const & array)
{
    std::string result = "(";
    for (ssize_t axis = 0; axis < array.ndim(); ++axis)
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

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
