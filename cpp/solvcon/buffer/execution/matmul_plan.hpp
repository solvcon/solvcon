#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/small_vector.hpp>

#include <stdexcept>

namespace solvcon
{

namespace detail
{

namespace execution
{

using shape_type = small_vector<ssize_t>;

class MatmulPlan
{
public:
    shape_type const & output_shape() const noexcept { return m_output_shape; }
    ssize_t rows() const noexcept { return m_rows; }
    ssize_t columns() const noexcept { return m_columns; }
    ssize_t inner_size() const noexcept { return m_inner_size; }
    ssize_t lhs_row_stride() const noexcept { return m_lhs_row_stride; }
    ssize_t lhs_inner_stride() const noexcept { return m_lhs_inner_stride; }
    ssize_t rhs_inner_stride() const noexcept { return m_rhs_inner_stride; }
    ssize_t rhs_column_stride() const noexcept { return m_rhs_column_stride; }

    template <typename Array>
    static MatmulPlan make(Array const & lhs, Array const & rhs);

private:
    shape_type m_output_shape;
    ssize_t m_rows = 0;
    ssize_t m_columns = 0;
    ssize_t m_inner_size = 0;
    ssize_t m_lhs_row_stride = 0;
    ssize_t m_lhs_inner_stride = 0;
    ssize_t m_rhs_inner_stride = 0;
    ssize_t m_rhs_column_stride = 0;
}; /* end class MatmulPlan */

template <typename Array>
MatmulPlan MatmulPlan::make(Array const & lhs, Array const & rhs)
{
    if (lhs.ndim() != 2 || rhs.ndim() != 2)
    {
        throw std::invalid_argument("planned matrix-matrix matmul requires rank-2 operands");
    }
    if (lhs.shape(1) != rhs.shape(0))
    {
        throw std::invalid_argument("planned matmul contracted dimensions differ");
    }

    MatmulPlan plan;
    plan.m_rows = lhs.shape(0);
    plan.m_columns = rhs.shape(1);
    plan.m_inner_size = lhs.shape(1);
    plan.m_output_shape = shape_type{plan.m_rows, plan.m_columns};
    plan.m_lhs_row_stride = lhs.stride(0);
    plan.m_lhs_inner_stride = lhs.stride(1);
    plan.m_rhs_inner_stride = rhs.stride(0);
    plan.m_rhs_column_stride = rhs.stride(1);
    return plan;
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
