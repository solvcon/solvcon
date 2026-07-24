#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/matmul_plan.hpp>

namespace solvcon
{

namespace detail
{

namespace execution
{

template <typename Array>
class MatmulExecutor
{
public:
    using value_type = typename Array::value_type;

    static Array execute(MatmulPlan const & plan, Array const & lhs, Array const & rhs);
}; /* end class MatmulExecutor */

template <typename Array>
Array MatmulExecutor<Array>::execute(MatmulPlan const & plan, Array const & lhs, Array const & rhs)
{
    Array output(plan.output_shape());
    value_type * output_data = output.logical_data();
    value_type const * lhs_data = lhs.logical_data();
    value_type const * rhs_data = rhs.logical_data();

    for (ssize_t row = 0; row < plan.rows(); ++row)
    {
        for (ssize_t column = 0; column < plan.columns(); ++column)
        {
            value_type total{};
            for (ssize_t inner = 0; inner < plan.inner_size(); ++inner)
            {
                ssize_t const lhs_offset = row * plan.lhs_row_stride() + inner * plan.lhs_inner_stride();
                ssize_t const rhs_offset = inner * plan.rhs_inner_stride() + column * plan.rhs_column_stride();
                total += lhs_data[lhs_offset] * rhs_data[rhs_offset];
            }
            output_data[row * plan.columns() + column] = total;
        }
    }
    return output;
}

} /* end namespace execution */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
