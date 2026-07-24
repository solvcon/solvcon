#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/execution/matmul.hpp>

namespace solvcon
{

namespace detail
{

template <typename Array>
class SimpleArrayExecution
{
public:
    static Array matmul_planned(Array const & self, Array const & other);
}; /* end class SimpleArrayExecution */

template <typename Array>
Array SimpleArrayExecution<Array>::matmul_planned(Array const & self, Array const & other)
{
    execution::MatmulPlan const plan = execution::MatmulPlan::make(self, other);
    return execution::MatmulExecutor<Array>::execute(plan, self, other);
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
