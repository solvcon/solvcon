/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/wrap_SimpleArray.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

pybind11::dict MatmulBinding::eligibility(layout_type const & lhs, layout_type const & rhs, bool blas_supported)
{
    plan_type const plan = plan_type::make(lhs, rhs);
    pybind11::dict result;
    for (kernel_type const kernel : {
             kernel_type::Naive,
             kernel_type::BlasDot,
             kernel_type::BlasGevm,
             kernel_type::BlasGemv,
             kernel_type::BlasGemm,
             kernel_type::Winograd,
         })
    {
        std::string_view const reason = solvcon::detail::matmul_rejection(plan, kernel, blas_supported);
        std::string_view const name = solvcon::detail::matmul_kernel_name(kernel);
        result[pybind11::str(name)] = reason.empty() ? pybind11::none() : pybind11::cast(reason);
    }
    return result;
}

void wrap_SimpleArray(pybind11::module & mod)
{
    pybind11::register_exception<MatmulKernelUnavailable>(mod, "MatmulKernelUnavailable", PyExc_ValueError);

    wrap_SimpleArray_bool(mod);
    wrap_SimpleArray_int(mod);
    wrap_SimpleArray_uint(mod);
    wrap_SimpleArray_float(mod);
    wrap_SimpleArray_complex(mod);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
