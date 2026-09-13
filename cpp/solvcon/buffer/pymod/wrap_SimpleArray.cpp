/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/wrap_SimpleArray.hpp> // Must be the first include.

#include <algorithm>
#include <limits>

namespace solvcon
{

namespace python
{

namespace detail
{

namespace
{

constexpr ssize_t MAX_ELEMENTS = std::numeric_limits<ssize_t>::max();

class MatmulLayout
{
public:
    MatmulLayout(pybind11::object const & shape, pybind11::object const & strides);
    ssize_t ndim() const { return static_cast<ssize_t>(m_shape.size()); }
    ssize_t shape(size_t axis) const { return m_shape[axis]; }
    ssize_t stride(size_t axis) const { return m_strides[axis]; }

private:
    solvcon::detail::shape_type m_shape;
    solvcon::detail::shape_type m_strides;
}; /* end class MatmulLayout */

} /* end namespace */

MatmulLayout::MatmulLayout(pybind11::object const & shape, pybind11::object const & strides)
    : m_shape(make_shape(shape))
    , m_strides(make_shape(strides))
{
    if (m_shape.empty() || m_shape.size() != m_strides.size())
    {
        throw std::invalid_argument("shape and strides must have the same nonzero rank");
    }
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

pybind11::dict matmul_kernel_eligibility(
    pybind11::object const & lhs_shape, pybind11::object const & lhs_strides, pybind11::object const & rhs_shape, pybind11::object const & rhs_strides, bool blas_supported)
{
    namespace py = pybind11;
    using solvcon::detail::MatmulKernel;

    MatmulLayout const lhs(lhs_shape, lhs_strides), rhs(rhs_shape, rhs_strides);
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
    auto const plan = solvcon::detail::MatmulPlan::make(lhs, rhs);
    py::dict result;
    for (auto kernel : {
             MatmulKernel::Naive,
             MatmulKernel::BlasDot,
             MatmulKernel::BlasGevm,
             MatmulKernel::BlasGemv,
             MatmulKernel::BlasGemm,
             MatmulKernel::Winograd,
         })
    {
        char const * const reason = solvcon::detail::matmul_rejection_reason(plan, kernel, blas_supported);
        result[py::str(solvcon::detail::matmul_kernel_name(kernel))] =
            reason ? py::object(py::str(reason)) : py::object(py::none());
    }
    return result;
}

} /* end namespace detail */

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
