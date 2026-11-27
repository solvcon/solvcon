/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/solvcon.hpp>

#include <solvcon/transform/pymod/transform_pymod.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapFourierTransform
    : public WrapBase<WrapFourierTransform, solvcon::FourierTransform, std::shared_ptr<solvcon::FourierTransform>>
{
    using base_type = WrapBase<WrapFourierTransform, solvcon::FourierTransform, std::shared_ptr<solvcon::FourierTransform>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    // Python picks the backend at runtime, so the dispatchers below map the
    // enum value onto the compile-time backend instantiations.
    template <template <typename> class T1, typename T2>
    static void fft_dispatch(
        solvcon::SimpleArray<T1<T2>> const & in, solvcon::SimpleArray<T1<T2>> & out, solvcon::FourierBackend backend)
    {
        if (solvcon::FourierBackend::cuda == backend)
        {
            wrapped_type::fft<solvcon::FourierBackend::cuda>(in, out);
        }
        else
        {
            wrapped_type::fft<solvcon::FourierBackend::cpu>(in, out);
        }
    }

    template <template <typename> class T1, typename T2>
    static void ifft_dispatch(
        solvcon::SimpleArray<T1<T2>> const & in, solvcon::SimpleArray<T1<T2>> & out, solvcon::FourierBackend backend)
    {
        if (solvcon::FourierBackend::cuda == backend)
        {
            wrapped_type::ifft<solvcon::FourierBackend::cuda>(in, out);
        }
        else
        {
            wrapped_type::ifft<solvcon::FourierBackend::cpu>(in, out);
        }
    }

    WrapFourierTransform(pybind11::module & mod, char const * pyname, char const * pydoc)
        : WrapBase<WrapFourierTransform, solvcon::FourierTransform, std::shared_ptr<solvcon::FourierTransform>>(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def_static(
                "fft",
                &fft_dispatch<solvcon::Complex, double>,
                py::arg("input"),
                py::arg("output"),
                py::arg("backend") = solvcon::FourierBackend::cpu)
            .def_static(
                "fft",
                &fft_dispatch<solvcon::Complex, float>,
                py::arg("input"),
                py::arg("output"),
                py::arg("backend") = solvcon::FourierBackend::cpu)
            .def_static(
                "ifft",
                &ifft_dispatch<solvcon::Complex, double>,
                py::arg("input"),
                py::arg("output"),
                py::arg("backend") = solvcon::FourierBackend::cpu)
            .def_static(
                "ifft",
                &ifft_dispatch<solvcon::Complex, float>,
                py::arg("input"),
                py::arg("output"),
                py::arg("backend") = solvcon::FourierBackend::cpu)
            .def_static("dft", &wrapped_type::dft<solvcon::Complex, double>, py::arg("input"), py::arg("output"))
            .def_static("dft", &wrapped_type::dft<solvcon::Complex, float>, py::arg("input"), py::arg("output"))
            .def_static("cuda_available", &solvcon::device::cuda::available);
    }

}; /* end class WrapFourierTransform */

void wrap_FourierTransform(pybind11::module & mod)
{
    pybind11::enum_<solvcon::FourierBackend>(mod, "FourierBackend")
        .value("cpu", solvcon::FourierBackend::cpu)
        .value("cuda", solvcon::FourierBackend::cuda);
    WrapFourierTransform::commit(mod, "FourierTransform", "Fourier transform library");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
