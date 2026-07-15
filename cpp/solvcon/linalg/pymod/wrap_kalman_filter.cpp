/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/solvcon.hpp>
#include <solvcon/linalg/pymod/linalg_pymod.hpp>
#include <pybind11/operators.h>

namespace solvcon
{

namespace python
{

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapKalmanStateInfo
    : public WrapBase<WrapKalmanStateInfo<T>, KalmanStateInfo<T>>
{
    using base_type = WrapBase<WrapKalmanStateInfo<T>, KalmanStateInfo<T>>;
    using wrapped_type = typename base_type::wrapped_type;
    using array_type = SimpleArray<T>;

    friend base_type;

    WrapKalmanStateInfo(pybind11::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                py::init(
                    [](size_t observation_size, size_t state_size)
                    {
                        return wrapped_type(observation_size, state_size);
                    }),
                py::arg("observation_size"),
                py::arg("state_size"))
            .def_readwrite(
                "prior_states", &wrapped_type::prior_states)
            .def_readwrite(
                "prior_states_covariance", &wrapped_type::prior_states_covariance)
            .def_readwrite(
                "posterior_states", &wrapped_type::posterior_states)
            .def_readwrite(
                "posterior_states_covariance", &wrapped_type::posterior_states_covariance);
    }

}; /* end class WrapKalmanStateInfo */

template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapKalmanFilter
    : public WrapBase<WrapKalmanFilter<T>, KalmanFilter<T>>
{
    using base_type = WrapBase<WrapKalmanFilter<T>, KalmanFilter<T>>;
    using wrapped_type = typename base_type::wrapped_type;
    using array_type = SimpleArray<T>;
    using real_type = typename wrapped_type::real_type;

    friend base_type;

    WrapKalmanFilter(pybind11::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                py::init(
                    [](array_type const & x, array_type const & f, py::object const & b, array_type const & h, real_type process_noise, real_type measurement_noise, real_type jitter)
                    {
                        array_type b_array;
                        if (b.is_none())
                        {
                            b_array = array_type(small_vector<ssize_t>{x.shape(0), 0});
                        }
                        else
                        {
                            b_array = b.cast<array_type>();
                        }
                        return wrapped_type(x, f, b_array, h, process_noise, measurement_noise, jitter);
                    }),
                py::arg("x"),
                py::arg("f"),
                py::arg("b") = py::none(),
                py::arg("h"),
                py::arg("process_noise"),
                py::arg("measurement_noise"),
                py::arg("jitter") = static_cast<real_type>(1e-9))
            .def(
                py::init(
                    [](array_type const & x,
                       array_type const & f,
                       py::object const & b,
                       array_type const & h,
                       array_type const & process_noise_covariance,
                       array_type const & measurement_noise_covariance,
                       array_type const & covariance,
                       real_type jitter)
                    {
                        array_type b_array;
                        if (b.is_none())
                        {
                            b_array = array_type(small_vector<ssize_t>{x.shape(0), 0});
                        }
                        else
                        {
                            b_array = b.cast<array_type>();
                        }
                        return wrapped_type(
                            x,
                            f,
                            b_array,
                            h,
                            process_noise_covariance,
                            measurement_noise_covariance,
                            covariance,
                            jitter);
                    }),
                py::arg("x"),
                py::arg("f"),
                py::arg("b") = py::none(),
                py::arg("h"),
                py::arg("q"),
                py::arg("r"),
                py::arg("p"),
                py::arg("jitter") = static_cast<real_type>(1e-9))
            .def_property_readonly(
                "state",
                &wrapped_type::state)
            .def_property_readonly(
                "covariance",
                &wrapped_type::covariance)
            .def(
                "predict",
                [](wrapped_type & self)
                { self.predict(); })
            .def(
                "predict",
                [](wrapped_type & self, array_type const & u)
                { self.predict(u); },
                py::arg("u"))
            .def(
                "update",
                &wrapped_type::update,
                py::arg("z"))
            .def(
                "batch_filter",
                [](wrapped_type & self, array_type const & zs, array_type const & us)
                {
                    return self.batch_filter(zs, us);
                },
                py::arg("zs"),
                py::arg("us"))
            .def(
                "batch_filter",
                [](wrapped_type & self, array_type const & zs)
                {
                    return self.batch_filter(zs);
                },
                py::arg("zs"));
    }

}; /* end class WrapKalmanFilter */

void wrap_states_info(pybind11::module & mod)
{
    WrapKalmanStateInfo<float>::commit(mod, "KalmanStateInfoFp32", "KalmanStateInfoFp32");
    WrapKalmanStateInfo<double>::commit(mod, "KalmanStateInfoFp64", "KalmanStateInfoFp64");
    WrapKalmanStateInfo<Complex<float>>::commit(mod, "KalmanStateInfoComplex64", "KalmanStateInfoComplex64");
    WrapKalmanStateInfo<Complex<double>>::commit(mod, "KalmanStateInfoComplex128", "KalmanStateInfoComplex128");
}

void wrap_kalman_filter(pybind11::module & mod)
{
    WrapKalmanFilter<float>::commit(mod, "KalmanFilterFp32", "Kalman Filter (float)");
    WrapKalmanFilter<double>::commit(mod, "KalmanFilterFp64", "Kalman Filter (double)");
    WrapKalmanFilter<Complex<float>>::commit(mod, "KalmanFilterComplex64", "Kalman Filter (complex float)");
    WrapKalmanFilter<Complex<double>>::commit(mod, "KalmanFilterComplex128", "Kalman Filter (complex double)");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
