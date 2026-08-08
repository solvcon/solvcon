/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/python/common.hpp> // Must be the first include.
#include <pybind11/stl.h>

#include <solvcon/multidim/multidim.hpp>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapGradientElement
    : public WrapBase<WrapGradientElement, GradientElement>
{

public:

    using base_type = WrapBase<WrapGradientElement, GradientElement>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapGradientElement(pybind11::module & mod, const char * pyname, const char * clsdoc);

    WrapGradientElement & wrap_management();
    WrapGradientElement & wrap_gradient();

    static void check_ifl(wrapped_type const & self, wrapped_type::int_type ifl);
    static void check_d(wrapped_type const & self, wrapped_type::int_type d);
    static void check_ifge(wrapped_type const & self, wrapped_type::int_type ifge);

}; /* end class WrapGradientElement */

void WrapGradientElement::check_ifl(wrapped_type const & self, wrapped_type::int_type ifl)
{
    if (ifl < 0 || ifl >= self.clnfc())
    {
        throw std::out_of_range(std::format(
            "GradientElement: ifl {} out of range [0, {})", ifl, self.clnfc()));
    }
}

void WrapGradientElement::check_d(wrapped_type const & self, wrapped_type::int_type d)
{
    if (d < 0 || d >= self.ndim())
    {
        throw std::out_of_range(std::format(
            "GradientElement: d {} out of range [0, {})", d, self.ndim()));
    }
}

void WrapGradientElement::check_ifge(wrapped_type const & self, wrapped_type::int_type ifge)
{
    if (ifge < 0 || ifge >= self.nfge())
    {
        throw std::out_of_range(std::format(
            "GradientElement: ifge {} out of range [0, {})", ifge, self.nfge()));
    }
}

WrapGradientElement::WrapGradientElement(pybind11::module & mod, const char * pyname, const char * clsdoc)
    : base_type(mod, pyname, clsdoc)
{
    (*this)
        .wrap_management()
        .wrap_gradient()
        //
        ;
}

WrapGradientElement & WrapGradientElement::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](std::shared_ptr<StaticMesh> const & mesh,
                   SimpleArray<wrapped_type::real_type> const & cecnd,
                   wrapped_type::int_type icl,
                   wrapped_type::real_type tau)
                {
                    return wrapped_type(*mesh, cecnd, icl, tau);
                }),
            py::arg("mesh"),
            py::arg("cecnd"),
            py::arg("icl"),
            py::arg("tau"),
            py::keep_alive<1, 2>())
        .def_property_readonly("icl", &wrapped_type::icl)
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("clnfc", &wrapped_type::clnfc)
        .def_property_readonly("nfge", &wrapped_type::nfge)
        .def_property_readonly("nfge_inverse", &wrapped_type::nfge_inverse)
        //
        ;

    return *this;
}

WrapGradientElement & WrapGradientElement::wrap_gradient()
{
    namespace py = pybind11;

    (*this)
        .def(
            "rcl",
            [](wrapped_type const & self, wrapped_type::int_type ifl)
            {
                check_ifl(self, ifl);
                return self.rcl(ifl);
            },
            py::arg("ifl"))
        .def(
            "idis",
            [](wrapped_type const & self, wrapped_type::int_type ifl, wrapped_type::int_type d)
            {
                check_ifl(self, ifl);
                check_d(self, d);
                return self.idis(ifl, d);
            },
            py::arg("ifl"),
            py::arg("d"))
        .def(
            "jdis",
            [](wrapped_type const & self, wrapped_type::int_type ifl, wrapped_type::int_type d)
            {
                check_ifl(self, ifl);
                check_d(self, d);
                return self.jdis(ifl, d);
            },
            py::arg("ifl"),
            py::arg("d"))
        .def(
            "faces",
            [](wrapped_type const & self, wrapped_type::int_type ifge)
            {
                check_ifge(self, ifge);
                auto const & tface = self.faces(ifge);
                std::vector<wrapped_type::int_type> out(self.ndim());
                for (uint8_t ivx = 0; ivx < self.ndim(); ++ivx)
                {
                    out[ivx] = tface[ivx];
                }
                return out;
            },
            py::arg("ifge"))
        .def(
            "displacement_matrix",
            [](wrapped_type const & self, wrapped_type::int_type ifge)
            {
                check_ifge(self, ifge);
                auto const dst = self.displacement_matrix(ifge);
                size_t const nd = self.ndim();
                std::vector<std::vector<wrapped_type::real_type>> out(
                    nd, std::vector<wrapped_type::real_type>(nd));
                for (size_t i = 0; i < nd; ++i)
                {
                    for (size_t j = 0; j < nd; ++j)
                    {
                        out[i][j] = dst[i][j];
                    }
                }
                return out;
            },
            py::arg("ifge"))
        .def(
            "solve_gradient",
            [](wrapped_type const & self,
               wrapped_type::int_type ifge,
               std::vector<wrapped_type::real_type> const & udf)
            {
                check_ifge(self, ifge);
                if (udf.size() != self.ndim())
                {
                    throw std::invalid_argument(std::format(
                        "GradientElement: udf size {} must equal ndim {}",
                        udf.size(),
                        self.ndim()));
                }
                wrapped_type::ge_vector_type u = {0, 0, 0};
                for (size_t d = 0; d < self.ndim(); ++d)
                {
                    u[d] = udf[d];
                }
                wrapped_type::ge_vector_type const g = self.solve_gradient(ifge, u);
                std::vector<wrapped_type::real_type> out(self.ndim());
                for (size_t d = 0; d < self.ndim(); ++d)
                {
                    out[d] = g[d];
                }
                return out;
            },
            py::arg("ifge"),
            py::arg("udf"))
        //
        ;

    return *this;
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapEulerCore
    : public WrapBase<WrapEulerCore, EulerCore, std::shared_ptr<EulerCore>>
{

public:

    using base_type = WrapBase<WrapEulerCore, EulerCore, std::shared_ptr<EulerCore>>;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

protected:

    WrapEulerCore(pybind11::module & mod, const char * pyname, const char * clsdoc);

    WrapEulerCore & wrap_management();
    WrapEulerCore & wrap_preparation();
    WrapEulerCore & wrap_march();
    WrapEulerCore & wrap_boundary();
    WrapEulerCore & wrap_array();

}; /* end class WrapEulerCore */

WrapEulerCore::WrapEulerCore(pybind11::module & mod, const char * pyname, const char * clsdoc)
    : base_type(mod, pyname, clsdoc)
{
    (*this)
        .wrap_management()
        .wrap_preparation()
        .wrap_march()
        .wrap_boundary()
        .wrap_array()
        //
        ;
}

WrapEulerCore & WrapEulerCore::wrap_management()
{
    namespace py = pybind11;

    (*this)
        .def(
            py::init(
                [](std::shared_ptr<StaticMesh> const & mesh, wrapped_type::real_type time_increment)
                {
                    return wrapped_type::construct(mesh, time_increment);
                }),
            py::arg("mesh"),
            py::arg("time_increment"))
        .def_property_readonly("ndim", &wrapped_type::ndim)
        .def_property_readonly("ncell", &wrapped_type::ncell)
        .def_property_readonly("ngstcell", &wrapped_type::ngstcell)
        .def_property_readonly("neq", &wrapped_type::neq)
        //
        ;

    return *this;
}

WrapEulerCore & WrapEulerCore::wrap_preparation()
{
    namespace py = pybind11;

    (*this)
        .def_timed("prepare_ce", &wrapped_type::prepare_ce)
        .def_timed(
            "init_solution",
            [](wrapped_type & self,
               wrapped_type::real_type gamma,
               wrapped_type::real_type rho,
               std::vector<wrapped_type::real_type> const & v,
               wrapped_type::real_type p)
            {
                if (v.size() != self.ndim())
                {
                    throw std::invalid_argument(std::format(
                        "EulerCore::init_solution: velocity size {} must equal ndim {}",
                        v.size(),
                        self.ndim()));
                }
                std::array<wrapped_type::real_type, 3> velocity = {0, 0, 0};
                for (size_t d = 0; d < self.ndim(); ++d)
                {
                    velocity[d] = v[d];
                }
                self.init_solution(gamma, rho, velocity, p);
            },
            py::arg("gamma"),
            py::arg("rho"),
            py::arg("v"),
            py::arg("p"))
        //
        ;

    return *this;
}

WrapEulerCore & WrapEulerCore::wrap_march()
{
    namespace py = pybind11;

    (*this)
        .def_property_readonly("time_increment", &wrapped_type::time_increment)
        .def_property("sigma0", &wrapped_type::sigma0, &wrapped_type::set_sigma0)
        .def_property("taumin", &wrapped_type::taumin, &wrapped_type::set_taumin)
        .def_property("tauscale", &wrapped_type::tauscale, &wrapped_type::set_tauscale)
        .def_timed("march", &wrapped_type::march, py::arg("steps"))
        .def_timed("march_substep", &wrapped_type::march_substep)
        .def_timed("update", &wrapped_type::update)
        .def_timed("calc_cfl", &wrapped_type::calc_cfl)
        .def_timed("calc_solt", &wrapped_type::calc_solt)
        .def_timed("calc_soln", &wrapped_type::calc_soln)
        .def_timed("calc_dsoln", &wrapped_type::calc_dsoln)
        //
        ;

    return *this;
}

WrapEulerCore & WrapEulerCore::wrap_boundary()
{
    namespace py = pybind11;

    (*this)
        .def(
            "add_nonrefl",
            [](wrapped_type & self, std::vector<wrapped_type::int_type> const & faces)
            {
                self.add_bc(EulerBC::NonReflective, faces, {});
            },
            py::arg("faces"))
        .def(
            "add_slipwall",
            [](wrapped_type & self, std::vector<wrapped_type::int_type> const & faces)
            {
                self.add_bc(EulerBC::SlipWall, faces, {});
            },
            py::arg("faces"))
        .def(
            "add_inlet",
            [](wrapped_type & self,
               std::vector<wrapped_type::int_type> const & faces,
               std::vector<wrapped_type::real_type> const & value)
            {
                self.add_bc(EulerBC::Inlet, faces, value);
            },
            py::arg("faces"),
            py::arg("value"))
        .def("clear_bc", &wrapped_type::clear_bc)
        .def(
            "get_normal_matrix",
            [](wrapped_type const & self, wrapped_type::int_type ifc)
            {
                auto const nface = static_cast<wrapped_type::int_type>(self.mesh()->nface());
                if (ifc < 0 || ifc >= nface)
                {
                    throw std::out_of_range(std::format(
                        "EulerCore: face {} out of range [0, {})", ifc, nface));
                }
                return self.get_normal_matrix(ifc);
            },
            py::arg("ifc"))
        .def_timed("bc_soln", &wrapped_type::bc_soln)
        .def_timed("bc_dsoln", &wrapped_type::bc_dsoln)
        //
        ;

    return *this;
}

WrapEulerCore & WrapEulerCore::wrap_array()
{
#define SC_DECL_ARRAY(NAME) \
    .expose_SimpleArray(#NAME, [](wrapped_type & self) -> decltype(auto) { return self.NAME(); })

    // clang-format off
    (*this)
        SC_DECL_ARRAY(cevol)
        SC_DECL_ARRAY(cecnd)
        SC_DECL_ARRAY(sfcnd)
        SC_DECL_ARRAY(sfnml)
        SC_DECL_ARRAY(so0c)
        SC_DECL_ARRAY(so0n)
        SC_DECL_ARRAY(so0t)
        SC_DECL_ARRAY(so1c)
        SC_DECL_ARRAY(so1n)
        SC_DECL_ARRAY(stm)
        SC_DECL_ARRAY(cflo)
        SC_DECL_ARRAY(cflc)
        SC_DECL_ARRAY(gamma)
        //
        ;
    // clang-format on

#undef SC_DECL_ARRAY

    return *this;
}

void wrap_multidim(pybind11::module & mod)
{
    WrapGradientElement::commit(mod, "GradientElement", "Gradient element for a single cell");
    WrapEulerCore::commit(mod, "EulerCore", "Solve the Euler equation");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
