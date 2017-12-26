/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <utility>
#include <memory>
#include <algorithm>
#include <cstring>

#include "march/march.hpp"
#include "march/gas/gas.hpp"
#include "march/python/python.hpp"

namespace py = pybind11;

using namespace march;
using Table = march::python::Table;

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasSolver
  : public python::WrapBase< WrapGasSolver<NDIM>, gas::Solver<NDIM>, std::shared_ptr<gas::Solver<NDIM>> >
{

    using base_type = python::WrapBase< WrapGasSolver<NDIM>, gas::Solver<NDIM>, std::shared_ptr<gas::Solver<NDIM>> >;
    using wrapped_type = typename base_type::wrapped_type;
    using block_type = typename wrapped_type::block_type;

    friend base_type;

    WrapGasSolver(py::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
#define DECL_MARCH_PYBIND_GAS_SOLVER_PARAMETER(TYPE, NAME) \
            .def_property( \
                #NAME, \
                [](wrapped_type const & self)            { return self.param().NAME(); }, \
                [](wrapped_type       & self, TYPE NAME) { self.param().NAME() = NAME; } \
            )
#define DECL_MARCH_PYBIND_GAS_SOLVER_STATE(TYPE, NAME) \
            .def_property( \
                #NAME, \
                [](wrapped_type const & self)            { return self.state().NAME; }, \
                [](wrapped_type       & self, TYPE NAME) { self.state().NAME = NAME; } \
            )

        using quantity_reference = gas::Quantity<NDIM> &;
        (*this)
            .def(py::init([](block_type & block) {
                return gas::Solver<NDIM>::construct(block.shared_from_this());
            }))
            .def_property_readonly("block", &wrapped_type::block)
            .def_property_readonly(
                "qty",
                [](wrapped_type & self) -> quantity_reference { return self.qty(); },
                py::return_value_policy::reference_internal // FIXME: if it's default, remove this line
            )
            DECL_MARCH_PYBIND_GAS_SOLVER_PARAMETER(real_type, sigma0)
            DECL_MARCH_PYBIND_GAS_SOLVER_PARAMETER(real_type, taumin)
            DECL_MARCH_PYBIND_GAS_SOLVER_PARAMETER(real_type, tauscale)
            DECL_MARCH_PYBIND_GAS_SOLVER_PARAMETER(real_type, stop_on_negative_density)
            DECL_MARCH_PYBIND_GAS_SOLVER_PARAMETER(real_type, stop_on_negative_energy)
            DECL_MARCH_PYBIND_GAS_SOLVER_STATE(real_type, time)
            DECL_MARCH_PYBIND_GAS_SOLVER_STATE(real_type, time_increment)
            DECL_MARCH_PYBIND_GAS_SOLVER_STATE(real_type, step_current)
            DECL_MARCH_PYBIND_GAS_SOLVER_STATE(real_type, step_global)
            DECL_MARCH_PYBIND_GAS_SOLVER_STATE(real_type, substep_run)
            DECL_MARCH_PYBIND_GAS_SOLVER_STATE(real_type, substep_current)
            .def("update", &wrapped_type::update)
            .def("calc_cfl", &wrapped_type::calc_cfl)
            .def("calc_solt", &wrapped_type::calc_solt)
            .def("calc_soln", &wrapped_type::calc_soln)
            .def("calc_dsoln", &wrapped_type::calc_dsoln)
            .def("init_solution", &wrapped_type::init_solution)
            .def_property_readonly("amsca", [](wrapped_type & self) { return static_cast<LookupTableCore>(self.sup().amsca); })
            .def_property_readonly("sol"  , [](wrapped_type & self) { return static_cast<LookupTableCore>(self.sol().arrays().so0c()); })
            .def_property_readonly("soln" , [](wrapped_type & self) { return static_cast<LookupTableCore>(self.sol().arrays().so0n()); })
            .def_property_readonly("solt" , [](wrapped_type & self) { return static_cast<LookupTableCore>(self.sol().arrays().so0t()); })
            .def_property_readonly("dsol" , [](wrapped_type & self) { return static_cast<LookupTableCore>(self.sol().arrays().so1c()); })
            .def_property_readonly("dsoln", [](wrapped_type & self) { return static_cast<LookupTableCore>(self.sol().arrays().so1n()); })
            .def_property_readonly("stm"  , [](wrapped_type & self) { return static_cast<LookupTableCore>(self.sol().arrays().stm()); })
            .def_property_readonly("cfl"  , [](wrapped_type & self) { return static_cast<LookupTableCore>(self.sol().arrays().cflc()); })
            .def_property_readonly("ocfl" , [](wrapped_type & self) { return static_cast<LookupTableCore>(self.sol().arrays().cflo()); })
        ;

        this->m_cls.attr("ALMOST_ZERO") = double(wrapped_type::ALMOST_ZERO);
        this->m_cls.attr("neq") = NDIM + 2;
        this->m_cls.attr("_interface_init_") = std::make_tuple("cecnd", "cevol", "sfmrc");
        this->m_cls.attr("_solution_array_") = std::make_tuple("solt", "sol", "soln", "dsol", "dsoln");

#undef DECL_MARCH_PYBIND_GAS_SOLVER_STATE
#undef DECL_MARCH_PYBIND_GAS_SOLVER_PARAMETER
    }

}; /* end class WrapGasSolver */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasQuantity
  : public python::WrapBase< WrapGasQuantity<NDIM>, gas::Quantity<NDIM>, std::unique_ptr<gas::Quantity<NDIM>> >
{

    using base_type = python::WrapBase< WrapGasQuantity<NDIM>, gas::Quantity<NDIM>, std::unique_ptr<gas::Quantity<NDIM>> >;
    using wrapped_type = typename base_type::wrapped_type;
    using solver_type = typename wrapped_type::solver_type;

    friend base_type;

    WrapGasQuantity(py::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {

#define DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(NAME, ARR) \
        .def_property( \
            #NAME, \
            [](wrapped_type & qty)                { return Table(qty.NAME()).ARR(); }, \
            [](wrapped_type & qty, py::array src) { Table::CopyInto(Table(qty.NAME()).ARR(), src); }, \
            #NAME " " #ARR " array")

        (*this)
            .def("update", &wrapped_type::update, "Update the physics")
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(density             , full)
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(velocity            , full)
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(vorticity           , full)
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(vorticity_magnitude , full)
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(ke                  , full)
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(pressure            , full)
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(temperature         , full)
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(soundspeed          , full)
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(mach                , full)
            DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(schlieren           , full)
        ;

#undef DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY
    }

}; /* end class WrapGasQuantity */

template< class TrimType, size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasTrimBase
  : public python::WrapBase< WrapGasTrimBase<TrimType, NDIM>, TrimType, std::unique_ptr<TrimType> >
{

    using base_type = python::WrapBase< WrapGasTrimBase<TrimType, NDIM>, TrimType, std::unique_ptr<TrimType> >;
    using wrapped_type = typename base_type::wrapped_type;
    using solver_type = typename wrapped_type::solver_type;

    friend base_type;

    WrapGasTrimBase(py::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        (*this)
            .def(py::init<solver_type &, BoundaryData &>())
            .def("apply_do0", &wrapped_type::apply_do0, "Apply to variables of 0th order derivative")
            .def("apply_do1", &wrapped_type::apply_do1, "Apply to variables of 1st order derivative")
        ;
    }

}; /* end class WrapGasTrimBase */

template< size_t NDIM > class MARCH_PYTHON_WRAPPER_VISIBILITY WrapGasTrimNoOp : public WrapGasTrimBase< gas::TrimNoOp<NDIM>, NDIM > {};
template< size_t NDIM > class MARCH_PYTHON_WRAPPER_VISIBILITY WrapGasTrimNonRefl : public WrapGasTrimBase< gas::TrimNonRefl<NDIM>, NDIM > {};
template< size_t NDIM > class MARCH_PYTHON_WRAPPER_VISIBILITY WrapGasTrimSlipWall : public WrapGasTrimBase< gas::TrimSlipWall<NDIM>, NDIM > {};
template< size_t NDIM > class MARCH_PYTHON_WRAPPER_VISIBILITY WrapGasTrimInlet : public WrapGasTrimBase< gas::TrimInlet<NDIM>, NDIM > {};

PyObject * python::ModuleInitializer::initialize_gas(py::module & upmod) {
    py::module gas = upmod.def_submodule("gas", "Gas dynamic solver");
    WrapGasSolver<2>::commit(gas, "Solver2D", "Gas-dynamic solver (2D).");
    WrapGasSolver<3>::commit(gas, "Solver3D", "Gas-dynamic solver (3D).");
    WrapGasQuantity<2>::commit(gas, "Quantity2D", "Gas-dynamics quantities (2D).");
    WrapGasQuantity<3>::commit(gas, "Quantity3D", "Gas-dynamics quantities (3D).");
    WrapGasTrimNoOp<2>::commit(gas, "TrimNoOp2D", "Gas-dynamics non-reflective trim (2D).");
    WrapGasTrimNoOp<3>::commit(gas, "TrimNoOp3D", "Gas-dynamics non-reflective trim (3D).");
    WrapGasTrimNonRefl<2>::commit(gas, "TrimNonRefl2D", "Gas-dynamics non-reflective trim (2D).");
    WrapGasTrimNonRefl<3>::commit(gas, "TrimNonRefl3D", "Gas-dynamics non-reflective trim (3D).");
    WrapGasTrimSlipWall<2>::commit(gas, "TrimSlipWall2D", "Gas-dynamics non-reflective trim (2D).");
    WrapGasTrimSlipWall<3>::commit(gas, "TrimSlipWall3D", "Gas-dynamics non-reflective trim (3D).");
    WrapGasTrimInlet<2>::commit(gas, "TrimInlet2D", "Gas-dynamics non-reflective trim (2D).");
    WrapGasTrimInlet<3>::commit(gas, "TrimInlet3D", "Gas-dynamics non-reflective trim (3D).");
    return gas.ptr();
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
