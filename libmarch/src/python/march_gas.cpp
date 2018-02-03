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

    /* aliases for dependent type name lookup */
    using base_type = python::WrapBase< WrapGasSolver<NDIM>, gas::Solver<NDIM>, std::shared_ptr<gas::Solver<NDIM>> >;
    using wrapped_type = typename base_type::wrapped_type;
    using block_type = typename wrapped_type::block_type;

    friend base_type;

    WrapGasSolver(py::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        using quantity_reference = gas::Quantity<NDIM> &;
        (*this)
            .def(py::init([](block_type & block) {
                return gas::Solver<NDIM>::construct(block.shared_from_this());
            }))
            .def_property_readonly("block", &wrapped_type::block)
            .def_property(
                "trims",
                [](wrapped_type & self) -> std::vector<gas::TrimBase<NDIM>*> {
                    std::vector<gas::TrimBase<NDIM>*> ret;
                    for (auto & trim : self.trims()) {
                        ret.push_back(trim.get());
                    }
                    return ret;
                },
                [](wrapped_type & self, py::list in_trims) {
                    using ptr_type = std::unique_ptr<gas::TrimBase<NDIM>>;
                    std::vector<ptr_type> trims;
                    for (auto obj : in_trims) {
                        push_trim<gas::TrimInterface<NDIM>>(obj, trims);
                        push_trim<gas::TrimNoOp<NDIM>>(obj, trims);
                        push_trim<gas::TrimNonRefl<NDIM>>(obj, trims);
                        push_trim<gas::TrimSlipWall<NDIM>>(obj, trims);
                        push_trim<gas::TrimInlet<NDIM>>(obj, trims);
                    }
                    self.trims() = std::move(trims);
                }
            )
            .def_property_readonly(
                "anchors",
                [](wrapped_type & self) -> typename wrapped_type::anchor_chain_type & { return self.anchors(); },
                py::return_value_policy::reference_internal // FIXME: if it's default, remove this line
            )
            .def_property_readonly(
                "param",
                [](wrapped_type & self) -> gas::Parameter & { return self.param(); },
                py::return_value_policy::reference_internal // FIXME: if it's default, remove this line
            )
            .def_property_readonly(
                "state",
                [](wrapped_type & self) -> gas::State & { return self.state(); },
                py::return_value_policy::reference_internal // FIXME: if it's default, remove this line
            )
            .def_property_readonly(
                "sol",
                [](wrapped_type & self) -> typename wrapped_type::solution_type & { return self.sol(); },
                py::return_value_policy::reference_internal // FIXME: if it's default, remove this line
            )
            .def_property_readonly(
                "qty",
                [](wrapped_type & self) -> quantity_reference { return self.qty(); },
                py::return_value_policy::reference_internal // FIXME: if it's default, remove this line
            )
            .def("trim_do0", &wrapped_type::trim_do0)
            .def("trim_do1", &wrapped_type::trim_do1)
            .def("march", &wrapped_type::march)
            .def("init_solution", &wrapped_type::init_solution)
        ;

        this->m_cls.attr("ALMOST_ZERO") = double(wrapped_type::ALMOST_ZERO);
        this->m_cls.attr("neq") = NDIM + 2;
        this->m_cls.attr("_interface_init_") = std::make_tuple("cecnd", "cevol", "sfmrc");
        this->m_cls.attr("_solution_array_") = std::make_tuple("solt", "sol", "soln", "dsol", "dsoln");
    }

    template < class TrimType > static
    void push_trim(py::handle obj, std::vector<std::unique_ptr<gas::TrimBase<NDIM>>> & trims) {
        py::detail::make_caster<TrimType> conv;
        if (conv.load(obj, true)) {
            auto * cobj = obj.cast<TrimType*>();
            trims.push_back(make_unique<TrimType>(*cobj));
        }
    }

}; /* end class WrapGasSolver */

// FIXME: Check if PythonAnchor results into cyclic reference to (Python
// wrapped) Solver.  I may need an instance counting template.
class PythonAnchor : public gas::CommonAnchor
{

public:

    virtual ~PythonAnchor() {}

    template <size_t NDIM> PythonAnchor(ctor_passkey const & pk, gas::Solver<NDIM> & svr)
      : gas::CommonAnchor(pk, svr) {}

    template <size_t NDIM>
    static std::shared_ptr<PythonAnchor> construct(gas::Solver<NDIM> & svr) {
        return std::make_shared<PythonAnchor>(ctor_passkey(), svr);
    }

#define DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(NAME) \
    void NAME() override { PYBIND11_OVERLOAD(void, gas::CommonAnchor, NAME); }

    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(provide)
    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(preloop)
    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(premarch)
    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(prefull)
    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(presub)
    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(postsub)
    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(postfull)
    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(postmarch)
    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(postloop)
    DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(exhaust)

#undef DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD

}; /* end class PythonAnchor */

class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasCommonAnchor
  : public python::WrapBase< WrapGasCommonAnchor, gas::CommonAnchor, std::shared_ptr<gas::CommonAnchor>, PythonAnchor >
{

    friend base_type;

    WrapGasCommonAnchor(py::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
#define DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(NAME) \
            .def(#NAME, &wrapped_type::NAME)

        (*this)
            .def(py::init(
                [](gas::Solver<2> & svr) { return wrapped_base_type::construct(svr); }
            ))
            .def(py::init(
                [](gas::Solver<3> & svr) { return wrapped_base_type::construct(svr); }
            ))
            .def_property_readonly(
                "solver",
                [](wrapped_type & self) -> py::object {
                    if        (2 == self.ndim()) {
                        return py::cast(self.solver<2>());
                    } else if (3 == self.ndim()) {
                        return py::cast(self.solver<3>());
                    } else {
                        return py::none();
                    }
                }
            )
            .def_property_readonly(
                "make_owner",
                [](wrapped_type & self) -> py::object {
                    if        (2 == self.ndim()) {
                        return py::cast(self.make_owner<2>());
                    } else if (3 == self.ndim()) {
                        return py::cast(self.make_owner<3>());
                    } else {
                        return py::none();
                    }
                }
            )
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(provide)
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(preloop)
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(premarch)
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(prefull)
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(presub)
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(postsub)
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(postfull)
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(postmarch)
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(postloop)
            DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR(exhaust)
        ;

#undef DECL_MARCH_PYBIND_GAS_PYTHON_ANCHOR
    }

}; /* end class WrapGasCommonAnchor */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasAnchor
  : public python::WrapBase< WrapGasAnchor<NDIM>, gas::Anchor<NDIM>, std::shared_ptr<gas::Anchor<NDIM>> >
{

    /* aliases for dependent type name lookup */
    using base_type = python::WrapBase< WrapGasAnchor<NDIM>, gas::Anchor<NDIM>, std::shared_ptr<gas::Anchor<NDIM>> >;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapGasAnchor(py::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
#define DECL_MARCH_PYBIND_GAS_ANCHOR(NAME) \
            .def( \
                #NAME, \
                [](wrapped_type & self) { \
                    return self.wrapped_type::NAME(); \
                } \
            )

        (*this)
            .def(py::init([](typename wrapped_type::solver_type & solver) {
                return wrapped_type::construct(solver);
            }))
            DECL_MARCH_PYBIND_GAS_ANCHOR(provide)
            DECL_MARCH_PYBIND_GAS_ANCHOR(preloop)
            DECL_MARCH_PYBIND_GAS_ANCHOR(premarch)
            DECL_MARCH_PYBIND_GAS_ANCHOR(prefull)
            DECL_MARCH_PYBIND_GAS_ANCHOR(presub)
            DECL_MARCH_PYBIND_GAS_ANCHOR(postsub)
            DECL_MARCH_PYBIND_GAS_ANCHOR(postfull)
            DECL_MARCH_PYBIND_GAS_ANCHOR(postmarch)
            DECL_MARCH_PYBIND_GAS_ANCHOR(postloop)
            DECL_MARCH_PYBIND_GAS_ANCHOR(exhaust)
            .def_property_readonly(
                "solver",
                [](wrapped_type & self) -> typename wrapped_type::solver_type & { return self.solver(); },
                py::return_value_policy::reference_internal // FIXME: if it's default, remove this line
            )
        ;
        // FIXME: allow Python to extend from Anchor with both 2/3D

#undef DECL_MARCH_PYBIND_GAS_ANCHOR
    }

}; /* end class WrapGasAnchor */

/* This is to workaround https://github.com/pybind/pybind11/issues/1145.  The
 * lifecycle of the derived Python instances is kept in the manager. */
template< size_t NDIM >
class PythonAnchorManager : public gas::AnchorChain<NDIM>::LifeManager {

public:

    void append(py::object const & pyobj) { m_list.push_back(pyobj); }

private:

    std::list<py::object> m_list;

}; /* class PythonAnchorManager */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasAnchorChain
  : public python::WrapBase< WrapGasAnchorChain<NDIM>, gas::AnchorChain<NDIM> >
{

    /* aliases for dependent type name lookup */
    using base_type = python::WrapBase< WrapGasAnchorChain<NDIM>, gas::AnchorChain<NDIM> >;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapGasAnchorChain(py::module & mod, const char * pyname, const char * clsdoc)
      : base_type(mod, pyname, clsdoc)
    {
        (*this)
            .def(
                "append",
                [](wrapped_type & self, std::shared_ptr<gas::CommonAnchor> const & ptr, std::string const & name) {
                    self.append(ptr->make_owner<NDIM>(), name);
                    using mtype = PythonAnchorManager<NDIM>;
                    if (!self.life_manager()) {
                        self.life_manager() = make_unique<mtype>();
                    }
                    mtype & mgr = dynamic_cast<mtype &>(*self.life_manager());
                    mgr.append(py::cast(ptr));
                },
                py::arg("obj"), py::arg("name") = ""
            )
            .def("append", &wrapped_type::append, py::arg("obj"), py::arg("name") = "")
            .def("provide", &wrapped_type::provide)
            .def("preloop", &wrapped_type::preloop)
            .def("premarch", &wrapped_type::premarch)
            .def("prefull", &wrapped_type::prefull)
            .def("presub", &wrapped_type::presub)
            .def("postsub", &wrapped_type::postsub)
            .def("postfull", &wrapped_type::postfull)
            .def("postmarch", &wrapped_type::postmarch)
            .def("postloop", &wrapped_type::postloop)
            .def("exhaust", &wrapped_type::exhaust)
        ;
    }

}; /* end class WrapGasAnchorChain */

class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasParameter
  : public python::WrapBase< WrapGasParameter, gas::Parameter >
{

    friend base_type;

    WrapGasParameter(py::module & mod, const char * pyname, const char * clsdoc)
      : base_type(mod, pyname, clsdoc)
    {
#define DECL_MARCH_PYBIND_GAS_PARAMETER(TYPE, NAME) \
            .def_property( \
                #NAME, \
                [](wrapped_type const & self)            { return self.NAME(); }, \
                [](wrapped_type       & self, TYPE NAME) { self.NAME() = NAME; } \
            )

        (*this)
            DECL_MARCH_PYBIND_GAS_PARAMETER(real_type, sigma0)
            DECL_MARCH_PYBIND_GAS_PARAMETER(real_type, taumin)
            DECL_MARCH_PYBIND_GAS_PARAMETER(real_type, tauscale)
            DECL_MARCH_PYBIND_GAS_PARAMETER(real_type, stop_on_negative_density)
            DECL_MARCH_PYBIND_GAS_PARAMETER(real_type, stop_on_negative_energy)
        ;

#undef DECL_MARCH_PYBIND_GAS_PARAMETER
    }

}; /* end class WrapGasParameter */

class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasState
  : public python::WrapBase< WrapGasState, gas::State >
{

    friend base_type;

    WrapGasState(py::module & mod, const char * pyname, const char * clsdoc)
      : base_type(mod, pyname, clsdoc)
    {
#define DECL_MARCH_PYBIND_GAS_STATE(TYPE, NAME) \
            .def_property( \
                #NAME, \
                [](wrapped_type const & self)            { return self.NAME; }, \
                [](wrapped_type       & self, TYPE NAME) { self.NAME = NAME; } \
            )

        (*this)
            DECL_MARCH_PYBIND_GAS_STATE(real_type, time)
            DECL_MARCH_PYBIND_GAS_STATE(real_type, time_increment)
            DECL_MARCH_PYBIND_GAS_STATE(gas::State::int_type, step_current)
            DECL_MARCH_PYBIND_GAS_STATE(gas::State::int_type, step_global)
            DECL_MARCH_PYBIND_GAS_STATE(gas::State::int_type, substep_run)
            DECL_MARCH_PYBIND_GAS_STATE(gas::State::int_type, substep_current)
            DECL_MARCH_PYBIND_GAS_STATE(real_type, cfl_min)
            DECL_MARCH_PYBIND_GAS_STATE(real_type, cfl_max)
            DECL_MARCH_PYBIND_GAS_STATE(gas::State::int_type, cfl_nadjusted)
            DECL_MARCH_PYBIND_GAS_STATE(gas::State::int_type, cfl_nadjusted_accumulated)
        ;

#undef DECL_MARCH_PYBIND_GAS_STATE
    }

}; /* end class WrapGasState */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasSolution
  : public python::WrapBase< WrapGasSolution<NDIM>, gas::Solution<NDIM> >
{

    /* aliases for dependent type name lookup */
    using base_type = python::WrapBase< WrapGasSolution<NDIM>, gas::Solution<NDIM> >;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapGasSolution(py::module & mod, const char * pyname, const char * clsdoc)
      : base_type(mod, pyname, clsdoc)
    {
#define DECL_MARCH_PYBIND_GAS_SOLUTION(NAME) \
            .def_property_readonly( \
                #NAME, \
                [](wrapped_type & self) { \
                    return static_cast<LookupTableCore>(self.arrays().NAME()); \
                })

        (*this)
            DECL_MARCH_PYBIND_GAS_SOLUTION(so0c)
            DECL_MARCH_PYBIND_GAS_SOLUTION(so0n)
            DECL_MARCH_PYBIND_GAS_SOLUTION(so0t)
            DECL_MARCH_PYBIND_GAS_SOLUTION(so1c)
            DECL_MARCH_PYBIND_GAS_SOLUTION(so1n)
            DECL_MARCH_PYBIND_GAS_SOLUTION(stm)
            DECL_MARCH_PYBIND_GAS_SOLUTION(cflc)
            DECL_MARCH_PYBIND_GAS_SOLUTION(cflo)
            DECL_MARCH_PYBIND_GAS_SOLUTION(gamma)
        ;

#undef DECL_MARCH_PYBIND_GAS_SOLUTION
    }

}; /* end class WrapGasSolution */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasQuantity
  : public python::WrapBase< WrapGasQuantity<NDIM>, gas::Quantity<NDIM>, std::unique_ptr<gas::Quantity<NDIM>> >
{

    /* aliases for dependent type name lookup */
    using base_type = python::WrapBase< WrapGasQuantity<NDIM>, gas::Quantity<NDIM>, std::unique_ptr<gas::Quantity<NDIM>> >;
    using wrapped_type = typename base_type::wrapped_type;
    using solver_type = typename wrapped_type::solver_type;

    friend base_type;

    WrapGasQuantity(py::module & mod, const char * pyname, const char * clsdoc)
      : base_type(mod, pyname, clsdoc)
    {

// FIXME: change the properties to be like those of Solution
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
  : public python::WrapBase< WrapGasTrimBase<TrimType, NDIM>, TrimType, std::unique_ptr<TrimType>, gas::TrimBase<NDIM> >
{

    /* aliases for dependent type name lookup */
    using base_type = python::WrapBase< WrapGasTrimBase<TrimType, NDIM>, TrimType, std::unique_ptr<TrimType>, gas::TrimBase<NDIM> >;
    using wrapped_type = typename base_type::wrapped_type;
    using solver_type = typename wrapped_type::solver_type;

    friend base_type;

    WrapGasTrimBase(py::module & mod, const char * pyname, const char * clsdoc)
      : base_type(mod, pyname, clsdoc)
    {
        (*this)
            .def(py::init<solver_type &, BoundaryData &>())
            .def("apply_do0",
                 // FIXME: This (and apply_do1 wrapper) requires "wrapped_type"
                 // to be explicitly specified with a lambda.  Otherwise, it
                 // calls the base method.  Polymorphism for
                 // march::gas::Solver::m_trims works fine when caller is from
                 // C++.  I don't know why it behaves like this.  Perhaps
                 // pybind11 does something strange with the virtual function
                 // table?  Before having time to investigate, I keep this
                 // treatment as a workaround.
                 [](wrapped_type & self) { self.wrapped_type::apply_do0(); },
                 "Apply to variables of 0th order derivative")
            .def("apply_do1",
                 [](wrapped_type & self) { self.wrapped_type::apply_do1(); },
                 "Apply to variables of 1st order derivative")
        ;
    }

}; /* end class WrapGasTrimBase */

template< size_t NDIM > class MARCH_PYTHON_WRAPPER_VISIBILITY WrapGasTrimNoOp : public WrapGasTrimBase< gas::TrimNoOp<NDIM>, NDIM > {};
template< size_t NDIM > class MARCH_PYTHON_WRAPPER_VISIBILITY WrapGasTrimInterface : public WrapGasTrimBase< gas::TrimInterface<NDIM>, NDIM > {};
template< size_t NDIM > class MARCH_PYTHON_WRAPPER_VISIBILITY WrapGasTrimNonRefl : public WrapGasTrimBase< gas::TrimNonRefl<NDIM>, NDIM > {};
template< size_t NDIM > class MARCH_PYTHON_WRAPPER_VISIBILITY WrapGasTrimSlipWall : public WrapGasTrimBase< gas::TrimSlipWall<NDIM>, NDIM > {};
template< size_t NDIM > class MARCH_PYTHON_WRAPPER_VISIBILITY WrapGasTrimInlet : public WrapGasTrimBase< gas::TrimInlet<NDIM>, NDIM > {};

PyObject * python::ModuleInitializer::initialize_gas(py::module & upmod) {
    py::module gas = upmod.def_submodule("gas", "Gas dynamic solver");
    // section: solver and associated data
    WrapGasSolver<2>::commit(gas, "Solver2D", "Gas-dynamics solver (2D).");
    WrapGasSolver<3>::commit(gas, "Solver3D", "Gas-dynamics solver (3D).");
    WrapGasCommonAnchor::commit(gas, "CommonAnchor", "Gas-dynamics multi-dimensional anchor.");
    WrapGasAnchor<2>::commit(gas, "Anchor2D", "Gas-dynamics anchor (2D).");
    WrapGasAnchor<3>::commit(gas, "Anchor3D", "Gas-dynamics anchor (3D).");
    WrapGasAnchorChain<2>::commit(gas, "AnchorChain2D", "Gas-dynamics sequential container for anchors (2D).");
    WrapGasAnchorChain<3>::commit(gas, "AnchorChain3D", "Gas-dynamics sequential container for anchors (3D).");
    WrapGasParameter::commit(gas, "Parameter", "Gas-dynamics solver parameters.");
    WrapGasState::commit(gas, "State", "Gas-dynamics solver states.");
    WrapGasSolution<2>::commit(gas, "Solution2D", "Gas-dynamics solution data (2D).");
    WrapGasSolution<3>::commit(gas, "Solution3D", "Gas-dynamics solution data (3D).");
    WrapGasQuantity<2>::commit(gas, "Quantity2D", "Gas-dynamics quantities (2D).");
    WrapGasQuantity<3>::commit(gas, "Quantity3D", "Gas-dynamics quantities (3D).");
    // section: boundary-condition treatments
    WrapGasTrimBase<gas::TrimBase<2>, 2>::commit(gas, "TrimBase2D", "Gas-dynamics trim base type (2D).");
    WrapGasTrimBase<gas::TrimBase<3>, 3>::commit(gas, "TrimBase3D", "Gas-dynamics trim base type (3D).");
    WrapGasTrimInterface<2>::commit(gas, "TrimInterface2D", "Gas-dynamics interface trim (2D).");
    WrapGasTrimInterface<3>::commit(gas, "TrimInterface3D", "Gas-dynamics interface trim (3D).");
    WrapGasTrimNoOp<2>::commit(gas, "TrimNoOp2D", "Gas-dynamics no-op trim (2D).");
    WrapGasTrimNoOp<3>::commit(gas, "TrimNoOp3D", "Gas-dynamics no-op trim (3D).");
    WrapGasTrimNonRefl<2>::commit(gas, "TrimNonRefl2D", "Gas-dynamics non-reflective trim (2D).");
    WrapGasTrimNonRefl<3>::commit(gas, "TrimNonRefl3D", "Gas-dynamics non-reflective trim (3D).");
    WrapGasTrimSlipWall<2>::commit(gas, "TrimSlipWall2D", "Gas-dynamics slip wall trim (2D).");
    WrapGasTrimSlipWall<3>::commit(gas, "TrimSlipWall3D", "Gas-dynamics slip wall trim (3D).");
    WrapGasTrimInlet<2>::commit(gas, "TrimInlet2D", "Gas-dynamics inlet trim (2D).");
    WrapGasTrimInlet<3>::commit(gas, "TrimInlet3D", "Gas-dynamics inlet trim (3D).");
    return gas.ptr();
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
