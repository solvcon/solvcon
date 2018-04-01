/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <memory>
#include <algorithm>
#include <cstring>
#include <list>

#include "march.hpp"
#include "march/gas.hpp"
#include "march/python/WrapBase.hpp"

namespace march {

namespace python {

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasSolver
  : public WrapBase< WrapGasSolver<NDIM>, gas::Solver<NDIM>, std::shared_ptr<gas::Solver<NDIM>> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapGasSolver<NDIM>, gas::Solver<NDIM>, std::shared_ptr<gas::Solver<NDIM>> >;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;
    using block_type = typename wrapped_type::block_type;

    friend base_type;

    WrapGasSolver(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        (*this)
            .wrap_class_attributes()
            .wrap_constructors()
            .wrap_attributes()
            .wrap_methods()
            .wrap_legacy()
        ;
    }

private:

    wrapper_type & wrap_class_attributes() {
        this->m_cls.attr("ALMOST_ZERO") = double(wrapped_type::ALMOST_ZERO);
        this->m_cls.attr("neq") = NDIM + 2;
        this->m_cls.attr("_interface_init_") = std::make_tuple("cecnd");
        this->m_cls.attr("_solution_array_") = std::make_tuple();
        return *this;
    }

    wrapper_type & wrap_constructors() {
        namespace py = pybind11;
        return this->def(
            py::init([](
                py::object pyblock
              , real_type sigma0
              , real_type time
              , real_type time_increment
              , typename wrapped_type::int_type report_interval
              , py::kwargs
            ) {
                block_type * block = py::cast<block_type *>(pyblock.attr("_ustblk"));
                assert(block);
                std::shared_ptr<wrapped_type> svr = wrapped_type::construct(block->shared_from_this());
                for (auto bc : py::list(pyblock.attr("bclist"))) {
                    std::string name = py::str(bc.attr("__class__").attr("__name__").attr("lstrip")("GasPlus"));
                    BoundaryData * data = py::cast<BoundaryData *>(bc.attr("_data"));
                    std::unique_ptr<gas::TrimBase<NDIM>> trim;
                    if        ("Interface" == name) {
                        trim = make_unique<gas::TrimInterface<NDIM>>(*svr, *data);
                    } else if ("NoOp"      == name) {
                        trim = make_unique<gas::TrimNoOp<NDIM>>(*svr, *data);
                    } else if ("NonRefl"   == name) {
                        trim = make_unique<gas::TrimNonRefl<NDIM>>(*svr, *data);
                    } else if ("SlipWall"  == name) {
                        trim = make_unique<gas::TrimSlipWall<NDIM>>(*svr, *data);
                    } else if ("Inlet"     == name) {
                        trim = make_unique<gas::TrimInlet<NDIM>>(*svr, *data);
                    } else {
                        /* do nothing for now */ // throw std::runtime_error("BC type unknown");
                    }
                    svr->trims().push_back(std::move(trim));
                }
                svr->param().sigma0() = sigma0;
                svr->state().time = time;
                svr->state().time_increment = time_increment;
                svr->state().report_interval = report_interval;
                if (report_interval) { svr->make_qty(); }
                return svr;
            }),
            py::arg("block"), py::arg("sigma0"), py::arg("time"), py::arg("time_increment"), py::arg("report_interval")
        );
    }

    wrapper_type & wrap_attributes() {
        namespace py = pybind11;
        return (*this)
            .def_property_readonly("block", &wrapped_type::block)
            .def_property_readonly(
                "trims"
              , [](wrapped_type & self) -> std::vector<gas::TrimBase<NDIM>*> {
                    std::vector<gas::TrimBase<NDIM>*> ret;
                    for (auto & trim : self.trims()) {
                        ret.push_back(trim.get());
                    }
                    return ret;
                }
            )
            .def_property_readonly(
                "anchors"
              , [](wrapped_type & self) -> typename wrapped_type::anchor_chain_type & { return self.anchors(); }
              , py::return_value_policy::reference_internal
            )
            .def_property_readonly(
                "param"
              , [](wrapped_type & self) -> gas::Parameter & { return self.param(); }
              , py::return_value_policy::reference_internal
            )
            .def_property_readonly(
                "state"
              , [](wrapped_type & self) -> gas::State & { return self.state(); }
              , py::return_value_policy::reference_internal
            )
            .def_property_readonly(
                "sol"
              , [](wrapped_type & self) -> typename wrapped_type::solution_type & { return self.sol(); }
              , py::return_value_policy::reference_internal
            )
            .def_property_readonly("qty"
                                 , [](wrapped_type const & self) { return self.qty(); }
                                 , py::return_value_policy::reference_internal)
        ;
    }

    wrapper_type & wrap_methods() {
        namespace py = pybind11;
        return (*this)
            .def("make_qty"
               , &wrapped_type::make_qty, py::arg("throw_on_exist") = false
               , py::return_value_policy::reference_internal)
            .def("trim_do0", &wrapped_type::trim_do0)
            .def("trim_do1", &wrapped_type::trim_do1)
            /* FIXME: to be enabled */ //.def("init_solution", &wrapped_type::init_solution)
        ;
    }

    wrapper_type & wrap_legacy() {
        namespace py = pybind11;
        return (*this)
            .def_property_readonly("runanchors", [](py::object self) { return self.attr("anchors"); }) // compatibility
            .def_property_readonly("svrn", [](wrapped_type const &) { return py::none(); }) // TO BE UPDATED
            .def_property_readonly("nsvr", [](wrapped_type const &) { return py::none(); }) // TO BE UPDATED
            .def("provide", [](wrapped_type & self) { self.anchors().provide(); })
            .def("preloop", [](wrapped_type & self) { self.anchors().preloop(); })
            .def("postloop", [](wrapped_type & self) { self.anchors().postloop(); })
            .def("exhaust", [](wrapped_type & self) { self.anchors().exhaust(); })
            .def(
                "march"
              , [](wrapped_type & self
                 , real_type time_current
                 , real_type time_increment
                 , typename wrapped_type::int_type steps_run
                 , py::object worker
                ) {
                    using namespace pybind11::literals;
                    self.march(time_current, time_increment, steps_run);
                    py::list cfl;
                    cfl.append(self.state().cfl_min);
                    cfl.append(self.state().cfl_max);
                    cfl.append(self.state().cfl_nadjusted);
                    cfl.append(self.state().cfl_nadjusted_accumulated);
                    py::dict marchret = py::dict("cfl"_a = cfl);
                    if (worker.is(py::none())) { /* FIXME: message-pass marchret */ }
                    return marchret;
                }
              , py::arg("time_current")
              , py::arg("time_increment")
              , py::arg("steps_run")
              , py::arg("worker") = py::none()
            )
            .def(
                "init"
              , [](wrapped_type & self, py::kwargs const &) {
                    self.sol().arrays().so0c().fill(wrapped_type::ALMOST_ZERO);
                    self.sol().arrays().so0n().fill(wrapped_type::ALMOST_ZERO);
                    self.sol().arrays().so0t().fill(wrapped_type::ALMOST_ZERO);
                    self.sol().arrays().so1c().fill(wrapped_type::ALMOST_ZERO);
                    self.sol().arrays().so1n().fill(wrapped_type::ALMOST_ZERO);
                }
            )
            .def(
                "final"
              , [](wrapped_type &, py::kwargs const &) { /* do nothing */ }
            )
            .def(
                "apply_bc"
              , [](wrapped_type & self) {
                    self.trim_do0();
                    self.trim_do1();
                }
            )
        ;
    }

}; /* end class WrapGasSolver */

/* trampoline class */
class PythonAnchor : public gas::CommonAnchor
{

public:

    virtual ~PythonAnchor() {}

    template <size_t NDIM> PythonAnchor(ctor_passkey const & pk, gas::Solver<NDIM> & svr)
      : CommonAnchor(pk, svr) {}

    template <size_t NDIM>
    static std::shared_ptr<PythonAnchor> construct(gas::Solver<NDIM> & svr) {
        return std::make_shared<PythonAnchor>(ctor_passkey(), svr);
    }

#define DECL_MARCH_GAS_PYTHON_ANCHOR_METHOD(NAME) \
    void NAME() override { PYBIND11_OVERLOAD(void, CommonAnchor, NAME); }

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
  : public WrapBase< WrapGasCommonAnchor, gas::CommonAnchor, std::shared_ptr<gas::CommonAnchor>, PythonAnchor >
{

    friend base_type;

    WrapGasCommonAnchor(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;

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
  : public WrapBase< WrapGasAnchor<NDIM>, gas::Anchor<NDIM>, std::shared_ptr<gas::Anchor<NDIM>> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapGasAnchor<NDIM>, gas::Anchor<NDIM>, std::shared_ptr<gas::Anchor<NDIM>> >;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapGasAnchor(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;

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
                py::return_value_policy::reference_internal
            )
        ;

#undef DECL_MARCH_PYBIND_GAS_ANCHOR
    }

}; /* end class WrapGasAnchor */

/* This is to workaround https://github.com/pybind/pybind11/issues/1145.  The
 * lifecycle of the derived Python instances is kept in the manager. */
template< size_t NDIM >
class PythonAnchorManager : public gas::AnchorChain<NDIM>::LifeManager {

public:

    void append(pybind11::object const & pyobj) { m_list.push_back(pyobj); }

private:

    std::list<pybind11::object> m_list;

}; /* class PythonAnchorManager */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasAnchorChain
  : public WrapBase< WrapGasAnchorChain<NDIM>, gas::AnchorChain<NDIM> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapGasAnchorChain<NDIM>, gas::AnchorChain<NDIM> >;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapGasAnchorChain(pybind11::module & mod, const char * pyname, const char * clsdoc)
      : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
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
  : public WrapBase< WrapGasParameter, gas::Parameter >
{

    friend base_type;

    WrapGasParameter(pybind11::module & mod, const char * pyname, const char * clsdoc)
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
  : public WrapBase< WrapGasState, gas::State >
{

    friend base_type;

    WrapGasState(pybind11::module & mod, const char * pyname, const char * clsdoc)
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
            DECL_MARCH_PYBIND_GAS_STATE(gas::State::int_type, report_interval)
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
  : public WrapBase< WrapGasSolution<NDIM>, gas::Solution<NDIM> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapGasSolution<NDIM>, gas::Solution<NDIM> >;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapGasSolution(pybind11::module & mod, const char * pyname, const char * clsdoc)
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
  : public WrapBase< WrapGasQuantity<NDIM>, gas::Quantity<NDIM>, std::shared_ptr<gas::Quantity<NDIM>> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapGasQuantity<NDIM>, gas::Quantity<NDIM>, std::shared_ptr<gas::Quantity<NDIM>> >;
    using wrapped_type = typename base_type::wrapped_type;
    using solver_type = typename wrapped_type::solver_type;

    friend base_type;

    WrapGasQuantity(pybind11::module & mod, const char * pyname, const char * clsdoc)
      : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;

#define DECL_MARCH_PYBIND_GAS_QUANTITY_REAL(NAME) \
        .def_property( \
            #NAME, \
            [](wrapped_type const & self               ) { return self.NAME(); }, \
            [](wrapped_type       & self, real_type val) { return self.NAME() = val; } \
        )
// FIXME: change the properties to be like those of Solution
#define DECL_MARCH_PYBIND_GAS_QUANTITY_ARRAY(NAME, ARR) \
        .def_property( \
            #NAME, \
            [](wrapped_type & qty)                { return Table(qty.NAME()).ARR(); }, \
            [](wrapped_type & qty, py::array src) { Table::CopyInto(Table(qty.NAME()).ARR(), src); }, \
            #NAME " " #ARR " array")

        (*this)
            DECL_MARCH_PYBIND_GAS_QUANTITY_REAL(gasconst)
            DECL_MARCH_PYBIND_GAS_QUANTITY_REAL(schlieren_k)
            DECL_MARCH_PYBIND_GAS_QUANTITY_REAL(schlieren_k0)
            DECL_MARCH_PYBIND_GAS_QUANTITY_REAL(schlieren_k1)
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
#undef DECL_MARCH_PYBIND_GAS_QUANTITY_REAL
    }

}; /* end class WrapGasQuantity */

template< class TrimType, size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapGasTrimBase
  : public WrapBase< WrapGasTrimBase<TrimType, NDIM>, TrimType, std::unique_ptr<TrimType>, gas::TrimBase<NDIM> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapGasTrimBase<TrimType, NDIM>, TrimType, std::unique_ptr<TrimType>, gas::TrimBase<NDIM> >;
    using wrapped_type = typename base_type::wrapped_type;
    using solver_type = typename wrapped_type::solver_type;

    friend base_type;

    WrapGasTrimBase(pybind11::module & mod, const char * pyname, const char * clsdoc)
      : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
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

} /* end namespace python */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
