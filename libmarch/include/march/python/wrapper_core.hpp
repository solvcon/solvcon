#pragma once

/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstring>

#include "march.hpp"
#include "march/python/WrapBase.hpp"

namespace march {

namespace python {

class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapBuffer
  : public WrapBase< WrapBuffer, Buffer, std::shared_ptr<Buffer> >
{

    friend base_type;

    WrapBuffer(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
    }

}; /* end class WrapBuffer */

class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapLookupTableCore
  : public WrapBase< WrapLookupTableCore, LookupTableCore >
{

    friend base_type;

    WrapLookupTableCore(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
        (*this)
            .init()
            .def_property_readonly("nghost", &LookupTableCore::nghost)
            .def_property_readonly("nbody", &LookupTableCore::nbody)
            .def_property_readonly("ncolumn", &LookupTableCore::ncolumn)
            .array_readwrite()
            .array_readonly()
            .pickle()
            .address()
            .def("__getattr__", [](LookupTableCore & tbl, py::object key) {
                return py::object(Table(tbl).full().attr(key));
            })
            .def_property_readonly("_nda", [](LookupTableCore & tbl) {
                return Table(tbl).full();
            })
        ;
    }

    wrapper_type & init() {
        namespace py = pybind11;
        return def(py::init([](py::args args, py::kwargs kwargs) {
            std::vector<index_type> dims(args.size()-1);
            index_type nghost = args[0].cast<index_type>();
            index_type nbody = args[1].cast<index_type>();
            dims[0] = nghost + nbody;
            for (size_t it=1; it<dims.size(); ++it) {
                dims[it] = args[it+1].cast<index_type>();
            }

            PyArray_Descr * descr = nullptr;
            if (kwargs && kwargs.contains("dtype")) {
                PyArray_DescrConverter(py::object(kwargs["dtype"]).ptr(), &descr);
            }
            if (nullptr == descr) {
                descr = PyArray_DescrFromType(NPY_INT);
            }
            DataTypeId dtid = static_cast<DataTypeId>(descr->type_num);
            Py_DECREF(descr);

            LookupTableCore table(nghost, nbody, dims, dtid);

            std::string creation("empty");
            if (kwargs && kwargs.contains("creation")) {
                creation = kwargs["creation"].cast<std::string>();
            }
            if        ("zeros" == creation) {
                memset(table.buffer()->template data<char>(), 0, table.buffer()->nbyte());
            } else if ("empty" == creation) {
                // do nothing
            } else {
                throw py::value_error("invalid creation type");
            }

            return table;
        }));
    }

    wrapper_type & array_readwrite() {
        namespace py = pybind11;
        return def_property(
            "F",
            [](LookupTableCore & tbl) { return Table(tbl).full(); },
            [](LookupTableCore & tbl, py::array src) { Table::CopyInto(Table(tbl).full(), src); },
            "Full array.")
        .def_property(
            "G",
            [](LookupTableCore & tbl) { return Table(tbl).ghost(); },
            [](LookupTableCore & tbl, py::array src) {
                if (tbl.nghost()) {
                    Table::CopyInto(Table(tbl).ghost(), src);
                } else {
                    throw py::index_error("ghost is zero");
                }
            },
            "Ghost-part array.")
        .def_property(
            "B",
            [](LookupTableCore & tbl) { return Table(tbl).body(); },
            [](LookupTableCore & tbl, py::array src) { Table::CopyInto(Table(tbl).body(), src); },
            "Body-part array.")
        ;
    }

    wrapper_type & array_readonly() {
        return def_property_readonly(
            "_ghostpart",
            [](LookupTableCore & tbl) { return Table(tbl).ghost(); },
            "Ghost-part array without setter.")
        .def_property_readonly(
            "_bodypart",
            [](LookupTableCore & tbl) { return Table(tbl).body(); },
            "Body-part array without setter.")
        ;
    }

    wrapper_type & pickle() {
        namespace py = pybind11;
        return def(py::pickle(
            [](LookupTableCore & tbl){
                return py::make_tuple(tbl.nghost(), tbl.nbody(), tbl.dims(), (long)tbl.datatypeid(), Table(tbl).full());
            },
            [](py::tuple tpl){
                if (tpl.size() != 5) { throw std::runtime_error("Invalid state for Table (LookupTableCore)!"); }
                index_type nghost = tpl[0].cast<index_type>();
                index_type nbody  = tpl[1].cast<index_type>();
                std::vector<index_type> dims = tpl[2].cast<std::vector<index_type>>();
                DataTypeId datatypeid = static_cast<DataTypeId>(tpl[3].cast<long>());
                py::array src = tpl[4].cast<py::array>();
                LookupTableCore tbl(nghost, nbody, dims, datatypeid);
                Table::CopyInto(Table(tbl).full(), src);
                return tbl;
            }
        ));
    }

    wrapper_type & address() {
        return def_property_readonly(
            "offset",
            [](LookupTableCore & tbl) { return tbl.nghost() * tbl.ncolumn(); },
            "Element offset from the head of the ndarray to where the body starts.")
        .def_property_readonly(
            "_ghostaddr",
            [](LookupTableCore & tbl) { return (Py_intptr_t) Table::BYTES(Table(tbl).full()); })
        .def_property_readonly(
            "_bodyaddr",
            [](LookupTableCore & tbl) { return (Py_intptr_t) Table::BYTES(Table(tbl).body()); })
        ;
    }

}; /* end class WrapLookupTableCore */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapVector
  : public WrapBase< WrapVector<NDIM>, Vector<NDIM> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapVector<NDIM>, Vector<NDIM> >;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapVector(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;

        wrapped_type dummy;

#define MARCH_PYBIND_VECTOR_SCALAR_OP(PYNAME, CXXOP) \
            .def("__i" #PYNAME "__", [](wrapped_type & self, real_type v) { self CXXOP v; return self; }) \
            .def("__" #PYNAME "__", [](wrapped_type & self, real_type v) { auto ret(self); ret CXXOP v; return ret; })

#define MARCH_PYBIND_VECTOR_VECTOR_OP(PYNAME, CXXOP) \
            .def("__i" #PYNAME "__", [](wrapped_type & self, wrapped_type const & v) { self CXXOP v; return self; }) \
            .def("__" #PYNAME "__", [](wrapped_type & self, wrapped_type const & v) { auto ret(self); ret CXXOP v; return ret; })

        (*this)
            .def(py::init([]() { return wrapped_type(); }))
            .def(py::init([](wrapped_type & other) { return wrapped_type(other); }))
            .add_element_init(dummy)
            .def("repr", &wrapped_type::repr, py::arg("indent")=0, py::arg("precision")=0)
            .def("__repr__", [](wrapped_type & self){ return self.repr(); })
            .def("__eq__", &wrapped_type::operator==)
            .def(
                "__hash__",
                [](wrapped_type const & self) {
                    py::list tmp;
                    for (size_t it=0; it<self.size(); ++it) {
                        tmp.append(self[it]);
                    }
                    return py::hash(tmp);
                }
            )
            .def("is_close_to", &wrapped_type::is_close_to)
            .def("__len__", &wrapped_type::size)
            .def(
                "__getitem__",
                [](wrapped_type & self, index_type i) { return self.at(i); },
                py::return_value_policy::copy
            )
            .def(
                "__setitem__",
                [](wrapped_type & self, index_type i, real_type v) { self.at(i) = v; }
            )
            MARCH_PYBIND_VECTOR_VECTOR_OP(add, +=)
            MARCH_PYBIND_VECTOR_VECTOR_OP(sub, -=)
            MARCH_PYBIND_VECTOR_SCALAR_OP(add, +=)
            MARCH_PYBIND_VECTOR_SCALAR_OP(sub, -=)
            MARCH_PYBIND_VECTOR_SCALAR_OP(mul, *=)
            MARCH_PYBIND_VECTOR_SCALAR_OP(div, /=)
            MARCH_PYBIND_VECTOR_SCALAR_OP(truediv, /=)
            // FIXME: add other functions
        ;

#undef MARCH_PYBIND_VECTOR_SCALAR_OP
#undef MARCH_PYBIND_VECTOR_VECTOR_OP
    }

    wrapper_type & add_element_init(Vector<3> const &) {
        namespace py = pybind11;
        (*this)
            .def(py::init([](real_type v0, real_type v1, real_type v2) { return wrapped_type(v0, v1, v2); }))
        ;
        return *this;
    }

    wrapper_type & add_element_init(Vector<2> const &) {
        namespace py = pybind11;
        (*this)
            .def(py::init([](real_type v0, real_type v1) { return wrapped_type(v0, v1); }))
        ;
        return *this;
    }

}; /* end class WrapVector */

} /* end namespace python */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
