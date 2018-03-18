#pragma once

/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <map>
#include <string>
#include <functional>

#include <pybind11/pybind11.h>

#include "march/python/WrapBase.hpp"

PYBIND11_DECLARE_HOLDER_TYPE(T, std::unique_ptr<T>);
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace march {

namespace python {

/**
 * Decorator of LookupTableCore to convert march::LookupTableCore to
 * pybind11::array and help array operations.
 */
class Table {

private:

    enum array_flavor { FULL = 0, GHOST = 1, BODY = 2 };

public:

    Table(LookupTableCore & table) : m_table(table) {}

    Table(Table const & ) = delete;
    Table(Table       &&) = delete;
    Table & operator=(Table const & ) = delete;
    Table & operator=(Table       &&) = delete;

    pybind11::array full () { return from(FULL ); }
    pybind11::array ghost() { return from(GHOST); }
    pybind11::array body () { return from(BODY ); }

    static int        NDIM (pybind11::array arr) { return PyArray_NDIM ((PyArrayObject *) arr.ptr()); }
    static npy_intp * DIMS (pybind11::array arr) { return PyArray_DIMS ((PyArrayObject *) arr.ptr()); }
    static char *     BYTES(pybind11::array arr) { return PyArray_BYTES((PyArrayObject *) arr.ptr()); }
    static void CopyInto(pybind11::array dst, pybind11::array src) {
        if (0 != PyArray_SIZE((PyArrayObject *) dst.ptr()) && 0 != PyArray_SIZE((PyArrayObject *) src.ptr())) {
            int ret = PyArray_CopyInto((PyArrayObject *) dst.ptr(), (PyArrayObject *) src.ptr());
            if (-1 == ret) { throw pybind11::error_already_set(); }
        }
    }

private:

    /**
     * \param flavor The requested type of array.
     * \return       ndarray object as a view to the input table.
     */
    pybind11::array from(array_flavor flavor) {
        npy_intp shape[m_table.ndim()];
        std::copy(m_table.dims().begin(), m_table.dims().end(), shape);

        npy_intp strides[m_table.ndim()];
        strides[m_table.ndim()-1] = m_table.elsize();
        for (ssize_t it = m_table.ndim()-2; it >= 0; --it) {
            strides[it] = shape[it+1] * strides[it+1];
        }

        void * data = nullptr;
        if        (FULL == flavor) {
            data = m_table.data();
        } else if (GHOST == flavor) {
            shape[0] = m_table.nghost();
            strides[0] = -strides[0];
            data = m_table.nghost() > 0 ? m_table.row(-1) : m_table.row(0);
        } else if (BODY == flavor) {
            shape[0] = m_table.nbody();
            data = m_table.row(0);
        } else {
            pybind11::pybind11_fail("NumPy: invalid array type");
        }

        pybind11::object tmp = pybind11::reinterpret_steal<pybind11::object>(
            PyArray_NewFromDescr(
                &PyArray_Type, PyArray_DescrFromType(m_table.datatypeid()), m_table.ndim(),
                shape, strides, data, NPY_ARRAY_WRITEABLE, nullptr));
        if (!tmp) { pybind11::pybind11_fail("NumPy: unable to create array view"); }

        pybind11::object buffer = pybind11::cast(m_table.buffer());
        pybind11::array ret;
        if (PyArray_SetBaseObject((PyArrayObject *)tmp.ptr(), buffer.inc_ref().ptr()) == 0) {
            ret = tmp;
        }
        return ret;
    }

    LookupTableCore & m_table;

}; /* end class Table */

class ModuleInitializer {

public:

    static ModuleInitializer & get_instance() {
        static ModuleInitializer inst;
        return inst;
    }

    void initialize(pybind11::module & topmod) {
        if (!m_initialized) {
            initialize_top(topmod);
            initialize_gas(topmod);
        }
    }

    bool is_initialized() const { return m_initialized; }

private:

    PyObject * initialize_top(pybind11::module & topmod);

    PyObject * initialize_gas(pybind11::module & upmod);

    ModuleInitializer() = default;
    ModuleInitializer(ModuleInitializer const & ) = delete;
    ModuleInitializer(ModuleInitializer       &&) = delete;
    ModuleInitializer & operator=(ModuleInitializer const & ) = delete;
    ModuleInitializer & operator=(ModuleInitializer       &&) = delete;

    bool m_initialized = false;

}; /* end class ModuleInitializer */

} /* end namespace python */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
