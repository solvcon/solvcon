#pragma once

/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <memory>
#include <type_traits>

PYBIND11_DECLARE_HOLDER_TYPE(T, std::unique_ptr<T>);
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace march {
namespace python {
template <typename T> class unique_ptr_holding_block {
    std::unique_ptr<T> ptr;
    std::shared_ptr<typename T::block_type> block_ptr;
public:
    unique_ptr_holding_block(T *p) : ptr(p), block_ptr(p->block().shared_from_this()) {}
    T *get() { return ptr.get(); }
}; /* end class unique_ptr_holding_block */
} /* end namespace python */
} /* end namespace march */
PYBIND11_DECLARE_HOLDER_TYPE(T, march::python::unique_ptr_holding_block<T>);

#ifdef __GNUG__
#  define MARCH_PYTHON_WRAPPER_VISIBILITY __attribute__((visibility("hidden")))
#else
#  define MARCH_PYTHON_WRAPPER_VISIBILITY
#endif

namespace march {

namespace python {

/**
 * Helper template for pybind11 class wrappers.
 */
template< class Wrapper, class Wrapped, class Holder = std::unique_ptr<Wrapped>, class WrappedBase = Wrapped >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapBase {

public:

    using wrapper_type = Wrapper;
    using wrapped_type = Wrapped;
    using wrapped_base_type = WrappedBase;
    using holder_type = Holder;
    using base_type = WrapBase< wrapper_type, wrapped_type, holder_type, wrapped_base_type >;
    using class_ = typename std::conditional<
        std::is_same< Wrapped, WrappedBase >::value
      , pybind11::class_< wrapped_type, holder_type >
      , pybind11::class_< wrapped_type, wrapped_base_type, holder_type >
    >::type;

    static wrapper_type & commit(pybind11::module & mod, const char * pyname, const char * clsdoc) {
        static wrapper_type derived(mod, pyname, clsdoc);
        return derived;
    }

    WrapBase() = delete;
    WrapBase(WrapBase const & ) = default;
    WrapBase(WrapBase       &&) = delete;
    WrapBase & operator=(WrapBase const & ) = default;
    WrapBase & operator=(WrapBase       &&) = delete;

#define DECL_MARCH_PYBIND_CLASS_METHOD(METHOD) \
    template< class... Args > \
    wrapper_type & METHOD(Args&&... args) { \
        m_cls.METHOD(std::forward<Args>(args)...); \
        return *static_cast<wrapper_type*>(this); \
    }

    DECL_MARCH_PYBIND_CLASS_METHOD(def)
    DECL_MARCH_PYBIND_CLASS_METHOD(def_readwrite)
    DECL_MARCH_PYBIND_CLASS_METHOD(def_property)
    DECL_MARCH_PYBIND_CLASS_METHOD(def_property_readonly)
    DECL_MARCH_PYBIND_CLASS_METHOD(def_property_readonly_static)

#undef DECL_MARCH_PYBIND_CLASS_METHOD

protected:

    WrapBase(pybind11::module & mod, const char * pyname, const char * clsdoc)
      : m_cls(mod, pyname, clsdoc)
    {
        expose_instance_count(typename std::is_base_of<InstanceCounter<wrapped_type>, wrapped_type>::type {});
    }

    class_ m_cls;

private:

    void expose_instance_count(std::true_type const &) {
        m_cls
            .def_property_readonly_static(
                "active_instance_count",
                [](pybind11::object const & /* self */) { return wrapped_type::active_instance_count(); }
            )
        ;
    }
    void expose_instance_count(std::false_type const &) {}

}; /* end class WrapBase */

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
            initialize_march(topmod);
            initialize_march_gas(topmod);
        }
    }

    bool is_initialized() const { return m_initialized; }

private:

    PyObject * initialize_march(pybind11::module & marchmod);

    PyObject * initialize_march_gas(pybind11::module & marchmod);

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
