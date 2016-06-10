#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <memory>
#include <vector>
#include <algorithm>

#include "march/march.hpp"

namespace py = pybind11;

using namespace march;
using namespace march::mesh;

/**
 * Python wrapper for march::mesh::LookupTableCore.
 */
class Table {

private:

    py::array m_nda;

    std::unique_ptr<LookupTableCore> m_table;

public:

    Table(
        index_type nghost
      , index_type nbody
      , const std::vector<index_type> & dims
      , PyArray_Descr * descr
      , bool do_zeros
    ) {
        npy_intp shape[dims.size()];
        std::copy_n(dims.begin(), dims.size(), shape);
        PyArrayObject * arrptr = nullptr;
        if (do_zeros) {
            arrptr = (PyArrayObject *) PyArray_Zeros(dims.size(), shape, descr, 0);
        } else {
            arrptr = (PyArrayObject *) PyArray_Empty(dims.size(), shape, descr, 0);
        }
        m_nda = py::array((PyObject *) arrptr, false);
        m_table.reset(new LookupTableCore(nghost, nbody, dims, PyArray_ITEMSIZE(arrptr), PyArray_BYTES(arrptr)));
        if (static_cast<size_t>(PyArray_NBYTES(arrptr)) != m_table->nbyte()) {
            throw py::value_error("nbyte mismatch");
        }
    }

    index_type nghost() const { return m_table->nghost(); }

    index_type nbody() const { return m_table->nbody(); }

    index_type offset() const { return m_table->nghost() * m_table->ncolumn(); }

    py::array get_full() const {
        PyArrayObject * arr = (PyArrayObject *) m_nda.ptr();
        return get_view(PyArray_SHAPE(arr)[0], PyArray_STRIDES(arr)[0], (void *) PyArray_BYTES(arr));
    }

    py::array get_ghost() const {
        PyArrayObject * arr = (PyArrayObject *) m_nda.ptr();
        return get_view(m_table->nghost(), -PyArray_STRIDES(arr)[0],
                        (void *) (m_table->nghost() > 0 ? m_table->row(-1) : m_table->row(0)));
    }

    py::array get_body() const {
        PyArrayObject * arr = (PyArrayObject *) m_nda.ptr();
        return get_view(m_table->nbody(), PyArray_STRIDES(arr)[0], (void *) m_table->row(0));
    }

    py::array nda() const { return m_nda; }

private:

    py::array get_view(npy_intp shape0, npy_intp strides0, void * data) const {
        PyArrayObject * arr = (PyArrayObject *) m_nda.ptr();
        npy_intp shape[PyArray_NDIM(arr)];
        npy_intp strides[PyArray_NDIM(arr)];
        shape[0] = shape0;
        strides[0] = strides0;
        std::copy_n(PyArray_SHAPE(arr)+1, PyArray_NDIM(arr)-1, shape+1);
        std::copy_n(PyArray_STRIDES(arr)+1, PyArray_NDIM(arr)-1, strides+1);
        Py_INCREF(PyArray_DESCR(arr));
        py::object tmp(
            PyArray_NewFromDescr(&PyArray_Type, PyArray_DESCR(arr),
                                 PyArray_NDIM(arr), shape, strides,
                                 data, PyArray_FLAGS(arr), nullptr),
            false);
        if (!tmp) { py::pybind11_fail("NumPy: unable to create array view!"); }
        py::array ret;
        m_nda.inc_ref();
        if (PyArray_SetBaseObject((PyArrayObject *)tmp.ptr(), (PyObject *)m_nda.ptr()) == 0) {
            ret = tmp;
        }
        return ret;
    }

}; /* end class Table */

PYBIND11_PLUGIN(march) {
    py::module mod("march", "pybind11 example plugin");

    import_array1(nullptr); // or numpy c api segfault.

    py::class_< Table >(mod, "Table", "Lookup table that allows ghost entity.")
        .def("__init__", [](py::args args, py::kwargs kwargs) {
            Table & table = *(args[0].cast<Table*>());

            std::vector<index_type> dims(args.size()-2);
            index_type nghost = args[1].cast<index_type>();
            index_type nbody = args[2].cast<index_type>();
            dims[0] = nghost + nbody;
            for (size_t it=1; it<dims.size(); ++it) {
                dims[it] = args[it+2].cast<index_type>();
            }

            std::string creation("empty");
            if (kwargs["creation"]) {
                creation = kwargs["creation"].cast<std::string>();
            }
            bool do_zeros;
            if        ("zeros" == creation) {
                do_zeros = true;
            } else if ("empty" == creation) {
                do_zeros = false;
            } else {
                throw py::value_error("invalid creation type");
            }

            PyArray_Descr * descr = nullptr;
            if (kwargs["dtype"]) {
                PyArray_DescrConverter(py::object(kwargs["dtype"]).ptr(), &descr);
            }
            if (nullptr == descr) {
                descr = PyArray_DescrFromType(NPY_INT);
            }

            new (&table) Table(nghost, nbody, dims, descr, do_zeros);
        })
        .def("__getattr__", [] (Table & tbl, py::object key) {
            return py::object(tbl.nda().attr(key));
        })
        .def_property_readonly("_nda", &Table::nda)
        .def_property_readonly("nghost", &Table::nghost)
        .def_property_readonly("nbody", &Table::nbody)
        .def_property_readonly("offset",  &Table::offset,
                               "Element offset from the head of the ndarray to where the body starts.")
        .def_property_readonly(
            "_ghostaddr",
            [](Table & tbl) {
                return (Py_intptr_t) PyArray_BYTES((PyArrayObject *) tbl.get_full().ptr());
            })
        .def_property_readonly(
            "_bodyaddr",
            [](Table & tbl) {
                return (Py_intptr_t) PyArray_BYTES((PyArrayObject *) tbl.get_body().ptr());
            })
        .def_property(
            "F",
            &Table::get_full,
            [](Table & tbl, py::array src) {
                PyArray_CopyInto((PyArrayObject *) tbl.get_full().ptr(), (PyArrayObject *) src.ptr());
            },
            "Full array.")
        .def_property(
            "G",
            &Table::get_ghost,
            [](Table & tbl, py::array src) {
                if (tbl.nghost()) {
                    PyArray_CopyInto((PyArrayObject *) tbl.get_ghost().ptr(), (PyArrayObject *) src.ptr());
                } else {
                    throw py::index_error("ghost is zero");
                }
            },
            "Ghost-part array.")
        .def_property_readonly("_ghostpart", &Table::get_ghost, "Ghost-part array without setter.")
        .def_property(
            "B",
            &Table::get_body,
            [](Table & tbl, py::array src) {
                PyArray_CopyInto((PyArrayObject *) tbl.get_body().ptr(), (PyArrayObject *) src.ptr());
            },
            "Body-part array.")
        .def_property_readonly("_bodypart", &Table::get_body, "Body-part array without setter.")
    ;

    return mod.ptr();
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
