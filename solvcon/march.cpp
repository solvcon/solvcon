#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <memory>
#include <vector>
#include <algorithm>
#include <cstring>

#include "march/march.hpp"

namespace py = pybind11;

using namespace march;
using namespace march::mesh;

/**
 * Decorator of LookupTableCore to convert march::mesh::LookupTableCore to
 * pybind11::array and help array operations.
 */
class Table {

private:

    enum array_flavor { FULL = 0, GHOST = 1, BODY = 2 };

public:

    Table(LookupTableCore & table) : m_table(table) {}

    Table(const Table &) = delete;

    Table(Table &&) = delete;

    Table & operator=(const Table &) = delete;

    Table & operator=(Table &&) = delete;

    py::array full() { return from(FULL); }

    py::array ghost() { return from(GHOST); }

    py::array body() { return from(BODY); }

    static int NDIM(py::array arr) { return PyArray_NDIM((PyArrayObject *) arr.ptr()); }

    static npy_intp * DIMS(py::array arr) { return PyArray_DIMS((PyArrayObject *) arr.ptr()); }

    static char * BYTES(py::array arr) { return PyArray_BYTES((PyArrayObject *) arr.ptr()); }

    static int CopyInto(py::array dst, py::array src) {
        return PyArray_CopyInto((PyArrayObject *) dst.ptr(), (PyArrayObject *) src.ptr());
    }

private:

    /**
     * \param flavor The requested type of array.
     * \return       ndarray object as a view to the input table.
     */
    py::array from(array_flavor flavor) {
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
            py::pybind11_fail("NumPy: invalid array type");
        }

        py::object tmp(
            PyArray_NewFromDescr(
                &PyArray_Type, PyArray_DescrFromType(m_table.datatypeid()), m_table.ndim(),
                shape, strides, data, NPY_ARRAY_WRITEABLE, nullptr),
            false);
        if (!tmp) { py::pybind11_fail("NumPy: unable to create array view"); }

        py::object buffer = py::cast(m_table.buffer());
        py::array ret;
        if (PyArray_SetBaseObject((PyArrayObject *)tmp.ptr(), buffer.inc_ref().ptr()) == 0) {
            ret = tmp;
        }
        return ret;
    }

    LookupTableCore & m_table;

};

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_PLUGIN(march) {
    py::module mod("march", "libmarch wrapper");

    import_array1(nullptr); // or numpy c api segfault.

    py::class_< Buffer, std::shared_ptr<Buffer> >(mod, "Buffer", "Internal data buffer");

    py::class_< LookupTableCore >(mod, "Table", "Lookup table that allows ghost entity.")
        .def("__init__", [](py::args args, py::kwargs kwargs) {
            LookupTableCore & table = *(args[0].cast<LookupTableCore*>());

            std::vector<index_type> dims(args.size()-2);
            index_type nghost = args[1].cast<index_type>();
            index_type nbody = args[2].cast<index_type>();
            dims[0] = nghost + nbody;
            for (size_t it=1; it<dims.size(); ++it) {
                dims[it] = args[it+2].cast<index_type>();
            }

            PyArray_Descr * descr = nullptr;
            if (kwargs["dtype"]) {
                PyArray_DescrConverter(py::object(kwargs["dtype"]).ptr(), &descr);
            }
            if (nullptr == descr) {
                descr = PyArray_DescrFromType(NPY_INT);
            }
            DataTypeId dtid = static_cast<DataTypeId>(descr->type_num);
            Py_DECREF(descr);

            new (&table) LookupTableCore(nghost, nbody, dims, dtid);

            std::string creation("empty");
            if (kwargs["creation"]) {
                creation = kwargs["creation"].cast<std::string>();
            }
            if        ("zeros" == creation) {
                memset(table.buffer()->template data<char>(), 0, table.buffer()->nbyte());
            } else if ("empty" == creation) {
                // do nothing
            } else {
                throw py::value_error("invalid creation type");
            }
        })
        .def("__getattr__", [](LookupTableCore & tbl, py::object key) {
            return py::object(Table(tbl).full().attr(key));
        })
        .def_property_readonly("_nda", [](LookupTableCore & tbl) {
            return Table(tbl).full();
        })
        .def_property_readonly("nghost", &LookupTableCore::nghost)
        .def_property_readonly("nbody", &LookupTableCore::nbody)
        .def_property_readonly(
            "offset",
            [](LookupTableCore & tbl){ return tbl.nghost() * tbl.ncolumn(); },
            "Element offset from the head of the ndarray to where the body starts.")
        .def_property_readonly(
            "_ghostaddr",
            [](LookupTableCore & tbl) { return (Py_intptr_t) Table::BYTES(Table(tbl).full()); })
        .def_property_readonly(
            "_bodyaddr",
            [](LookupTableCore & tbl) { return (Py_intptr_t) Table::BYTES(Table(tbl).body()); })
        .def_property(
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
        .def_property_readonly(
            "_ghostpart",
            [](LookupTableCore & tbl) { return Table(tbl).ghost(); },
            "Ghost-part array without setter.")
        .def_property(
            "B",
            [](LookupTableCore & tbl) { return Table(tbl).body(); },
            [](LookupTableCore & tbl, py::array src) { Table::CopyInto(Table(tbl).body(), src); },
            "Body-part array.")
        .def_property_readonly(
            "_bodypart",
            [](LookupTableCore & tbl) { return Table(tbl).body(); },
            "Body-part array without setter.")
        .def("__getstate__", [](LookupTableCore & tbl) {
            return py::make_tuple(tbl.nghost(), tbl.nbody(), tbl.dims(), (long)tbl.datatypeid(), Table(tbl).full());
        })
        .def("__setstate__", [](LookupTableCore & tbl, py::tuple tpl) {
            if (tpl.size() != 5) { throw std::runtime_error("Invalid state for Table (LookupTableCore)!"); }
            index_type nghost = tpl[0].cast<index_type>();
            index_type nbody  = tpl[1].cast<index_type>();
            std::vector<index_type> dims = tpl[2].cast<std::vector<index_type>>();
            DataTypeId datatypeid = static_cast<DataTypeId>(tpl[3].cast<long>());
            py::array src = tpl[4].cast<py::array>();
            new (&tbl) LookupTableCore(nghost, nbody, dims, datatypeid);
            Table::CopyInto(Table(tbl).full(), src);
        });
    ;

    py::class_< BoundaryData >(mod, "BoundaryData", "Data of a boundary condition.")
        .def(py::init<index_type>())
        .def_property_readonly_static("BFREL", [](py::object /* self */) { return BoundaryData::BFREL; })
        .def_property(
            "facn",
            [](BoundaryData & bnd) {
                if (0 == bnd.facn().nbyte()) {
                    npy_intp shape[2] = {0, BoundaryData::BFREL};
                    return py::array(
                        PyArray_NewFromDescr(
                            &PyArray_Type, PyArray_DescrFromType(bnd.facn().datatypeid()), 2 /* nd */,
                            shape, nullptr /* strides */, nullptr /* data */, 0 /* flags */, nullptr),
                        false);
                } else {
                    return Table(bnd.facn()).body();
                }
            },
            [](BoundaryData & bnd, py::array src) {
                if (Table::NDIM(src) != 2) {
                    throw py::index_error("BoundaryData.facn input array dimension isn't 2");
                }
                if (Table::DIMS(src)[1] != BoundaryData::BFREL) {
                    throw py::index_error("BoundaryData.facn second axis mismatch");
                }
                index_type nface = Table::DIMS(src)[0];
                if (nface != bnd.facn().nbody()) {
                    bnd.facn() = std::remove_reference<decltype(bnd.facn())>::type(0, nface);
                }
                if (0 != bnd.facn().nbyte()) {
                    Table::CopyInto(Table(bnd.facn()).body(), src);
                }
            },
            "List of faces."
        )
        .def_property(
            "values",
            [](BoundaryData & bnd) {
                if (0 == bnd.values().nbyte()) {
                    npy_intp shape[2] = {0, bnd.nvalue()};
                    return py::array(
                        PyArray_NewFromDescr(
                            &PyArray_Type, PyArray_DescrFromType(bnd.values().datatypeid()), 2 /* nd */,
                            shape, nullptr /* strides */, nullptr /* data */, 0 /* flags */, nullptr),
                        false);
                } else {
                    return Table(bnd.values()).body();
                }
            },
            [](BoundaryData & bnd, py::array src) {
                if (Table::NDIM(src) != 2) {
                    throw py::index_error("BoundaryData.values input array dimension isn't 2");
                }
                index_type nface = Table::DIMS(src)[0];
                index_type nvalue = Table::DIMS(src)[1];
                if (nface != bnd.values().nbody() || nvalue != bnd.values().ncolumn()) {
                    bnd.values() = std::remove_reference<decltype(bnd.values())>::type(0, nface, {nface, nvalue}, type_to<real_type>::id);
                }
                if (0 != bnd.values().nbyte()) {
                    Table::CopyInto(Table(bnd.values()).body(), src);
                }
            },
            "Attached (specified) value for each boundary face."
        )
        .def("good_shape", &BoundaryData::good_shape)
    ;

    return mod.ptr();
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
