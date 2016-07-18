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
 * Helper class to convert march::mesh::LookupTableCore to pybind11::array.
 */
class make_array {

public:

    enum array_flavor { FULL = 0, GHOST = 1, BODY = 2 };

    make_array(array_flavor flavor) : m_flavor(flavor) {}

    make_array() = delete;
    make_array(const make_array &) = delete;
    make_array(make_array &&) = delete;
    make_array & operator=(const make_array &) = delete;
    make_array & operator=(make_array &&) = delete;

    static py::array full_from(LookupTableCore & tbl) {
        static make_array worker(FULL);
        return worker.from(tbl);
    }

    static py::array ghost_from(LookupTableCore & tbl) {
        static make_array worker(GHOST);
        return worker.from(tbl);
    }

    static py::array body_from(LookupTableCore & tbl) {
        static make_array worker(BODY);
        return worker.from(tbl);
    }

private:

    /**
     * \param tbl The input LookupTableCore.
     * \return    ndarray object as a view to the input table.
     */
    py::array from(LookupTableCore & tbl) const {
        npy_intp shape[tbl.ndim()];
        std::copy(tbl.dims().begin(), tbl.dims().end(), shape);

        npy_intp strides[tbl.ndim()];
        strides[tbl.ndim()-1] = tbl.elsize();
        for (ssize_t it = tbl.ndim()-2; it >= 0; --it) {
            strides[it] = shape[it+1] * strides[it+1];
        }

        void * data = nullptr;
        if        (FULL == m_flavor) {
            data = tbl.data();
        } else if (GHOST == m_flavor) {
            shape[0] = tbl.nghost();
            strides[0] = -strides[0];
            data = tbl.nghost() > 0 ? tbl.row(-1) : tbl.row(0);
        } else if (BODY == m_flavor) {
            shape[0] = tbl.nbody();
            data = tbl.row(0);
        } else {
            py::pybind11_fail("NumPy: invalid array type");
        }

        py::object tmp(
            PyArray_NewFromDescr(
                &PyArray_Type, PyArray_DescrFromType(tbl.datatypeid()), tbl.ndim(),
                shape, strides, data, NPY_ARRAY_WRITEABLE, nullptr),
            false);
        if (!tmp) { py::pybind11_fail("NumPy: unable to create array view"); }

        py::object buffer = py::cast(tbl.buffer());
        py::array ret;
        if (PyArray_SetBaseObject((PyArrayObject *)tmp.ptr(), buffer.inc_ref().ptr()) == 0) {
            ret = tmp;
        }
        return ret;
    }

    array_flavor m_flavor;

}; /* end class make_array */

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
            return py::object(make_array::full_from(tbl).attr(key));
        })
        .def_property_readonly("_nda", [](LookupTableCore & tbl) {
            return make_array::full_from(tbl);
        })
        .def_property_readonly("nghost", &LookupTableCore::nghost)
        .def_property_readonly("nbody", &LookupTableCore::nbody)
        .def_property_readonly(
            "offset",
            [](LookupTableCore & tbl){ return tbl.nghost() * tbl.ncolumn(); },
            "Element offset from the head of the ndarray to where the body starts.")
        .def_property_readonly(
            "_ghostaddr",
            [](LookupTableCore & tbl) {
                return (Py_intptr_t) PyArray_BYTES((PyArrayObject *) make_array::full_from(tbl).ptr());
            })
        .def_property_readonly(
            "_bodyaddr",
            [](LookupTableCore & tbl) {
                return (Py_intptr_t) PyArray_BYTES((PyArrayObject *) make_array::body_from(tbl).ptr());
            })
        .def_property(
            "F",
            [](LookupTableCore & tbl) { return make_array::full_from(tbl); },
            [](LookupTableCore & tbl, py::array src) {
                PyArray_CopyInto((PyArrayObject *) make_array::full_from(tbl).ptr(), (PyArrayObject *) src.ptr());
            },
            "Full array.")
        .def_property(
            "G",
            [](LookupTableCore & tbl) { return make_array::ghost_from(tbl); },
            [](LookupTableCore & tbl, py::array src) {
                if (tbl.nghost()) {
                    PyArray_CopyInto((PyArrayObject *) make_array::ghost_from(tbl).ptr(), (PyArrayObject *) src.ptr());
                } else {
                    throw py::index_error("ghost is zero");
                }
            },
            "Ghost-part array.")
        .def_property_readonly(
            "_ghostpart",
            [](LookupTableCore & tbl) { return make_array::ghost_from(tbl); },
            "Ghost-part array without setter.")
        .def_property(
            "B",
            [](LookupTableCore & tbl) { return make_array::body_from(tbl); },
            [](LookupTableCore & tbl, py::array src) {
                PyArray_CopyInto((PyArrayObject *) make_array::body_from(tbl).ptr(), (PyArrayObject *) src.ptr());
            },
            "Body-part array.")
        .def_property_readonly(
            "_bodypart",
            [](LookupTableCore & tbl) { return make_array::body_from(tbl); },
            "Body-part array without setter.")
    ;

    py::class_< BoundaryData >(mod, "BoundaryData", "Data of a boundary condition.")
        .def("__init__", [](BoundaryData *bnd) { new (bnd) BoundaryData(); })
        .def("__init__", [](BoundaryData *bnd, index_type nvalue) { new (bnd) BoundaryData(nvalue); })
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
                    return make_array::body_from(bnd.facn());
                }
            },
            [](BoundaryData & bnd, py::array src) {
                if (PyArray_NDIM((PyArrayObject *)src.ptr()) != 2) {
                    throw py::index_error("BoundaryData.facn input array dimension isn't 2");
                }
                if (PyArray_DIMS((PyArrayObject *)src.ptr())[1] != BoundaryData::BFREL) {
                    throw py::index_error("BoundaryData.facn second axis mismatch");
                }
                index_type nface = PyArray_DIMS((PyArrayObject *)src.ptr())[0];
                if (nface != bnd.facn().nbody()) {
                    bnd.facn() = std::remove_reference<decltype(bnd.facn())>::type(0, nface);
                }
                if (0 != bnd.facn().nbyte()) {
                    PyArray_CopyInto((PyArrayObject *) make_array::body_from(bnd.facn()).ptr(), (PyArrayObject *) src.ptr());
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
                    return make_array::body_from(bnd.values());
                }
            },
            [](BoundaryData & bnd, py::array src) {
                if (PyArray_NDIM((PyArrayObject *)src.ptr()) != 2) {
                    throw py::index_error("BoundaryData.values input array dimension isn't 2");
                }
                index_type nface = PyArray_DIMS((PyArrayObject *)src.ptr())[0];
                index_type nvalue = PyArray_DIMS((PyArrayObject *)src.ptr())[1];
                if (nface != bnd.values().nbody() || nvalue != bnd.values().ncolumn()) {
                    bnd.values() = std::remove_reference<decltype(bnd.values())>::type(0, nface, {nface, nvalue}, type_to<real_type>::id);
                }
                if (0 != bnd.values().nbyte()) {
                    PyArray_CopyInto((PyArrayObject *) make_array::body_from(bnd.values()).ptr(), (PyArrayObject *) src.ptr());
                }
            },
            "Attached (specified) value for each boundary face."
        )
        .def("good_shape", &BoundaryData::good_shape)
    ;

    return mod.ptr();
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
