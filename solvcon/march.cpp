#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <utility>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstring>

#include "march/march.hpp"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

using namespace march;

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

    py::array full () { return from(FULL ); }
    py::array ghost() { return from(GHOST); }
    py::array body () { return from(BODY ); }

    static int        NDIM (py::array arr) { return PyArray_NDIM ((PyArrayObject *) arr.ptr()); }
    static npy_intp * DIMS (py::array arr) { return PyArray_DIMS ((PyArrayObject *) arr.ptr()); }
    static char *     BYTES(py::array arr) { return PyArray_BYTES((PyArrayObject *) arr.ptr()); }
    static void CopyInto(py::array dst, py::array src) {
        if (0 != PyArray_SIZE((PyArrayObject *) dst.ptr()) && 0 != PyArray_SIZE((PyArrayObject *) src.ptr())) {
            int ret = PyArray_CopyInto((PyArrayObject *) dst.ptr(), (PyArrayObject *) src.ptr());
            if (-1 == ret) { throw py::error_already_set(); }
        }
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

        py::object tmp = py::reinterpret_steal<py::object>(
            PyArray_NewFromDescr(
                &PyArray_Type, PyArray_DescrFromType(m_table.datatypeid()), m_table.ndim(),
                shape, strides, data, NPY_ARRAY_WRITEABLE, nullptr));
        if (!tmp) { py::pybind11_fail("NumPy: unable to create array view"); }

        py::object buffer = py::cast(m_table.buffer());
        py::array ret;
        if (PyArray_SetBaseObject((PyArrayObject *)tmp.ptr(), buffer.inc_ref().ptr()) == 0) {
            ret = tmp;
        }
        return ret;
    }

    LookupTableCore & m_table;

}; /* end class Table */

/**
 * Helper class for pybind11 class wrappers.
 */
template< class Wrapper, class Wrapped, class Holder = std::unique_ptr<Wrapped>> class WrapBase {

public:

    typedef Wrapper wrapper_type;
    typedef Wrapped wrapped_type;
    typedef Holder holder_type;
    typedef WrapBase< wrapper_type, wrapped_type, holder_type > base_type;

    static wrapper_type & commit(py::module & mod, const char * pyname, const char * clsdoc) {
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
    DECL_MARCH_PYBIND_CLASS_METHOD(def_property)
    DECL_MARCH_PYBIND_CLASS_METHOD(def_property_readonly)
    DECL_MARCH_PYBIND_CLASS_METHOD(def_property_readonly_static)

#undef DECL_MARCH_PYBIND_CLASS_METHOD

protected:

    WrapBase(py::module & mod, const char * pyname, const char * clsdoc)
        : m_cls(py::class_< wrapped_type, holder_type >(mod, pyname, clsdoc))
    {}

    py::class_< wrapped_type, holder_type > m_cls;

}; /* end class WrapBase */

class WrapLookupTableCore : public WrapBase< WrapLookupTableCore, LookupTableCore > {

    friend base_type;

    WrapLookupTableCore(py::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
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
        return def("__init__", [](py::args args, py::kwargs kwargs) {
            LookupTableCore & table = *(args[0].cast<LookupTableCore*>());

            std::vector<index_type> dims(args.size()-2);
            index_type nghost = args[1].cast<index_type>();
            index_type nbody = args[2].cast<index_type>();
            dims[0] = nghost + nbody;
            for (size_t it=1; it<dims.size(); ++it) {
                dims[it] = args[it+2].cast<index_type>();
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

            new (&table) LookupTableCore(nghost, nbody, dims, dtid);

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
        });
    }

    wrapper_type & array_readwrite() {
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
        return def("__getstate__", [](LookupTableCore & tbl) {
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

class WrapBoundaryData : public WrapBase< WrapBoundaryData, BoundaryData > {

    friend base_type;

    WrapBoundaryData(py::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        (*this)
            .def(py::init<index_type>())
            .def_property_readonly_static("BFREL", [](py::object const & /* self */) { return BoundaryData::BFREL; })
            .facn()
            .values()
            .pickle()
            .def("good_shape", &BoundaryData::good_shape)
        ;
    }

    wrapper_type & facn() {
        return def_property(
            "facn",
            [](BoundaryData & bnd) {
                if (0 == bnd.facn().nbyte()) {
                    npy_intp shape[2] = {0, BoundaryData::BFREL};
                    return py::reinterpret_steal<py::array>(
                        PyArray_NewFromDescr(
                            &PyArray_Type, PyArray_DescrFromType(bnd.facn().datatypeid()), 2 /* nd */,
                            shape, nullptr /* strides */, nullptr /* data */, 0 /* flags */, nullptr));
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
        );
    }

    wrapper_type & values() {
        return def_property(
            "values",
            [](BoundaryData & bnd) {
                if (0 == bnd.values().nbyte()) {
                    npy_intp shape[2] = {0, bnd.nvalue()};
                    return py::reinterpret_steal<py::array>(
                        PyArray_NewFromDescr(
                            &PyArray_Type, PyArray_DescrFromType(bnd.values().datatypeid()), 2 /* nd */,
                            shape, nullptr /* strides */, nullptr /* data */, 0 /* flags */, nullptr));
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
        );
    }

    wrapper_type & pickle() {
        return (*this)
        .def("__getstate__", &getstate)
        .def("__setstate__", &setstate)
        ;
    }

public:

    static py::object getstate(wrapped_type & bnd) {
        return py::make_tuple(bnd.nbound(), bnd.nvalue(), Table(bnd.facn()).full(), Table(bnd.values()).full());
    }

    static void setstate(wrapped_type & bnd, py::tuple tpl) {
        if (tpl.size() != 4) { throw std::runtime_error("Invalid state for BoundaryData!"); }
        index_type nbound = tpl[0].cast<index_type>();
        index_type nvalue = tpl[1].cast<index_type>();
        py::array facn_farr   = tpl[2].cast<py::array>();
        py::array values_farr = tpl[3].cast<py::array>();
        new (&bnd) wrapped_type(nbound, nvalue);
        Table::CopyInto(Table(bnd.facn()  ).full(), facn_farr  );
        Table::CopyInto(Table(bnd.values()).full(), values_farr);
    }

}; /* end class WrapBoundaryData */

namespace march {
/**
 * This is a workaround for the unability of pybind11 to initialize a
 * shared_ptr using make_shared.
 */
class UnstructuredBlockConstructorAgent {

public:

    template< size_t NDIM >
    static void placement_new(
        UnstructuredBlock<NDIM> & blk
      , index_type nnode, index_type nface, index_type ncell, index_type nbound
      , index_type ngstnode, index_type ngstface, index_type ngstcell
      , bool use_incenter
    ) {
        new (&blk) UnstructuredBlock<NDIM>(
            typename UnstructuredBlock<NDIM>::ctor_passkey()
          , nnode, nface, ncell, nbound, ngstnode, ngstface, ngstcell, use_incenter
        );
    }

}; /* end class UnstructuredBlockConstructorAgent */ } /* end namespace march */

template< size_t NDIM >
class WrapUnstructuredBlock
  : public WrapBase< WrapUnstructuredBlock<NDIM>, UnstructuredBlock<NDIM>, std::shared_ptr<UnstructuredBlock<NDIM>> >
{

    /* I don't know why I need to duplicate these typedef's, but clang doesn't compile if I don't do it. */
    typedef WrapUnstructuredBlock<NDIM> wrapper_type;
    typedef UnstructuredBlock<NDIM> wrapped_type;
    typedef std::shared_ptr<UnstructuredBlock<NDIM>> holder_type;
    typedef WrapBase< wrapper_type, wrapped_type, holder_type > base_type;

    friend base_type;

    WrapUnstructuredBlock(py::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        (*this)
            .def("__init__", [](wrapped_type & self) {
                UnstructuredBlockConstructorAgent::placement_new(self, 0, 0, 0, 0, 0, 0, 0, false);
            })
            .def("__init__", [](wrapped_type & self, index_type nnode, index_type nface, index_type ncell, bool use_incenter) {
                UnstructuredBlockConstructorAgent::placement_new(self, nnode, nface, ncell, 0, 0, 0, 0, use_incenter);
            })
            .shapes()
            .def_property_readonly("use_incenter", &UnstructuredBlock<NDIM>::use_incenter)
            .tables()
            .arrays()
            .def(
                "set_bndvec",
                [](UnstructuredBlock<NDIM> & blk, py::list bndlist) {
                    std::vector<BoundaryData> bndvec;
                    for (auto obj : bndlist) {
                        bndvec.push_back(py::cast<BoundaryData>(obj));
                    }
                    blk.bndvec() = std::move(bndvec);
                },
                "Set boundary data; temporary workaround."
            )
            .def_property_readonly(
                "_bndvec_size",
                [](UnstructuredBlock<NDIM> & blk) { return blk.bndvec().size(); },
                "Size of the boundary data vector; temporary workaround."
            )
            .def(
                "get_bnddata",
                [](UnstructuredBlock<NDIM> & blk, size_t idx) {
                    return blk.bndvec().at(idx);
                },
                "Get boundary data; temporary workaround."
            )
            .def("calc_metric", &UnstructuredBlock<NDIM>::calc_metric)
            .def("build_interior", &UnstructuredBlock<NDIM>::build_interior)
            .def("build_boundary", &UnstructuredBlock<NDIM>::build_boundary)
            .def("build_ghost", &UnstructuredBlock<NDIM>::build_ghost)
            .def(
                "partition",
                [](UnstructuredBlock<NDIM> & blk, index_type npart) {
                    int edgecut;
                    LookupTable<index_type, 0> parts;
                    std::tie(edgecut, parts) = blk.partition(npart);
                    LookupTableCore parts_core = static_cast<LookupTableCore>(parts);
                    return py::make_tuple(edgecut, Table(parts_core).full());
                }
            )
            .pickle()
            .def_property_readonly_static("FCMND", [](py::object const & /* self */) { return UnstructuredBlock<NDIM>::FCMND; })
            .def_property_readonly_static("CLMND", [](py::object const & /* self */) { return UnstructuredBlock<NDIM>::CLMND; })
            .def_property_readonly_static("CLMFC", [](py::object const & /* self */) { return UnstructuredBlock<NDIM>::CLMFC; })
            .def_property_readonly_static("FCNCL", [](py::object const & /* self */) { return UnstructuredBlock<NDIM>::FCNCL; })
            .def_property_readonly_static("FCREL", [](py::object const & /* self */) { return UnstructuredBlock<NDIM>::FCREL; })
            .def_property_readonly_static("BFREL", [](py::object const & /* self */) { return UnstructuredBlock<NDIM>::BFREL; })
        ;
    }

    wrapper_type & shapes() {
        return (*this)
        .def_property_readonly("ndim", &UnstructuredBlock<NDIM>::ndim)
        .def_property_readonly("nnode", &UnstructuredBlock<NDIM>::nnode)
        .def_property_readonly("nface", &UnstructuredBlock<NDIM>::nface)
        .def_property_readonly("ncell", &UnstructuredBlock<NDIM>::ncell)
        .def_property_readonly("nbound", &UnstructuredBlock<NDIM>::nbound)
        .def_property_readonly("ngstnode", &UnstructuredBlock<NDIM>::ngstnode)
        .def_property_readonly("ngstface", &UnstructuredBlock<NDIM>::ngstface)
        .def_property_readonly("ngstcell", &UnstructuredBlock<NDIM>::ngstcell)
        ;
    }

    wrapper_type & tables() {
#define DECL_MARCH_PYBIND_USTBLOCK_TABLE(NAME, DOC) \
        .def_property_readonly( \
            "tb" #NAME, \
            [](UnstructuredBlock<NDIM> & blk) { return static_cast<LookupTableCore>(blk.NAME()); }, \
            #DOC " table")

#define DECL_MARCH_PYBIND_USTBLOCK_TABLES \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(ndcrd, "Node coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(fccnd, "Face center coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(fcnml, "Face center coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(fcara, "Face center coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(clcnd, "Cell center coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(clvol, "Cell volume") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(fctpn, "Face type number") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(cltpn, "Cell type number") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(clgrp, "Cell group number") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(fcnds, "Face nodes") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(fccls, "Face cells") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(clnds, "Cell nodes") \
        DECL_MARCH_PYBIND_USTBLOCK_TABLE(clfcs, "Cell faces")

        return (*this)
        DECL_MARCH_PYBIND_USTBLOCK_TABLES
        ;

#undef DECL_MARCH_PYBIND_USTBLOCK_TABLES
#undef DECL_MARCH_PYBIND_USTBLOCK_TABLE
    }

    wrapper_type & arrays() {
#define DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, NAME, ARR, DOC) \
        .def_property( \
            #PREFIX #NAME, \
            [](UnstructuredBlock<NDIM> & blk)                { return Table(blk.NAME()).ARR(); }, \
            [](UnstructuredBlock<NDIM> & blk, py::array src) { Table::CopyInto(Table(blk.NAME()).ARR(), src); }, \
            #DOC " " #ARR " array")

#define DECL_MARCH_PYBIND_USTBLOCK_ARRAYS(PREFIX, ARR) \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, ndcrd, ARR, "Node coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, fccnd, ARR, "Face center coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, fcnml, ARR, "Face center coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, fcara, ARR, "Face center coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, clcnd, ARR, "Cell center coordinates") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, clvol, ARR, "Cell volume") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, fctpn, ARR, "Face type number") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, cltpn, ARR, "Cell type number") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, clgrp, ARR, "Cell group number") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, fcnds, ARR, "Face nodes") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, fccls, ARR, "Face cells") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, clnds, ARR, "Cell nodes") \
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(PREFIX, clfcs, ARR, "Cell faces")

        return (*this)
        DECL_MARCH_PYBIND_USTBLOCK_ARRAYS(, body)
        DECL_MARCH_PYBIND_USTBLOCK_ARRAYS(gst, ghost)
        DECL_MARCH_PYBIND_USTBLOCK_ARRAYS(sh, full)
        DECL_MARCH_PYBIND_USTBLOCK_ARRAY(, bndfcs, full, "Boundary faces")
        ;

#undef DECL_MARCH_PYBIND_USTBLOCK_ARRAYS
#undef DECL_MARCH_PYBIND_USTBLOCK_ARRAY
    }

    wrapper_type & pickle() {
        return (*this)
        .def("__getstate__", [](wrapped_type & blk) {
            py::dict pickled;
            // shapes.
            pickled["nnode"   ] = py::cast(blk.nnode());
            pickled["nface"   ] = py::cast(blk.nface());
            pickled["ncell"   ] = py::cast(blk.ncell());
            pickled["nbound"  ] = py::cast(blk.nbound());
            pickled["ngstnode"] = py::cast(blk.ngstnode());
            pickled["ngstface"] = py::cast(blk.ngstface());
            pickled["ngstcell"] = py::cast(blk.ngstcell());
            pickled["use_incenter"] = py::cast(blk.use_incenter());
            // arrays.
            pickled["ndcrd"] = Table(blk.ndcrd()).full();
            pickled["fccnd"] = Table(blk.fccnd()).full();
            pickled["fcnml"] = Table(blk.fcnml()).full();
            pickled["fcara"] = Table(blk.fcara()).full();
            pickled["clcnd"] = Table(blk.clcnd()).full();
            pickled["clvol"] = Table(blk.clvol()).full();
            pickled["fctpn"] = Table(blk.fctpn()).full();
            pickled["cltpn"] = Table(blk.cltpn()).full();
            pickled["clgrp"] = Table(blk.clgrp()).full();
            pickled["fcnds"] = Table(blk.fcnds()).full();
            pickled["fccls"] = Table(blk.fccls()).full();
            pickled["clnds"] = Table(blk.clnds()).full();
            pickled["clfcs"] = Table(blk.clfcs()).full();
            pickled["bndfcs"] = Table(blk.bndfcs()).full();
            // bndvec.
            py::list bndlist;
            for (auto & bnd : blk.bndvec()) {
                bndlist.append(WrapBoundaryData::getstate(bnd));
            }
            pickled["bndvec"] = bndlist;
            return pickled;
        })
        .def("__setstate__", [](wrapped_type & blk, py::dict pickled) {
            // shapes.
            index_type nnode    = pickled["nnode"   ].cast<index_type>();
            index_type nface    = pickled["nface"   ].cast<index_type>();
            index_type ncell    = pickled["ncell"   ].cast<index_type>();
            index_type nbound   = pickled["nbound"  ].cast<index_type>();
            index_type ngstnode = pickled["ngstnode"].cast<index_type>();
            index_type ngstface = pickled["ngstface"].cast<index_type>();
            index_type ngstcell = pickled["ngstcell"].cast<index_type>();
            bool use_incenter   = pickled["use_incenter"].cast<bool>();
            UnstructuredBlockConstructorAgent::placement_new(blk, nnode, nface, ncell, nbound, ngstnode, ngstface, ngstcell, use_incenter);
            // arrays.
            Table::CopyInto(Table(blk.ndcrd()).full(), py::array(pickled["ndcrd"]));
            Table::CopyInto(Table(blk.fccnd()).full(), py::array(pickled["fccnd"]));
            Table::CopyInto(Table(blk.fcnml()).full(), py::array(pickled["fcnml"]));
            Table::CopyInto(Table(blk.fcara()).full(), py::array(pickled["fcara"]));
            Table::CopyInto(Table(blk.clcnd()).full(), py::array(pickled["clcnd"]));
            Table::CopyInto(Table(blk.clvol()).full(), py::array(pickled["clvol"]));
            Table::CopyInto(Table(blk.fctpn()).full(), py::array(pickled["fctpn"]));
            Table::CopyInto(Table(blk.cltpn()).full(), py::array(pickled["cltpn"]));
            Table::CopyInto(Table(blk.clgrp()).full(), py::array(pickled["clgrp"]));
            Table::CopyInto(Table(blk.fcnds()).full(), py::array(pickled["fcnds"]));
            Table::CopyInto(Table(blk.fccls()).full(), py::array(pickled["fccls"]));
            Table::CopyInto(Table(blk.clnds()).full(), py::array(pickled["clnds"]));
            Table::CopyInto(Table(blk.clfcs()).full(), py::array(pickled["clfcs"]));
            Table::CopyInto(Table(blk.bndfcs()).full(), py::array(pickled["bndfcs"]));
            // bndvec.
            py::list bndlist = static_cast<py::object>(pickled["bndvec"]);
            BoundaryData bndstorage[1];
            for (py::handle pybnd : bndlist) {
                BoundaryData & bnd = *bndstorage;
                WrapBoundaryData::setstate(bnd, py::cast<py::tuple>(pybnd));
                blk.bndvec().push_back(std::move(bnd));
            }
        })
        ;
    }

}; /* end class WrapUnstructuredBlock */

static PyObject *march_init(py::module & mod) {
    mod.doc() = "libmarch wrapper";

    import_array1(nullptr); // or numpy c api segfault.

    py::class_< Buffer, std::shared_ptr<Buffer> >(mod, "Buffer", "Internal data buffer");
    WrapLookupTableCore::commit(mod, "Table", "Lookup table that allows ghost entity.");
    WrapBoundaryData::commit(mod, "BoundaryData", "Data of a boundary condition.");
    WrapUnstructuredBlock<2>::commit(mod, "UnstructuredBlock2D", "Two-dimensional unstructured mesh block.");
    WrapUnstructuredBlock<3>::commit(mod, "UnstructuredBlock3D", "Three-dimensional unstructured mesh block.");

    return mod.ptr();
}

PYBIND11_MODULE(march, mod) {
    march_init(mod);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
