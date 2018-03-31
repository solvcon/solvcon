/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>

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
WrapBoundaryData
  : public WrapBase< WrapBoundaryData, BoundaryData >
{

    friend base_type;

    WrapBoundaryData(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
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
        namespace py = pybind11;
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
        namespace py = pybind11;
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
        namespace py = pybind11;
        return (*this)
            .def(py::pickle(&getstate, &setstate))
        ;
    }

public:

    static pybind11::tuple getstate(wrapped_type & bnd) {
        namespace py = pybind11;
        return py::make_tuple(bnd.nbound(), bnd.nvalue(), Table(bnd.facn()).full(), Table(bnd.values()).full());
    }

    static wrapped_type setstate(pybind11::tuple tpl) {
        namespace py = pybind11;
        if (tpl.size() != 4) { throw std::runtime_error("Invalid state for BoundaryData!"); }
        index_type nbound = tpl[0].cast<index_type>();
        index_type nvalue = tpl[1].cast<index_type>();
        py::array facn_farr   = tpl[2].cast<py::array>();
        py::array values_farr = tpl[3].cast<py::array>();
        wrapped_type bnd(nbound, nvalue);
        Table::CopyInto(Table(bnd.facn()  ).full(), facn_farr  );
        Table::CopyInto(Table(bnd.values()).full(), values_farr);
        return bnd;
    }

}; /* end class WrapBoundaryData */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapUnstructuredBlock
  : public WrapBase< WrapUnstructuredBlock<NDIM>, UnstructuredBlock<NDIM>, std::shared_ptr<UnstructuredBlock<NDIM>> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapUnstructuredBlock<NDIM>, UnstructuredBlock<NDIM>, std::shared_ptr<UnstructuredBlock<NDIM>> >;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    friend base_type;

    WrapUnstructuredBlock(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
        (*this)
            .def(py::init([]() {
                return UnstructuredBlock<NDIM>::construct(0, 0, 0, 0, 0, 0, 0, false);
            }))
            .def(py::init([](index_type nnode, index_type nface, index_type ncell, bool use_incenter) {
                return UnstructuredBlock<NDIM>::construct(nnode, nface, ncell, 0, 0, 0, 0, use_incenter);
            }))
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
// FIXME: in the future, remove "tb" from the "table" attributes
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
        namespace py = pybind11;
// FIXME: in the future, change the way to expose arrays
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
        namespace py = pybind11;
        return (*this)
        .def(py::pickle(
            [](wrapped_type & blk) {
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
            },
            [](py::dict pickled) {
                // shapes.
                index_type nnode    = pickled["nnode"   ].cast<index_type>();
                index_type nface    = pickled["nface"   ].cast<index_type>();
                index_type ncell    = pickled["ncell"   ].cast<index_type>();
                index_type nbound   = pickled["nbound"  ].cast<index_type>();
                index_type ngstnode = pickled["ngstnode"].cast<index_type>();
                index_type ngstface = pickled["ngstface"].cast<index_type>();
                index_type ngstcell = pickled["ngstcell"].cast<index_type>();
                bool use_incenter   = pickled["use_incenter"].cast<bool>();
                auto blk_holder = UnstructuredBlock<NDIM>::construct(nnode, nface, ncell, nbound, ngstnode, ngstface, ngstcell, use_incenter);
                auto & blk = *blk_holder;
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
                for (py::handle pybnd : bndlist) {
                    BoundaryData bnd = WrapBoundaryData::setstate(py::cast<py::tuple>(pybnd));
                    blk.bndvec().push_back(std::move(bnd));
                }
                return blk_holder;
            }))
        ;
    }

}; /* end class WrapUnstructuredBlock */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapBasicCE
  : public WrapBase< WrapBasicCE<NDIM>, BasicCE<NDIM> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapBasicCE<NDIM>, BasicCE<NDIM> >;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    using block_type = UnstructuredBlock<NDIM>;

    friend base_type;

    WrapBasicCE(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
        (*this)
            .def(
                py::init([](block_type & block, index_type icl, index_type ifl, bool init_sentinel) {
                    return wrapped_type(block, icl, ifl, init_sentinel);
                }),
                py::arg("block"), py::arg("icl"), py::arg("ifl"), py::arg("init_sentinel")=true
            )
            .def("repr", &wrapped_type::repr, py::arg("indent")=0, py::arg("precision")=0)
            .def("__repr__", [](wrapped_type & self){ return self.repr(); })
            .def("init_sentinel", &wrapped_type::init_sentinel)
            .def_readwrite("cnd", &wrapped_type::cnd)
            .def_readwrite("vol", &wrapped_type::vol)
            .def_property_readonly("nsubface", [](wrapped_type &){ return index_type(block_type::FCMND); })
            .def("get_sfcnd", [](wrapped_type & self, index_type isf){ return self.sfcnd.at(isf); })
            .def("get_sfnml", [](wrapped_type & self, index_type isf){ return self.sfnml.at(isf); })
        ;
    }

}; /* end class WrapBasicCE */

template< size_t NDIM >
class
MARCH_PYTHON_WRAPPER_VISIBILITY
WrapConservationElement
  : public WrapBase< WrapConservationElement<NDIM>, ConservationElement<NDIM> >
{

    /* aliases for dependent type name lookup */
    using base_type = WrapBase< WrapConservationElement<NDIM>, ConservationElement<NDIM> >;
    using wrapper_type = typename base_type::wrapper_type;
    using wrapped_type = typename base_type::wrapped_type;

    using block_type = UnstructuredBlock<NDIM>;

    friend base_type;

    WrapConservationElement(pybind11::module & mod, const char * pyname, const char * clsdoc)
        : base_type(mod, pyname, clsdoc)
    {
        namespace py = pybind11;
        (*this)
            .def(
                py::init([](block_type & block, index_type icl, bool init_sentinel) {
                    return wrapped_type(block, icl, init_sentinel);
                }),
                py::arg("block"), py::arg("icl"), py::arg("init_sentinel")=true
            )
            .def("repr", &wrapped_type::repr, py::arg("indent")=0, py::arg("precision")=0)
            .def("__repr__", [](wrapped_type & self){ return self.repr(); })
            .def("init_sentinel", &wrapped_type::init_sentinel)
            .def_readwrite("cnd", &wrapped_type::cnd)
            .def_readwrite("vol", &wrapped_type::vol)
            .def_property_readonly("nbce", [](wrapped_type & self) { return self.bces.size(); })
            .def("get_bce", [](wrapped_type & self, index_type ibce) { return self.bces.at(ibce); })
        ;
    }

}; /* end class WrapConservationElement */

} /* end namespace python */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
