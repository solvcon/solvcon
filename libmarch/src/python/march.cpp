/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <utility>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstring>

#include "march.hpp"
#include "march/python/wrapper_march.hpp"

namespace march {

namespace python {

PyObject * ModuleInitializer::initialize_march(pybind11::module & mod) {
#if defined(Py_DEBUG)
    ::march::setup_debug();
#endif // Py_DEBUG

    import_array1(nullptr); // or numpy c api segfault.

    mod.doc() = "libmarch wrapper";

    // section: core
    WrapBuffer::commit(mod, "Buffer", "Internal data buffer");
    WrapLookupTableCore::commit(mod, "Table", "Lookup table that allows ghost entity.");
    WrapVector<2>::commit(mod, "Vector2D", "Cartesian vector (2D).");
    WrapVector<3>::commit(mod, "Vector3D", "Cartesian vector (3D).");

    // section: mesh
    WrapBoundaryData::commit(mod, "BoundaryData", "Data of a boundary condition.");
    WrapUnstructuredBlock<2>::commit(mod, "UnstructuredBlock2D", "Unstructured mesh block (2D).");
    WrapUnstructuredBlock<3>::commit(mod, "UnstructuredBlock3D", "Unstructured mesh block (3D).");
    WrapNodeHand<2>::commit(mod, "NodeHand2D", "Hand to a node (2D).");
    WrapNodeHand<3>::commit(mod, "NodeHand3D", "Hand to a node (3D).");
    WrapFaceHand<2>::commit(mod, "FaceHand2D", "Hand to a face (2D).");
    WrapFaceHand<3>::commit(mod, "FaceHand3D", "Hand to a face (3D).");
    WrapCellHand<2>::commit(mod, "CellHand2D", "Hand to a cell (2D).");
    WrapCellHand<3>::commit(mod, "CellHand3D", "Hand to a cell (3D).");
    WrapBasicCE<2>::commit(mod, "BasicCE2D", "Basic conservation element (2D).");
    WrapBasicCE<3>::commit(mod, "BasicCE3D", "Basic conservation element (3D).");
    WrapConservationElement<2>::commit(mod, "ConservationElement2D", "Conservation element (2D).");
    WrapConservationElement<3>::commit(mod, "ConservationElement3D", "Conservation element (3D).");

    return mod.ptr();
}

} /* end namespace python */

} /* end namespace march */

PYBIND11_MODULE(libmarch, mod) {
    ::march::python::ModuleInitializer::get_instance().initialize(mod);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
