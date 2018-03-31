/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h>

#include "march.hpp"
#include "march/gas.hpp"
#include "march/python/wrapper_gas.hpp"

namespace march {

namespace python {

PyObject * ModuleInitializer::initialize_march_gas(pybind11::module & marchmod) {
    auto gasmod = marchmod.def_submodule("gas", "Gas dynamic solver");

    // section: solver and associated data
    WrapGasSolver<2>::commit(gasmod, "Solver2D", "Gas-dynamics solver (2D).");
    WrapGasSolver<3>::commit(gasmod, "Solver3D", "Gas-dynamics solver (3D).");
    WrapGasCommonAnchor::commit(gasmod, "CommonAnchor", "Gas-dynamics multi-dimensional anchor.");
    WrapGasAnchor<2>::commit(gasmod, "Anchor2D", "Gas-dynamics anchor (2D).");
    WrapGasAnchor<3>::commit(gasmod, "Anchor3D", "Gas-dynamics anchor (3D).");
    WrapGasAnchorChain<2>::commit(gasmod, "AnchorChain2D", "Gas-dynamics sequential container for anchors (2D).");
    WrapGasAnchorChain<3>::commit(gasmod, "AnchorChain3D", "Gas-dynamics sequential container for anchors (3D).");
    WrapGasParameter::commit(gasmod, "Parameter", "Gas-dynamics solver parameters.");
    WrapGasState::commit(gasmod, "State", "Gas-dynamics solver states.");
    WrapGasSolution<2>::commit(gasmod, "Solution2D", "Gas-dynamics solution data (2D).");
    WrapGasSolution<3>::commit(gasmod, "Solution3D", "Gas-dynamics solution data (3D).");
    WrapGasQuantity<2>::commit(gasmod, "Quantity2D", "Gas-dynamics quantities (2D).");
    WrapGasQuantity<3>::commit(gasmod, "Quantity3D", "Gas-dynamics quantities (3D).");

    // section: boundary-condition treatments
    WrapGasTrimBase<gas::TrimBase<2>, 2>::commit(gasmod, "TrimBase2D", "Gas-dynamics trim base type (2D).");
    WrapGasTrimBase<gas::TrimBase<3>, 3>::commit(gasmod, "TrimBase3D", "Gas-dynamics trim base type (3D).");
    WrapGasTrimInterface<2>::commit(gasmod, "TrimInterface2D", "Gas-dynamics interface trim (2D).");
    WrapGasTrimInterface<3>::commit(gasmod, "TrimInterface3D", "Gas-dynamics interface trim (3D).");
    WrapGasTrimNoOp<2>::commit(gasmod, "TrimNoOp2D", "Gas-dynamics no-op trim (2D).");
    WrapGasTrimNoOp<3>::commit(gasmod, "TrimNoOp3D", "Gas-dynamics no-op trim (3D).");
    WrapGasTrimNonRefl<2>::commit(gasmod, "TrimNonRefl2D", "Gas-dynamics non-reflective trim (2D).");
    WrapGasTrimNonRefl<3>::commit(gasmod, "TrimNonRefl3D", "Gas-dynamics non-reflective trim (3D).");
    WrapGasTrimSlipWall<2>::commit(gasmod, "TrimSlipWall2D", "Gas-dynamics slip wall trim (2D).");
    WrapGasTrimSlipWall<3>::commit(gasmod, "TrimSlipWall3D", "Gas-dynamics slip wall trim (3D).");
    WrapGasTrimInlet<2>::commit(gasmod, "TrimInlet2D", "Gas-dynamics inlet trim (2D).");
    WrapGasTrimInlet<3>::commit(gasmod, "TrimInlet3D", "Gas-dynamics inlet trim (3D).");

    return gasmod.ptr();
}

} /* end namespace python */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
