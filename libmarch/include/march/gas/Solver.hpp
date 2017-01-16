#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>
#include <limits>
#include <memory>

#include "march/core/core.hpp"
#include "march/mesh/mesh.hpp"

#include "march/gas/Solver_decl.hpp"
#include "march/gas/Quantity.hpp"

namespace march {

namespace gas {

template< size_t NDIM >
void Solver<NDIM>::update(real_type time, real_type time_increment) {
    m_state.time = time;
    m_state.time_increment = time_increment;
    m_sol.update();
}

template< size_t NDIM >
Solver<NDIM>::Solver(
    const Solver<NDIM>::ctor_passkey &
  , const std::shared_ptr<Solver<NDIM>::block_type> & block
)
  : m_block(block)
  , m_cecnd(block->ngstcell(), block->ncell())
  , m_sol(block->ngstcell(), block->ncell())
  , m_sup(block->ngstcell(), block->ncell())
  , m_qty(*this)
{
    for (index_type icl=0; icl<block->ncell(); ++icl) {
        reinterpret_cast<Vector<NDIM> &>(m_cecnd[icl]) = CompoundCE<NDIM>(*block, icl).cnd;
    }
    // the ghost CE centroid is initialized to the mirror image of the interior CE.
    for (index_type ibnd=0; ibnd<block->nbound(); ++ibnd) {
        const auto ifc = block->bndfcs()[ibnd][0];
        const auto icl = block->fccls()[ifc][0]; // interior cell
        const auto jcl = block->fccls()[ifc][1]; // ghost cell
        const CompoundCE<NDIM> icce(*block, icl);
        reinterpret_cast<Vector<NDIM> &>(m_cecnd[jcl]) = icce.mirror_centroid(
            reinterpret_cast<const Vector<NDIM> &>(block->fccnd()[ifc])
          , reinterpret_cast<const Vector<NDIM> &>(block->fcnml()[ifc])
        );
    }
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
