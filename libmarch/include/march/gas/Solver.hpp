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
void Solver<NDIM>::update(real_type time , real_type time_increment)
{
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

template< size_t NDIM >
void Solver<NDIM>::init_solution(
    real_type gas_constant
  , real_type gamma
  , real_type density
  , real_type temperature
) {
    for (index_type icl=0; icl<m_block->ncell(); ++icl) {
        m_sol.so0n(icl).set_by(gas_constant, gamma, density, temperature);
    }
}

template< size_t NDIM >
void Solver<NDIM>::throw_on_negative_density(const std::string & srcloc, index_type icl) const {
    auto pso0n = m_sol.so0n(icl);
    if (m_param.stop_on_negative_density() != 0
     && pso0n.density() < 0
     && fabs(pso0n.density()) > m_param.stop_on_negative_density()
    ) {
        throw std::runtime_error(string_format(
            "negative density\n" "in function: %s\n" "%s\n" "%s\n" "%s\n"
            "density = %g (abs > %g)\n"
          , srcloc.c_str()
          , m_block->info_string().c_str()
          , m_block->cell_info_string(icl).c_str()
          , m_state.step_info_string().c_str()
          , pso0n.density()
          , m_param.stop_on_negative_density()
        ));
    }
}

template< size_t NDIM >
void Solver<NDIM>::throw_on_negative_energy(const std::string & srcloc, index_type icl) const {
    auto pso0n = m_sol.so0n(icl);
    if (m_param.stop_on_negative_energy() != 0
     && pso0n.energy() < 0
     && fabs(pso0n.energy()) > m_param.stop_on_negative_energy()
    ) {
        throw std::runtime_error(string_format(
            "negative energy\n" "in function: %s\n" "%s\n" "%s\n" "%s\n"
            "energy = %g (abs > %g)\n"
          , srcloc.c_str()
          , m_block->info_string().c_str()
          , m_block->cell_info_string(icl).c_str()
          , m_state.step_info_string().c_str()
          , pso0n.energy()
          , m_param.stop_on_negative_density()
        ));
    }
}

template< size_t NDIM >
void Solver<NDIM>::throw_on_cfl_adjustment(const std::string & srcloc, index_type icl) const {
    if (m_param.stop_on_cfl_adjustment() != 0 && m_sol.cflc(icl) == 1) {
        throw std::runtime_error(string_format(
            "cfl adjusted\n" "in function: %s\n" "%s\n" "%s\n" "%s\n"
            "energy = %g\n"
            "pressure = %g\n"
            "max_wavespeed = %g\n"
            "original cfl (cflo) = %g\n"
            "corrected cfl (cflc) = %g\n"
            "difference (cflc - cflo) = %g\n"
          , srcloc.c_str()
          , m_block->info_string().c_str()
          , m_block->cell_info_string(icl).c_str()
          , m_state.step_info_string().c_str()
          , m_sol.so0n(icl).energy()
          , m_sol.so0n(icl).pressure(m_sol.gamma(icl))
          , m_sol.so0n(icl).max_wavespeed(m_sol.gamma(icl))
          , m_sol.cflo(icl)
          , m_sol.cflc(icl)
          , m_sol.cflc(icl) - m_sol.cflo(icl)
        ));
    }
}

template< size_t NDIM >
void Solver<NDIM>::throw_on_cfl_overflow(const std::string & srcloc, index_type icl) const {
    if (m_param.stop_on_cfl_adjustment() != 0 && m_sol.cflc(icl) > 1) {
        throw std::runtime_error(string_format(
            "cfl overflow\n" "in function: %s\n" "%s\n" "%s\n" "%s\n"
            "energy = %g\n"
            "pressure = %g\n"
            "max_wavespeed = %g\n"
            "original cfl (cflo) = %g\n"
            "corrected cfl (cflc) = %g\n"
            "difference (cflc - cflo) = %g\n"
          , srcloc.c_str()
          , m_block->info_string().c_str()
          , m_block->cell_info_string(icl).c_str()
          , m_state.step_info_string().c_str()
          , m_sol.so0n(icl).energy()
          , m_sol.so0n(icl).pressure(m_sol.gamma(icl))
          , m_sol.so0n(icl).max_wavespeed(m_sol.gamma(icl))
          , m_sol.cflo(icl)
          , m_sol.cflc(icl)
          , m_sol.cflc(icl) - m_sol.cflo(icl)
        ));
    }
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
