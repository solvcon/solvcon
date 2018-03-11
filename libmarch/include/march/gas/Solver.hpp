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
#include "march/gas/Anchor.hpp"
#include "march/gas/Quantity.hpp"

namespace march {

namespace gas {

template< size_t NDIM >
Solver<NDIM>::Solver(
    const Solver<NDIM>::ctor_passkey &
  , const std::shared_ptr<Solver<NDIM>::block_type> & block
)
  : InstanceCounter<Solver<NDIM>>()
  , m_block(block)
  , m_cecnd(block->ngstcell(), block->ncell())
  , m_sol(block->ngstcell(), block->ncell())
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
void Solver<NDIM>::update(real_type time , real_type time_increment)
{
    m_state.time = time;
    m_state.time_increment = time_increment;
    m_sol.update();
}

template< size_t NDIM >
void Solver<NDIM>::calc_so0t() {
    // references.
    auto & block = *m_block;
    // jacobian matrix.
    Jacobian<neq, ndim> jaco;
    for (index_type icl=0; icl<block.ncell(); ++icl) {
        auto piso0t = m_sol.so0t(icl);
        auto piso1c = m_sol.so1c(icl);
        jaco.update(m_sol.gamma(icl), *m_sol.so0c(icl));
        for (index_type ieq=0; ieq<neq; ieq++) {
            piso0t[ieq] = 0.0;
            for (index_type idm=0; idm<NDIM; idm++) {
                real_type val = 0.0;
                for (index_type jeq=0; jeq<neq; jeq++) {
                    val += jaco.jacos[ieq][jeq][idm]*piso1c[jeq][idm];
                }
                piso0t[ieq] -= val;
            }
        }
    }
}

template< size_t NDIM >
void Solver<NDIM>::calc_so0n() {
    // references.
    const auto & block = *m_block;
    // buffers.
    Jacobian<neq, ndim> jaco;

    const real_type qdt = m_state.time_increment * 0.25;
    const real_type hdt = m_state.time_increment * 0.5;
    for (index_type icl=0; icl<block.ncell(); ++icl) {
        const CompoundCE<NDIM> icce(block, icl);
        auto piso0n = m_sol.so0n(icl);
        piso0n = 0.0; // initialize fluxes.

        const auto & tclfcs = block.clfcs()[icl];
        for (index_type ifl=0; ifl<tclfcs[0]; ++ifl) {
            const BasicCE<NDIM> & ibce = icce.bces[ifl];
            const index_type ifc = tclfcs[ifl+1];
            const auto & tfcnds = block.fcnds()[ifc];
            const index_type jcl = block.fcrcl(ifc, icl); // neighboring cell.
            const auto & jcecnd = reinterpret_cast<const Vector<NDIM> &>(m_cecnd[jcl]);
            const auto pjso0c = m_sol.so0c(jcl);
            const auto pjso0t = m_sol.so0t(jcl);
            const auto pjso1c = m_sol.so1c(jcl);

            // spatial flux (given time).
            for (index_type ieq=0; ieq<neq; ++ieq) {
                real_type fusp = pjso0c[ieq];
                fusp += (ibce.cnd - jcecnd).dot(pjso1c[ieq]);
                piso0n[ieq] += fusp * ibce.vol;
            }

            // temporal flux (given space).
            jaco.update(m_sol.gamma(icl), *pjso0c);
            for (index_type inf=0; inf<tfcnds[0]; ++inf) {
                real_type usfc[neq];
                vector_type dfcn[neq];
                // solution at sub-face center.
                for (index_type ieq=0; ieq<neq; ++ieq) {
                    usfc[ieq] = qdt * pjso0t[ieq];
                    usfc[ieq] += (ibce.sfcnd[inf] - jcecnd).dot(pjso1c[ieq]);
                }
                // spatial derivatives.
                for (index_type ieq=0; ieq<neq; ++ieq) {
                    dfcn[ieq] = jaco.fcn[ieq];
                    for (index_type jeq=0; jeq<neq; ++jeq) {
                        dfcn[ieq] += jaco.jacos[ieq][jeq] * usfc[jeq];
                    }
                }
                // temporal flux.
                for (index_type ieq=0; ieq<neq; ++ieq) {
                    piso0n[ieq] -= hdt * dfcn[ieq].dot(ibce.sfnml[inf]);
                }
            }
        }

        // update solutions.
        for (index_type ieq=0; ieq<neq; ++ieq) {
            piso0n[ieq] /= icce.vol;
        }

        throw_on_negative_density(__func__, icl);
        throw_on_negative_energy(__func__, icl);
    }
}

template< size_t NDIM >
void Solver<NDIM>::calc_cfl() {
    // references.
    auto & block = *m_block;
    const real_type hdt = m_state.time_increment / 2.0;
    for (index_type icl=0; icl<block.ncell(); ++icl) {
        auto & cflc = m_sol.cflc(icl);
        auto & cflo = m_sol.cflo(icl);
        auto piso0n = m_sol.so0n(icl);
        const CompoundCE<NDIM> cce(block, icl);
        const auto & tclfcs = block.clfcs()[icl];
        // estimate distance.
        real_type dist = DBL_MAX;
        for (index_type ifl=0; ifl<tclfcs[0]; ++ifl) {
            // distance.
            const auto vec = cce.bces[ifl].cnd - cce.cnd;
            // minimal value.
            dist = fmin(vec.length(), dist);
        };
        // wave speed.
        const real_type ga = m_sol.gamma(icl);
        const real_type ga1 = ga - 1.0;
        real_type wspd = piso0n.momentum().square();
        const real_type ke = wspd/(2.0*piso0n.density());
        const real_type pr = ga1 * (piso0n.energy() - ke);
        const real_type pr_adj = (pr+fabs(pr))/2.0;
        wspd = sqrt(ga*pr_adj/piso0n.density()) + sqrt(wspd)/piso0n.density();
        // CFL.
        cflo = hdt*wspd/dist;
        // if pressure is null, make CFL to be 1.
        cflc = (cflo-1.0) * pr_adj/(pr_adj+TINY) + 1.0;
        throw_on_cfl_adjustment(__func__, icl);
        throw_on_cfl_overflow(__func__, icl);
        // correct negative pressure.
        piso0n.energy() = pr_adj/ga1 + ke + TINY;
    }
}

template< size_t NDIM >
void Solver<NDIM>::march(real_type time_current, real_type time_increment, Solver<NDIM>::int_type steps_run)
{
    state().step_current = 0;
    anchors().premarch();
    while (state().step_current < steps_run) {
        state().substep_current = 0;
        anchors().prefull();
        if (qty() && 0 != state().report_interval && 0 == state().step_global) { qty()->update(); }
        while (state().substep_current < state().substep_run) {
            // set up time
            state().time = time_current;
            state().time_increment = time_increment;
            anchors().presub();
            // marching methods
            update(state().time, state().time_increment);
            calc_so0t();
            calc_so0n();
            //ibcsoln(); // to be implemented
            trim_do0();
            calc_cfl();
            calc_so1n();
            //ibcdsoln(); // to be implemented
            trim_do1();
            // increment time
            time_current += state().time_increment / state().substep_run;
            state().time = time_current;
            state().time_increment = time_increment;
            state().substep_current += 1;
            anchors().postsub();
        }
        if (
            qty() && 0 != state().report_interval &&
            ((0 != state().step_global) && (0 == state().step_global % state().report_interval))
        ) { qty()->update(); }
        state().step_global += 1;
        state().step_current += 1;
        anchors().postfull();
    }
    anchors().postmarch();
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
