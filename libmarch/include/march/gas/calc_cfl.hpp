#pragma once

/*
 * Copyright (c) 2014, Yung-Yu Chen <yyc@solvcon.net>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cfloat>
#include <stdexcept>

#include "march/mesh/mesh.hpp"

#include "march/gas/Solver_decl.hpp"

namespace march {

namespace gas {

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

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
