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

#include "march/mesh/mesh.hpp"

#include "march/gas/Solver_decl.hpp"

namespace march {

namespace gas {

template< size_t NDIM >
void Solver<NDIM>::calc_cfl() {
    // references.
    auto & block = *m_block;
    auto & amsca = m_sup.amsca;
    auto & soln = m_sol.soln;
    auto & cfl = m_sol.cfl;
    auto & ocfl = m_sol.ocfl;
    auto & clfcs = block.clfcs();
    // indices.
    index_type clnfc;
    // pointers.
    index_type *pclfcs;
    real_type *pamsca, *pcfl, *pocfl, *psoln;
    // scalars.
    real_type hdt, dist, wspd, ga, ga1, pr, ke;
    // arrays.
    vector_type vec;
    hdt = m_state.time_increment / 2.0;
    for (index_type icl=0; icl<block.ncell(); ++icl) {
        pamsca = &amsca[icl][0];
        pcfl = &cfl[icl];
        pocfl = &ocfl[icl];
        psoln = &soln[icl][0];
        const CompoundCE<NDIM> cce(block, icl);
        pclfcs = &clfcs[icl][0];
        // estimate distance.
        dist = DBL_MAX;
        clnfc = pclfcs[0];
        for (index_type ifl=0; ifl<clnfc; ++ifl) {
            // distance.
            vec = cce.bces[ifl].cnd - cce.cnd;
            // minimal value.
            dist = fmin(vec.length(), dist);
        };
        // wave speed.
        ga = pamsca[0];
        ga1 = ga - 1.0;
        if (NDIM == 3) { wspd = psoln[1]*psoln[1] + psoln[2]*psoln[2] + psoln[3]*psoln[3]; }
        else           { wspd = psoln[1]*psoln[1] + psoln[2]*psoln[2]; }
        ke = wspd/(2.0*psoln[0]);
        pr = ga1 * (psoln[1+NDIM] - ke);
        pr = (pr+fabs(pr))/2.0;
        wspd = sqrt(ga*pr/psoln[0]) + sqrt(wspd)/psoln[0];
        // CFL.
        pocfl[0] = hdt*wspd/dist;
        // if pressure is null, make CFL to be 1.
        pcfl[0] = (pocfl[0]-1.0) * pr/(pr+TINY) + 1.0;
        // correct negative pressure.
        psoln[1+NDIM] = pr/ga1 + ke + TINY;
        // advance.
        pamsca += NSCA;
        pcfl += 1;
        pocfl += 1;
        psoln += NEQ;
        pclfcs += CLMFC+1;
    }
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
