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

#include "march/gas/Solver_decl.hpp"

namespace march {

namespace gas {

template< size_t NDIM >
void Solver<NDIM>::calc_soln() {
    // references.
    const auto & block = *m_block;
    const auto & amsca = m_sup.amsca;
    const auto & sol = m_sol.sol;
    const auto & dsol = m_sol.dsol;
    const auto & solt = m_sol.solt;
    auto & soln = m_sol.soln;
    // buffers.
    Jacobian<NEQ, NDIM> jaco;
    vector_type vec_tmp;

    const real_type qdt = m_state.time_increment * 0.25;
    const real_type hdt = m_state.time_increment * 0.5;
    for (index_type icl=0; icl<block.ncell(); ++icl) {
        const CompoundCE<NDIM> icce(block, icl);

        // initialize fluxes.
        for (index_type ieq=0; ieq<NEQ; ++ieq) {
            soln[icl][ieq] = 0.0;
        }

        const auto & tclfcs = block.clfcs()[icl];
        for (index_type ifl=0; ifl<tclfcs[0]; ++ifl) {
            const BasicCE<NDIM> & ibce = icce.bces[ifl];
            const index_type ifc = tclfcs[ifl+1];
            const auto & tfcnds = block.fcnds()[ifc];
            const index_type jcl = block.fcrcl(ifc, icl); // neighboring cell.
            const auto & jcecnd = reinterpret_cast<const Vector<NDIM> &>(m_cecnd[jcl]);

            // spatial flux (given time).
            for (index_type ieq=0; ieq<NEQ; ++ieq) {
                real_type fusp = sol[jcl][ieq];
                vec_tmp = dsol[jcl][ieq];
                fusp += (ibce.cnd - jcecnd).dot(vec_tmp);
                soln[icl][ieq] += fusp * ibce.vol;
            }

            // temporal flux (given space).
            jaco(amsca[icl][0], sol[jcl]);
            for (index_type inf=0; inf<tfcnds[0]; ++inf) {
                real_type usfc[NEQ];
                vector_type dfcn[NEQ];
                // solution at sub-face center.
                for (index_type ieq=0; ieq<NEQ; ++ieq) {
                    usfc[ieq] = qdt * solt[jcl][ieq];
                    vec_tmp = dsol[jcl][ieq];
                    usfc[ieq] += (ibce.sfcnd[inf] - jcecnd).dot(vec_tmp);
                }
                // spatial derivatives.
                for (index_type ieq=0; ieq<NEQ; ++ieq) {
                    dfcn[ieq] = jaco.fcn[ieq];
                    for (index_type jeq=0; jeq<NEQ; ++jeq) {
                        vec_tmp = jaco.jacos[ieq][jeq];
                        vec_tmp *= usfc[jeq];
                        dfcn[ieq] += vec_tmp;
                    }
                }
                // temporal flux.
                for (index_type ieq=0; ieq<NEQ; ++ieq) {
                    soln[icl][ieq] -= hdt * dfcn[ieq].dot(ibce.sfnml[inf]);
                }
            }
        }

        // update solutions.
        for (index_type ieq=0; ieq<NEQ; ++ieq) {
            soln[icl][ieq] /= icce.vol;
        }
    }
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
