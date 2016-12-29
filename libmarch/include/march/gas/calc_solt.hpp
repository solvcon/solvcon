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
void Solver<NDIM>::calc_solt() {
    // references.
    auto & block = *m_block;
    auto & solt = m_sol.solt;
    auto & sol = m_sol.sol;
    auto & dsol = m_sol.dsol;
    auto & amsca = m_sup.amsca;
    // pointers.
    real_type *psolt, *pidsol, *pdsol;
    // scalars.
    real_type val;
    // jacobian matrix.
    Jacobian<NEQ, NDIM> jaco;
    // interators.
    index_type icl, ieq, jeq, idm;
    for (icl=0; icl<block.ncell(); ++icl) {
        psolt = &solt[icl][0];
        pidsol = &dsol[icl][0];
        jaco(amsca[icl][0], sol[icl]);
        for (ieq=0; ieq<NEQ; ieq++) {
            psolt[ieq] = 0.0;
            for (idm=0; idm<NDIM; idm++) {
                val = 0.0;
                pdsol = pidsol;
                for (jeq=0; jeq<NEQ; jeq++) {
                    val += jaco.jacos[ieq][jeq][idm]*pdsol[idm];
                    pdsol += NDIM;
                };
                psolt[ieq] -= val;
            }
        }
    }
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
