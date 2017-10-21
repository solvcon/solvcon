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

#include "march/mesh/UnstructuredBlock/class.hpp"

namespace march {

template< size_t NDIM >
void UnstructuredBlock<NDIM>::CEMesh::calc_sf(const UnstructuredBlock<NDIM> & block) {
    // references.
    const auto & ndcrd = block.ndcrd();
    const auto & fcnds = block.fcnds();
    const auto & fccls = block.fccls();
    const auto & clfcs = block.clfcs();
    const auto & clcnd = block.clcnd();
    // indices.
    index_type clnfc, fcnnd;
    // partial pointers.
    const index_type *pclfcs, *pfcnds, *pfccls;
    const real_type *pndcrd, *pclcnd;
    real_type (*psfmrc)[2][NDIM];
    // scalars.
    real_type voe, disu0, disu1, disu2, disv0, disv1, disv2;
    // arrays.
    real_type crd[FCMND+1][NDIM], cnde[NDIM];
    // interators.
    index_type icl, ifl, inf, ifc, jcl;
    for (icl=0; icl<block.ncell(); ++icl) {
        pclfcs = &clfcs[icl][0];
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ++ifl) {
            ifc = pclfcs[ifl];
            // face node coordinates.
            pfcnds = &fcnds[ifc][0];
            fcnnd = pfcnds[0];
            for (inf=0; inf<fcnnd; ++inf) {
                pndcrd = &ndcrd[pfcnds[inf+1]][0];
                crd[inf][0] = pndcrd[0];
                crd[inf][1] = pndcrd[1];
                if (NDIM == 3) {
                    crd[inf][2] = pndcrd[2];
                }
            }
            crd[fcnnd][0] = crd[0][0];
            crd[fcnnd][1] = crd[0][1];
            if (NDIM == 3) {
                crd[fcnnd][2] = crd[0][2];
            }
            // neighboring cell center.
            pfccls = &fccls[ifc][0];
            jcl = pfccls[0] + pfccls[1] - icl;
            pclcnd = &clcnd[jcl][0];
            cnde[0] = pclcnd[0];
            cnde[1] = pclcnd[1];
            if (NDIM == 3) {
                cnde[2] = pclcnd[2];
            }
            // calculate geometric center of the bounding sub-face.
            psfmrc = (double (*)[2][NDIM])(&sfmrc[0][0]
                + ((icl*CLMFC + ifl-1)*FCMND*2*NDIM));
            for (inf=0; inf<fcnnd; ++inf) {
                psfmrc[inf][0][0] = cnde[0] + crd[inf][0];
                if (NDIM == 3) {
                    psfmrc[inf][0][0] += crd[inf+1][0];
                }
                psfmrc[inf][0][0] /= NDIM;
                psfmrc[inf][0][1] = cnde[1] + crd[inf][1];
                if (NDIM == 3) {
                    psfmrc[inf][0][1] += crd[inf+1][1];
                }
                psfmrc[inf][0][1] /= NDIM;
                if (NDIM == 3) {
                    psfmrc[inf][0][2] = cnde[2] + crd[inf][2];
                    psfmrc[inf][0][2] += crd[inf+1][2];
                    psfmrc[inf][0][2] /= NDIM;
                }
            }
            // calculate outward area vector of the bounding sub-face.
            if (NDIM == 3) {
                voe = (pfccls[0] - icl) + DBL_MIN;
                voe /= (icl - pfccls[0]) + DBL_MIN;
                voe *= 0.5;
                for (inf=0; inf<fcnnd; ++inf) {
                    disu0 = crd[inf  ][0] - cnde[0];
                    disu1 = crd[inf  ][1] - cnde[1];
                    disu2 = crd[inf  ][2] - cnde[2];
                    disv0 = crd[inf+1][0] - cnde[0];
                    disv1 = crd[inf+1][1] - cnde[1];
                    disv2 = crd[inf+1][2] - cnde[2];
                    psfmrc[inf][1][0] = (disu1*disv2 - disu2*disv1) * voe;
                    psfmrc[inf][1][1] = (disu2*disv0 - disu0*disv2) * voe;
                    psfmrc[inf][1][2] = (disu0*disv1 - disu1*disv0) * voe;
                }
            } else {
                voe = (crd[0][0]-cnde[0])*(crd[1][1]-cnde[1])
                    - (crd[0][1]-cnde[1])*(crd[1][0]-cnde[0]);
                voe /= fabs(voe);
                psfmrc[0][1][0] = -(cnde[1]-crd[0][1]) * voe;
                psfmrc[0][1][1] =  (cnde[0]-crd[0][0]) * voe;
                psfmrc[1][1][0] =  (cnde[1]-crd[1][1]) * voe;
                psfmrc[1][1][1] = -(cnde[0]-crd[1][0]) * voe;
            }
        }
    }
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
