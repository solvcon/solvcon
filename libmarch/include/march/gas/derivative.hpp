#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 *
 * This file includes code that implements the derivative calculation that
 * makes use of gradient element.  Solver<NDIM>::calc_so1n is the solver
 * inteface.
 */

#include <cstdint>

#include "march/core.hpp"
#include "march/mesh.hpp"
#include "march/gas/Solver_decl.hpp"

namespace march {

namespace gas {

/**
 * Calculate the weight of gradient and the derivative.
 */
template< size_t NDIM, size_t NEQ, int32_t ALPHA=1, bool TAYLOR=true >
struct GradientWeigh {

    typedef Solution<NDIM> solution_type;

    static constexpr size_t NFGE_MAX = GEType::NFGE_MAX;

    static constexpr real_type ALMOST_ZERO = Solver<NDIM>::ALMOST_ZERO;

    const GradientElement<NDIM> & gelem;
    const solution_type & shouse;

    /**
     * Gradient of each fundamental gradient element.
     */
    Vector<NDIM> grad[NFGE_MAX][NEQ];
    /**
     * Weighting of each fundamental gradient element.
     */
    real_type widv[NFGE_MAX][NEQ];
    /**
     * Total weighting of all fundamental gradient elements.
     */
    real_type wacc[NEQ];
    real_type sigma_max[NEQ]; // FIXME: document

    /**
     * @param[in] gelem  The shape of gradient elements.
     * @param[in] shouse  Container of solution variables and their derivatives.
     * @param[in] hdt     Half increment time (delta t).
     * @param[in] sgm0    Parameter for weighting.
     */
    GradientWeigh(
        const GradientElement<NDIM> & gelem
      , const solution_type & shouse
      , const real_type hdt
      , const real_type sgm0
    )
      : gelem(gelem)
      , shouse(shouse)
    {
#ifdef MH_DEBUG
        fill_sentinel(&grad[0][0][0], NFGE_MAX * NEQ * NDIM);
        fill_sentinel(&widv[0][0], NFGE_MAX * NEQ);
        fill_sentinel(&wacc[0], NEQ);
        fill_sentinel(&sigma_max[0], NEQ);
#endif // MH_DEBUG

        // calculate gradient and weighting delta.
        for (index_type ieq=0; ieq<NEQ; ++ieq) { wacc[ieq] = 0; }
        for (index_type ifge=0; ifge<gelem.getype.nfge; ++ifge) {
            // interpolated solution
            const auto udf = interpolate_solution(gelem.getype.faces[ifge], hdt);
            // displacement matrix
            const auto dst = gelem.calc_displacement_matrix(ifge);
            // inverse (unnormalized) displacement matrix
            const auto dnv = unnormalized_inverse(dst);
            real_type voc;
            if (NDIM == 3) { voc = dnv.column(2).dot(dst[2]); }
            else           { voc = dst[0][0]*dst[1][1] - dst[0][1]*dst[1][0]; }
            // calculate gradient and weighting delta.
            for (index_type ieq=0; ieq<NEQ; ++ieq) {
                // store for later widv.
                grad[ifge][ieq] = product(dnv, udf[ieq]) / voc;
                // W-1/2 weighting function.
                real_type wgt = grad[ifge][ieq].square();
                if      (0 == ALPHA) { wgt = 1.0; }
                else if (1 == ALPHA) { wgt = 1.0 / sqrt(wgt+ALMOST_ZERO); }
                else if (2 == ALPHA) { wgt = 1.0 / (wgt+ALMOST_ZERO); }
                else                 { wgt = 1.0 / pow(sqrt(wgt+ALMOST_ZERO), ALPHA); }
                //wgt = 1.0 / pow(sqrt(wgt+ALMOST_ZERO), ALPHA); // may be useful for debugging.
                // store and accumulate weighting function.
                wacc[ieq] += wgt;
                widv[ifge][ieq] = wgt;
            }
        }

        // calculate W-3/4 delta and sigma_max.
        real_type wpa[NEQ][2]; // W-3/4 parameter.
        for (index_type ieq=0; ieq<NEQ; ++ieq) { wpa[ieq][0] = wpa[ieq][1] = 0.0; }
        const auto ofg1 = gelem.getype.nfge_inverse;
        for (index_type ifge=0; ifge<gelem.getype.nfge; ++ifge) {
            for (index_type ieq=0; ieq<NEQ; ++ieq) {
                const real_type wgt = widv[ifge][ieq] / wacc[ieq] - ofg1;
                widv[ifge][ieq] = wgt;
                wpa[ieq][0] = fmax(wpa[ieq][0], wgt);
                wpa[ieq][1] = fmin(wpa[ieq][1], wgt);
            }
        }
        for (index_type ieq=0; ieq<NEQ; ++ieq) {
            sigma_max[ieq] = fmin(
                (1.0-ofg1)/(wpa[ieq][0]+ALMOST_ZERO),
                    -ofg1 /(wpa[ieq][1]-ALMOST_ZERO)
            );
            sigma_max[ieq] = fmin(sigma_max[ieq], sgm0);
        }
    }

    /**
     * @param[out] dsoln  The result derivative.
     */
    void operator() (typename solution_type::o1hand_type pso1n) const {
        pso1n = 0;
        const auto ofg1 = gelem.getype.nfge_inverse;
        for (index_type isub=0; isub<gelem.getype.nfge; ++isub) {
            for (index_type ieq=0; ieq<NEQ; ++ieq) {
                const real_type wgt = ofg1 + sigma_max[ieq] * widv[isub][ieq];
                pso1n[ieq] += wgt * grad[isub][ieq];
            }
        }
    }

private:

    std::array<Vector<NDIM>, NEQ> interpolate_solution(
        const GEType::fge_facelist_type & tface
      , const real_type hdt
    ) const {
        std::array<Vector<NDIM>, NEQ> udf;
        const auto piso0n = shouse.so0n(gelem.icl);
        for (index_type ivx=0; ivx<NDIM; ++ivx) {
            const index_type ifl = tface[ivx]-1;
            assert(ifl >= 0);
            const auto jcl = gelem.rcls[ifl];
            const auto pjso0c = shouse.so0c(jcl);
            const auto pjso0n = shouse.so0n(jcl);
            const auto pjso0t = shouse.so0t(jcl);
            const auto pjso1c = shouse.so1c(jcl);
            for (index_type ieq=0; ieq<NEQ; ++ieq) {
                if (TAYLOR) { udf[ieq][ivx] = pjso0c[ieq] + hdt*pjso0t[ieq] - piso0n[ieq]; }
                else        { udf[ieq][ivx] = pjso0n[ieq] - piso0n[ieq]; }
                udf[ieq][ivx] += gelem.jdis[ifl].dot(pjso1c[ieq]);
            }
        }
        return udf;
    }

}; /* end struct GradientWeigh */

template< size_t NDIM >
void Solver<NDIM>::calc_so1n() {
    // references.
    const auto & block = *m_block;
    const real_type hdt = m_state.time_increment * 0.5;
    for (index_type icl=0; icl<block.ncell(); ++icl) {
        // determine sigma0 and tau.
        const real_type cfl = m_sol.cflc(icl);
        const real_type sgm0 = m_param.sigma0() / fabs(cfl);
        const real_type tau = m_param.taumin() + fabs(cfl) * m_param.tauscale();
        // calculate gradient.
        const GradientElement<ndim> gelem(block, m_cecnd, icl, tau);
        const GradientWeigh<ndim,neq> gweigh(gelem, m_sol, hdt, sgm0);
        gweigh(m_sol.so1n(icl));
    }
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
