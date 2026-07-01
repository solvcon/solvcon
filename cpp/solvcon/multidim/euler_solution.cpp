/*
 * Copyright (c) 2017, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multidim/euler.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <vector>

namespace solvcon
{

void EulerCore::initialize_solution()
{
    ssize_t const total = m_ngstcell + m_ncell;
    ssize_t const neq = m_neq;
    ssize_t const ndim = m_ndim;

    auto alloc = [this](SimpleArray<real_type> & arr,
                        small_vector<ssize_t> const & shape)
    {
        arr = SimpleArray<real_type>(shape, 0);
        arr.set_nghost(m_ngstcell);
    };

    alloc(m_so0c, {total, neq});
    alloc(m_so0n, {total, neq});
    alloc(m_so0t, {total, neq});
    alloc(m_so1c, {total, neq, ndim});
    alloc(m_so1n, {total, neq, ndim});
    alloc(m_stm, {total, neq});
    alloc(m_cflo, {total});
    alloc(m_cflc, {total});
    alloc(m_gamma, {total});
}

/**
 * Initialize the conserved solution to a uniform primitive state.  Only the
 * leading ndim entries of @p velocity are read.  Stores momentum = density *
 * velocity and energy = pressure / (gamma - 1) + 0.5 * density * |velocity|^2
 * for uniform inflow.
 *
 * @param gamma     Ratio of specific heat; must be greater than 1.
 * @param rho       Uniform density; must be positive.
 * @param velocity  Flow velocity; only the leading ndim entries are read.
 * @param p         Uniform static pressure; must be non-negative.
 */
void EulerCore::init_solution(
    real_type gamma,
    real_type rho,
    std::array<real_type, 3> const & velocity,
    real_type p)
{
    size_t const ndim = m_ndim;
    // Guard the physical state: gamma > 1 keeps p / (gamma - 1) finite here and
    // in calc_cfl, rho > 0 keeps the per-cell division in calc_cfl finite.
    if (gamma <= 1.0)
    {
        throw std::invalid_argument(std::format(
            "EulerCore::init_solution: gamma {} must be > 1", gamma));
    }
    if (rho <= 0.0)
    {
        throw std::invalid_argument(std::format(
            "EulerCore::init_solution: rho {} must be > 0", rho));
    }
    if (p < 0.0)
    {
        throw std::invalid_argument(std::format(
            "EulerCore::init_solution: pressure {} must be >= 0", p));
    }

    real_type vsq = 0.0;
    for (size_t d = 0; d < ndim; ++d)
    {
        vsq += velocity[d] * velocity[d];
    }
    // Primitive-to-conserved conversion for a uniform state.
    real_type const energy = p / (gamma - 1.0) + 0.5 * rho * vsq;

    // gamma is a uniform physical constant; fill every row, ghost included.
    m_gamma.fill(gamma);

    for (int_type icl = 0; icl < m_ncell; ++icl)
    {
        m_so0n(icl, 0) = rho;
        for (size_t d = 0; d < ndim; ++d)
        {
            m_so0n(icl, 1 + d) = rho * velocity[d];
        }
        m_so0n(icl, m_neq - 1) = energy;
    }
}

/**
 * Compute the per-cell original and clamped CFL numbers (cflo, cflc) from the
 * new-step solution, and rewrite the stored energy to keep the pressure
 * non-negative.
 */
void EulerCore::calc_cfl()
{
    constexpr real_type TINY = 1.e-60;

    size_t const ndim = m_ndim;
    real_type const hdt = m_time_increment / 2.0;
    auto const & msh = *m_mesh;

    for (int_type icl = 0; icl < m_ncell; ++icl)
    {
        int_type const clnfc = msh.clfcs(icl, 0);

        // Estimate the minimal CCE-centroid-to-BCE-centroid distance.
        real_type dist = std::numeric_limits<real_type>::max();
        for (int_type ifl = 1; ifl <= clnfc; ++ifl)
        {
            size_t const bce_col = static_cast<size_t>(ifl) * ndim;
            real_type d2 = 0.0;
            for (size_t d = 0; d < ndim; ++d)
            {
                real_type const diff = m_cecnd(icl, bce_col + d) - m_cecnd(icl, d);
                d2 += diff * diff;
            }
            dist = std::min(dist, std::sqrt(d2));
        }

        // Wave speed from the new-step solution.
        real_type const ga = m_gamma(icl);
        real_type const ga1 = ga - 1.0;
        real_type const density = m_so0n(icl, 0);
        real_type momsq = 0.0;
        for (size_t d = 0; d < ndim; ++d)
        {
            real_type const mom = m_so0n(icl, 1 + d);
            momsq += mom * mom;
        }
        real_type const ke = momsq / (2.0 * density);
        real_type const energy = m_so0n(icl, m_neq - 1);
        real_type const pr = ga1 * (energy - ke);
        // Clamp pressure to be non-negative for the square root.
        real_type const pr_adj = (pr + std::fabs(pr)) / 2.0;
        real_type const wspd =
            std::sqrt(ga * pr_adj / density) + std::sqrt(momsq) / density;

        // CFL number.
        real_type const cflo = hdt * wspd / dist;
        m_cflo(icl) = cflo;
        // If the pressure is null, force the clamped CFL to be 1.
        m_cflc(icl) = (cflo - 1.0) * pr_adj / (pr_adj + TINY) + 1.0;

        // Rewrite the stored energy from the pressure-positive value, adding
        // TINY so a null pressure stays strictly positive.
        m_so0n(icl, m_neq - 1) = pr_adj / ga1 + ke + TINY;
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
