/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/onedim/Euler1DCore.hpp>
#include <solvcon/toggle/toggle.hpp>
#include <cmath>

namespace solvcon
{

namespace onedim
{

std::ostream & operator<<(std::ostream & os, const Euler1DCore & sol)
{
    os << "Euler1DCore(ncoord=" << sol.ncoord() << ", time_increment=" << sol.time_increment() << ")";
    return os;
}

void Euler1DCore::initialize_data(size_t ncoord)
{
    SOLVCON_PROFILE_SCOPE("Euler1DCore::initialize_data");
    if (0 == ncoord % 2)
    {
        throw std::invalid_argument("ncoord cannot be even");
    }
    auto const coord_count = static_cast<ssize_t>(ncoord);
    m_coord = SimpleArray<double>(/*length*/ ncoord);
    m_cfl = SimpleArray<double>(/*length*/ ncoord);
    m_so0 = SimpleArray<double>(/*shape*/ small_vector<ssize_t>{coord_count, NVAR});
    m_so1 = SimpleArray<double>(/*shape*/ small_vector<ssize_t>{coord_count, NVAR});
    m_gamma = SimpleArray<double>(/*shape*/ small_vector<ssize_t>{coord_count}, /*value*/ 1.4);
}

SimpleArray<double> Euler1DCore::density() const
{
    SOLVCON_PROFILE_SCOPE("Euler1DCore::density");
    SimpleArray<double> ret(ncoord());
    for (size_t it = 0; it < ncoord(); ++it)
    {
        ret(it) = density(it);
    }
    return ret;
}

SimpleArray<double> Euler1DCore::velocity() const
{
    SOLVCON_PROFILE_SCOPE("Euler1DCore::velocity");
    SimpleArray<double> ret(ncoord());
    for (size_t it = 0; it < ncoord(); ++it)
    {
        ret(it) = velocity(it);
    }
    return ret;
}

SimpleArray<double> Euler1DCore::pressure() const
{
    SOLVCON_PROFILE_SCOPE("Euler1DCore::pressure");
    SimpleArray<double> ret(ncoord());
    for (size_t it = 0; it < ncoord(); ++it)
    {
        ret(it) = pressure(it);
    }
    return ret;
}

void Euler1DCore::update_cfl(bool odd_plane)
{
    SOLVCON_PROFILE_SCOPE("Euler1DCore::update_cfl");
    const int_type start = BOUND_COUNT - (odd_plane ? 1 : 0);
    const auto stop = static_cast<int_type>(ncoord() - BOUND_COUNT - (odd_plane ? 0 : 1));
    const double hdt = m_time_increment / 2;
    for (int_type it = start; it < stop; it += 2)
    {
        const double ga = m_gamma(it);
        // TODO: I didn't verify the formula.
        // wave speed.
        double wspd = m_so0(it, 1);
        wspd *= wspd;
        const double ke = wspd / (2.0 * m_so0(it, 0));
        double pr = (ga - 1.0) * (m_so0(it, 2) - ke);
        pr = (pr + std::abs(pr)) / 2.0;
        wspd = std::sqrt(ga * pr / m_so0(it, 0)) + std::sqrt(wspd) / m_so0(it, 0); // NOLINT(readability-math-missing-parentheses)
        // CFL.
        const double dxpos = m_coord(it + 1) - m_coord(it);
        const double dxneg = m_coord(it) - m_coord(it - 1);
        const double cfl = hdt * wspd / (dxpos < dxneg ? dxpos : dxneg);
        // Set back.
        m_cfl(it) = cfl;
    }
}

void Euler1DCore::march_half_so0(bool odd_plane)
{
    SOLVCON_PROFILE_SCOPE("Euler1DCore::march_half_so0");
    const int_type start = BOUND_COUNT - (odd_plane ? 1 : 0);
    const auto stop = static_cast<int_type>(ncoord() - BOUND_COUNT - (odd_plane ? 0 : 1));
    // Kernal at xneg solution element.
    Euler1DKernel kernxn{};
    kernxn
        .set_time_increment(m_time_increment);
    // Kernal at xpos solution element.
    Euler1DKernel kernxp{};
    kernxp
        .set_time_increment(m_time_increment)
        // Populate using the solution element.
        .set_value(start, m_gamma, m_coord, m_so0, m_so1)
        .derive();
    for (int_type ic = start; ic < stop; ic += 2)
    {
        // Update kernels (avoid duplicate expensiave calculation).
        kernxn = kernxp;
        kernxp
            .set_value(ic + 2, m_gamma, m_coord, m_so0, m_so1)
            .derive();
        // Calculate flux through the lower left and lower right of conservation element.
        const std::array<double, 3> flux_ll = kernxn.calc_flux_ll();
        const std::array<double, 3> flux_lr = kernxp.calc_flux_lr();
        // Calculate the variables using flux conservation.
        double const xneg = m_coord(ic);
        double const xpos = m_coord(ic + 2);
        double const dx = xpos - xneg;
        m_so0(ic + 1, 0) = (flux_ll[0] + flux_lr[0]) / dx;
        m_so0(ic + 1, 1) = (flux_ll[1] + flux_lr[1]) / dx;
        m_so0(ic + 1, 2) = (flux_ll[2] + flux_lr[2]) / dx;
    }
}

void Euler1DCore::treat_boundary_so0()
{
    /* Non-reflecting boundary condition (NRBC) type 3 with $\lambda=0$
     * (the third set in Chang 05) */
    // Set outside value from inside value.
    {
        // Left boundary.
        size_t const ic = 1;
        m_so0(ic, 0) = m_so0(ic + 1, 0);
        m_so0(ic, 1) = m_so0(ic + 1, 1);
        m_so0(ic, 2) = m_so0(ic + 1, 2);
    }
    {
        // Right boundary.
        size_t const ic = ncoord() - 2;
        m_so0(ic, 0) = m_so0(ic - 1, 0);
        m_so0(ic, 1) = m_so0(ic - 1, 1);
        m_so0(ic, 2) = m_so0(ic - 1, 2);
    }
}

void Euler1DCore::treat_boundary_so1()
{
    /* Non-reflecting boundary condition (NRBC) type 3 with $\lambda=0$
     * (the third set in Chang 05) */
    // Set outside value from inside value.
    {
        // Left boundary.
        size_t const ic = 1;
        m_so1(ic, 0) = m_so1(ic + 1, 0);
        m_so1(ic, 1) = m_so1(ic + 1, 1);
        m_so1(ic, 2) = m_so1(ic + 1, 2);
    }
    {
        // Right boundary.
        size_t const ic = ncoord() - 2;
        m_so1(ic, 0) = m_so1(ic - 1, 0);
        m_so1(ic, 1) = m_so1(ic - 1, 1);
        m_so1(ic, 2) = m_so1(ic - 1, 2);
    }
}

SimpleArray<double> Euler1DCore::temperature() const
{
    SOLVCON_PROFILE_SCOPE("Euler1DCore::temperature");
    SimpleArray<double> ret(ncoord());
    for (size_t it = 0; it < ncoord(); ++it)
    {
        ret(it) = temperature(it);
    }
    return ret;
}

SimpleArray<double> Euler1DCore::internal_energy() const
{
    SOLVCON_PROFILE_SCOPE("Euler1DCore::internal_energy");
    SimpleArray<double> ret(ncoord());
    for (size_t it = 0; it < ncoord(); ++it)
    {
        ret(it) = internal_energy(it);
    }
    return ret;
}

SimpleArray<double> Euler1DCore::entropy() const
{
    SOLVCON_PROFILE_SCOPE("Euler1DCore::entropy");
    SimpleArray<double> ret(ncoord());
    for (size_t it = 0; it < ncoord(); ++it)
    {
        ret(it) = entropy(it);
    }
    return ret;
}

} /* end namespace onedim */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
