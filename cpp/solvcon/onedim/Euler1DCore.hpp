#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * One-dimensional Euler equation solver core using the CESE method.
 *
 * @ingroup group_onedim
 */

#include <solvcon/onedim/core.hpp>
#include <solvcon/base.hpp>
#include <solvcon/math/math.hpp>
#include <solvcon/buffer/buffer.hpp>
#include <solvcon/toggle/toggle.hpp>
#include <memory>

namespace solvcon
{

namespace onedim
{

/**
 * One-dimensional Euler solver core using the CESE method.
 *
 * @ingroup group_onedim
 *
 * It holds the NVAR (3) conserved variables (density, momentum, and
 * total energy) in the so0 array of shape (ncoord, NVAR) and their
 * spatial derivatives in so1, and marches them in space-time. Accessors
 * derive density, velocity, pressure, temperature, internal energy, and
 * entropy from the conserved variables.
 */
class Euler1DCore
    : public std::enable_shared_from_this<Euler1DCore>
{

public:

    constexpr static size_t BOUND_COUNT = 2;
    static constexpr uint8_t NVAR = 3;
    static constexpr double TINY = 1.e-100;
    static constexpr double R = 8.31446261815324;

private:

    struct ctor_passkey
    {
    }; /* end struct ctor_passkey */

public:

    std::shared_ptr<Euler1DCore> clone()
    {
        auto ret = std::make_shared<Euler1DCore>(*this);
        return ret;
    }

    template <class... Args>
    static std::shared_ptr<Euler1DCore> construct(Args &&... args)
    {
        return std::make_shared<Euler1DCore>(std::forward<Args>(args)..., ctor_passkey());
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Euler1DCore(size_t ncoord, double time_increment, ctor_passkey const &)
        : m_time_increment(time_increment)
    {
        initialize_data(ncoord);
    }

    explicit Euler1DCore(ctor_passkey const &);

    Euler1DCore() = delete;
    Euler1DCore(Euler1DCore const &) = default;
    Euler1DCore(Euler1DCore &&) = default;
    Euler1DCore & operator=(Euler1DCore const &) = default;
    Euler1DCore & operator=(Euler1DCore &&) = default;
    ~Euler1DCore() = default;

    void initialize_data(size_t ncoord);

    double time_increment() const { return m_time_increment; }

    size_t ncoord() const { return m_coord.size(); }
    SimpleArray<double> const & coord() const { return m_coord; }
    SimpleArray<double> & coord() { return m_coord; }

    SimpleArray<double> const & cfl() const { return m_cfl; }
    SimpleArray<double> & cfl() { return m_cfl; }

    SimpleArray<double> const & so0() const { return m_so0; }
    SimpleArray<double> & so0() { return m_so0; }

    SimpleArray<double> const & so1() const { return m_so1; }
    SimpleArray<double> & so1() { return m_so1; }

    SimpleArray<double> const & gamma() const { return m_gamma; }
    SimpleArray<double> & gamma() { return m_gamma; }

    double density(size_t it) const { return m_so0(it, 0); }
    SimpleArray<double> density() const;
    double velocity(size_t it) const { return m_so0(it, 1) / (m_so0(it, 0) + TINY); }
    SimpleArray<double> velocity() const;
    double pressure(size_t it) const;
    SimpleArray<double> pressure() const;
    double temperature(size_t it) const;
    SimpleArray<double> temperature() const;
    double internal_energy(size_t it) const;
    SimpleArray<double> internal_energy() const;
    double entropy(size_t it) const { return pressure(it) / std::pow(density(it), m_gamma(it)); }
    SimpleArray<double> entropy() const;

    void update_cfl(bool odd_plane);
    void march_half_so0(bool odd_plane);
    template <size_t ALPHA>
    void march_half_so1_alpha(bool odd_plane);
    void treat_boundary_so0();
    void treat_boundary_so1();

    void setup_march() { update_cfl(false); }
    template <size_t ALPHA>
    void march_half1_alpha();
    template <size_t ALPHA>
    void march_half2_alpha();
    template <size_t ALPHA>
    void march_alpha(size_t steps);

private:

    real_type m_time_increment = 0;
    SimpleArray<double> m_coord;
    SimpleArray<double> m_cfl;
    SimpleArray<double> m_so0;
    SimpleArray<double> m_so1;
    SimpleArray<double> m_gamma;
}; /* end class Euler1DCore */

std::ostream & operator<<(std::ostream & os, const Euler1DCore & sol);

inline double Euler1DCore::pressure(size_t it) const
{
    double ret = m_so0(it, 1);
    ret *= ret;
    ret /= 2.0 * m_so0(it, 0) + TINY;
    ret = m_so0(it, 2) - ret;
    ret *= m_gamma(it) - 1.0;
    return ret;
}

inline double Euler1DCore::temperature(size_t it) const
{
    double ret = (m_gamma(it) - 1.0) / R;
    ret *= m_so0(it, 2) / m_so0(it, 0) - 0.5 * pow<2>(m_so0(it, 1)) / pow<2>(m_so0(it, 0));
    return ret;
}

inline double Euler1DCore::internal_energy(size_t it) const
{
    double ret = m_so0(it, 2) / m_so0(it, 0);
    ret -= 0.5 * (pow<2>(m_so0(it, 1)) / pow<2>(m_so0(it, 0)));
    return ret;
}

/**
 * Per-element computation kernel for the one-dimensional Euler solver.
 *
 * @ingroup group_onedim
 *
 * It caches one solution element state (the conserved variables u, their
 * spatial derivative ux, and the local geometry) and derives the
 * Jacobian, flux, and space-time derivatives used to integrate the
 * conservation element flux.
 */
struct Euler1DKernel
{
    static constexpr double tiny = 1.e-100;

    Euler1DKernel() = default;

    Euler1DKernel & set_time_increment(double time_increment)
    {
        hdt = time_increment / 2.0;
        qdt = hdt / 2.0;
        return *this;
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Euler1DKernel & set_value(size_t ic, SimpleArray<double> const & gamma, SimpleArray<double> const & coord, SimpleArray<double> const & so0, SimpleArray<double> const & so1)
    {
        ga = gamma(ic);
        x = coord(ic);
        xneg = coord(ic - 1);
        xpos = coord(ic + 1);
        xctr = (xpos + xneg) * 0.5;
        u[0] = so0(ic, 0);
        u[1] = so0(ic, 1);
        u[2] = so0(ic, 2);
        ux[0] = so1(ic, 0);
        ux[1] = so1(ic, 1);
        ux[2] = so1(ic, 2);
        return *this;
    }

    Euler1DKernel & derive()
    {
        SOLVCON_PROFILE_SCOPE("Euler1DKernel::derive");

        // TODO: reduce numerical calculation.
        jac[0][0] = 0.0;
        jac[0][1] = 1.0;
        jac[0][2] = 0.0;
        jac[1][0] = (ga - 3.0) / 2.0 * u[1] * u[1] / (u[0] * u[0] + tiny);
        jac[1][1] = -(ga - 3.0) * u[1] / (u[0] + tiny);
        jac[1][2] = ga - 1.0;
        jac[2][0] = (ga - 1.0) * u[1] * u[1] * u[1] / (u[0] * u[0] * u[0] + tiny) - ga * u[1] * u[2] / (u[0] * u[0] + tiny);
        jac[2][1] = ga * u[2] / (u[0] + tiny) - 3.0 / 2.0 * (ga - 1.0) * u[1] * u[1] / (u[0] * u[0] + tiny);
        jac[2][2] = ga * u[1] / (u[0] + tiny);

        f[0] = u[1];
        f[1] = (ga - 1.0) * u[2] + (3.0 - ga) / 2.0 * u[1] * u[1] / (u[0] + tiny);
        f[2] = ga * u[1] * u[2] / (u[0] + tiny) - (ga - 1.0) / 2.0 * u[1] * u[1] * u[1] / (u[0] * u[0] + tiny);

        // Also ut = -fx
        ut[0] = -jac[0][0] * ux[0] - jac[0][1] * ux[1] - jac[0][2] * ux[2];
        ut[1] = -jac[1][0] * ux[0] - jac[1][1] * ux[1] - jac[1][2] * ux[2];
        ut[2] = -jac[2][0] * ux[0] - jac[2][1] * ux[1] - jac[2][2] * ux[2];

        // ft = d[f,u] \cdot ut
        ft[0] = jac[0][0] * ut[0] + jac[0][1] * ut[1] + jac[0][2] * ut[2];
        ft[1] = jac[1][0] * ut[0] + jac[1][1] * ut[1] + jac[1][2] * ut[2];
        ft[2] = jac[2][0] * ut[0] + jac[2][1] * ut[1] + jac[2][2] * ut[2];

        up[0] = u[0] + (x - xctr) * ux[0] /* displacement in x */ + hdt * ut[0] /* displacement in t */;
        up[1] = u[1] + (x - xctr) * ux[1] /* displacement in x */ + hdt * ut[1] /* displacement in t */;
        up[2] = u[2] + (x - xctr) * ux[2] /* displacement in x */ + hdt * ut[2] /* displacement in t */;

        return *this;
    }

    std::array<double, 3> calc_flux_ll()
    {
        const double deltax = xpos - x;
        const double dxmid = 0.5 * (x + xpos) - xctr;
        const double dxctr = x - xctr;
        double const r0 = deltax * (u[0] + dxmid * ux[0]) + hdt * (f[0] - (dxctr * ut[0]) + (qdt * ft[0]));
        double const r1 = deltax * (u[1] + dxmid * ux[1]) + hdt * (f[1] - (dxctr * ut[1]) + (qdt * ft[1]));
        double const r2 = deltax * (u[2] + dxmid * ux[2]) + hdt * (f[2] - (dxctr * ut[2]) + (qdt * ft[2]));
        return std::array<double, 3>{r0, r1, r2};
    }

    std::array<double, 3> calc_flux_lr()
    {
        const double deltax = x - xneg;
        const double dxmid = 0.5 * (x + xneg) - xctr;
        const double dxctr = x - xctr;
        double const r0 = deltax * (u[0] + dxmid * ux[0]) - hdt * (f[0] - (dxctr * ut[0]) + (qdt * ft[0]));
        double const r1 = deltax * (u[1] + dxmid * ux[1]) - hdt * (f[1] - (dxctr * ut[1]) + (qdt * ft[1]));
        double const r2 = deltax * (u[2] + dxmid * ux[2]) - hdt * (f[2] - (dxctr * ut[2]) + (qdt * ft[2]));
        return std::array<double, 3>{r0, r1, r2};
    }

    double ga; //< Heat capacity ratio.
    double qdt; //< Quarter of time increment.
    double hdt; //< Half of time increment.
    double x; //< Grid point.
    double xctr; //< Solution point.
    double xneg; //< Left point of the solution element.
    double xpos; //< Right point of the solution element.
    std::array<double, 3> u; // Variable.
    std::array<double, 3> ux; // First-order derivative of u with respect to space.
    std::array<std::array<double, 3>, 3> jac; //< Jacobian.
    std::array<double, 3> f; //< Function.
    std::array<double, 3> ut; //< First-order derivative of u with respect to time.
    std::array<double, 3> ft; //< First-order derivative of f with respect to time.
    std::array<double, 3> up; //< Derived variable.
}; /* end struct Euler1DKernel */

template <size_t ALPHA>
inline void Euler1DCore::march_half_so1_alpha(bool odd_plane)
{
    SOLVCON_PROFILE_SCOPE("Euler1DKernel::march_half_so1_alpha");

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
        // Calculate the gradient.
        for (size_t iv = 0; iv < 3; ++iv)
        {
            const double utp = m_so0(ic + 1, iv);
            const double duxn = (utp - kernxn.up[iv]) / (kernxn.xpos - kernxn.x);
            const double duxp = (kernxp.up[iv] - utp) / (kernxp.x - kernxp.xneg);
            const double fan = pow<ALPHA>(std::abs(duxn));
            const double fap = pow<ALPHA>(std::abs(duxp));
            m_so1(ic + 1, iv) = (fap * duxn + fan * duxp) / (fap + fan + Euler1DKernel::tiny);
        }
    }
}

template <size_t ALPHA>
inline void Euler1DCore::march_half1_alpha()
{
    march_half_so0(/*odd_plane*/ false);
    update_cfl(/*odd_plane*/ false);
    march_half_so1_alpha<ALPHA>(false);
}

template <size_t ALPHA>
inline void Euler1DCore::march_half2_alpha()
{
    // In the second half step, no treating boundary conditions.
    march_half_so0(/*odd_plane*/ true);
    update_cfl(/*odd_plane*/ true);
    march_half_so1_alpha<ALPHA>(true);
}

template <size_t ALPHA>
inline void Euler1DCore::march_alpha(size_t steps)
{
    for (size_t it = 0; it < steps; ++it)
    {
        march_half1_alpha<ALPHA>();
        treat_boundary_so0();
        treat_boundary_so1();
        march_half2_alpha<ALPHA>();
    }
}

} /* end namespace onedim */
} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
