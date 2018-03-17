#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <algorithm>
#include <cmath>

#include "march/core.hpp"
#include "march/mesh.hpp"

#include "march/gas/Solver_decl.hpp"

namespace march {

namespace gas {

/**
 * Passive quantities that may be calculated from solution data.
 */
template< size_t NDIM >
class Quantity
  : public InstanceCounter<Quantity<NDIM>>
  , public std::enable_shared_from_this<Quantity<NDIM>>
{

public:

    using solver_type = Solver<NDIM>;
    using solution_type = typename solver_type::solution_type;
    using block_type = UnstructuredBlock<NDIM>;
    using vector_type = Vector<NDIM>;

    using o0hand_type = typename solution_type::o0hand_type;
    using o1hand_type = typename solution_type::o1hand_type;

    static constexpr real_type ALMOST_ZERO = solver_type::ALMOST_ZERO;

    class ctor_passkey {
        ctor_passkey() = default;
        friend Quantity<NDIM>;
    };

    Quantity(ctor_passkey const &, solver_type const & solver)
      : m_solver(solver)
      , m_density             (solver.block()->ngstcell(), solver.block()->ncell())
      , m_velocity            (solver.block()->ngstcell(), solver.block()->ncell())
      , m_vorticity           (solver.block()->ngstcell(), solver.block()->ncell())
      , m_vorticity_magnitude (solver.block()->ngstcell(), solver.block()->ncell())
      , m_ke                  (solver.block()->ngstcell(), solver.block()->ncell())
      , m_pressure            (solver.block()->ngstcell(), solver.block()->ncell())
      , m_temperature         (solver.block()->ngstcell(), solver.block()->ncell())
      , m_soundspeed          (solver.block()->ngstcell(), solver.block()->ncell())
      , m_mach                (solver.block()->ngstcell(), solver.block()->ncell())
      , m_schlieren           (solver.block()->ngstcell(), solver.block()->ncell())
    {}

    Quantity() = delete;
    Quantity(Quantity const & ) = delete;
    Quantity(Quantity       &&) = delete;
    Quantity & operator=(Quantity const & ) = delete;
    Quantity & operator=(Quantity       &&) = delete;

    static std::shared_ptr<Quantity<NDIM>> construct(solver_type const & solver) {
        return std::make_shared<Quantity<NDIM>>(ctor_passkey(), solver);
    }

    real_type const & gasconst() const { return m_gasconst; }
    real_type       & gasconst()       { return m_gasconst; }
    real_type const & schlieren_k() const { return m_schlieren_k; }
    real_type       & schlieren_k()       { return m_schlieren_k; }
    real_type const & schlieren_k0() const { return m_schlieren_k0; }
    real_type       & schlieren_k0()       { return m_schlieren_k0; }
    real_type const & schlieren_k1() const { return m_schlieren_k1; }
    real_type       & schlieren_k1()       { return m_schlieren_k1; }

    void update() {
        update_density();
        update_velocity();
        update_vorticity();
        update_misc();
        update_schlieren();
    }

#define MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(NAME, ELEMTYPE, NDIM) \
    LookupTable<ELEMTYPE, NDIM> const & NAME() const { return m_##NAME; } \
    LookupTable<ELEMTYPE, NDIM>       & NAME()       { return m_##NAME; }
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(density             , real_type, 0   )
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(velocity            , real_type, NDIM)
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(vorticity           , real_type, NDIM)
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(vorticity_magnitude , real_type, 0   )
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(ke                  , real_type, 0   )
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(pressure            , real_type, 0   )
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(temperature         , real_type, 0   )
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(soundspeed          , real_type, 0   )
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(mach                , real_type, 0   )
    MARCH_GAS_QUANTITY_TABLE_DECL_METHODS(schlieren           , real_type, 0   )
#undef MARCH_GAS_QUANTITY_TABLE_DECL_METHODS

private:

    block_type const & block() const { return *m_solver.block(); }

    /**
     * Shift from solution point to cell center.
     */
    vector_type get_shift(index_type const icl) {
        vector_type ret = reinterpret_cast<vector_type const &>(block().clcnd()[icl]);
        ret -= reinterpret_cast<vector_type const &>(m_solver.cecnd()[icl]);
        return ret;
    }

    const o0hand_type so0n(index_type const icl) const {
        return m_solver.sol().so0n(icl);
    }

    const o1hand_type so1n(index_type const icl) const {
        return m_solver.sol().so1n(icl);
    }

    void update_density();
    void update_velocity();
    void update_vorticity();
    void update_misc();
    void update_schlieren();

    solver_type const & m_solver;
    real_type m_gasconst = 1;
    real_type m_schlieren_k = 1;
    real_type m_schlieren_k0 = 0;
    real_type m_schlieren_k1 = 1;
    LookupTable<real_type, 0> m_density;
    LookupTable<real_type, NDIM> m_velocity;
    LookupTable<real_type, NDIM> m_vorticity;
    LookupTable<real_type, 0> m_vorticity_magnitude;
    LookupTable<real_type, 0> m_ke; //< kinetic energy
    LookupTable<real_type, 0> m_pressure;
    LookupTable<real_type, 0> m_temperature;
    LookupTable<real_type, 0> m_soundspeed;
    LookupTable<real_type, 0> m_mach;
    LookupTable<real_type, 0> m_schlieren;

}; /* end class Quantity */

namespace detail {

template< size_t NDIM >
Vector<NDIM>
compute_vorticity(
    typename Solver<NDIM>::solution_type::o1hand_type const deriv
  , Vector<NDIM> const & vel
  , real_type const rho
);

template<>
inline
Vector<2>
compute_vorticity(
    typename Solver<2>::solution_type::o1hand_type const deriv
  , Vector<2> const & vel
  , real_type const rho
) {
    Vector<2> ret;
    ret[0] = ((deriv[2][0] - deriv[1][1])
            - (vel[1]*deriv[0][0] - vel[0]*deriv[0][1])) / rho;
    ret[1] = ret[0];
    return ret;
}

template<>
inline
Vector<3>
compute_vorticity(
    typename Solver<3>::solution_type::o1hand_type const deriv
  , Vector<3> const & vel
  , real_type rho
) {
    Vector<3> ret;
    ret[0] = ((deriv[3][1] - deriv[2][2])
            - (vel[2]*deriv[0][1] - vel[1]*deriv[0][2])) / rho;
    ret[1] = ((deriv[1][2] - deriv[3][0])
            - (vel[0]*deriv[0][2] - vel[2]*deriv[0][0])) / rho;
    ret[2] = ((deriv[2][0] - deriv[1][1])
            - (vel[1]*deriv[0][0] - vel[0]*deriv[0][1])) / rho;
    return ret;
}

} /* end namespace detail */

template< size_t NDIM >
void Quantity<NDIM>::update_density() {
    for (index_type icl=-block().ngstcell(); icl<block().ncell(); ++icl) {
        m_density[icl] = so0n(icl)[0] + so1n(icl)[0].dot(get_shift(icl));
    }
}

template< size_t NDIM >
void Quantity<NDIM>::update_velocity() {
    for (index_type icl=-block().ngstcell(); icl<block().ncell(); ++icl) {
        // input
        vector_type const sft = get_shift(icl);
        auto const & soln = so0n(icl);
        auto const & dsoln = so1n(icl);
        real_type const rho = m_density[icl];
        // output
        auto & tvel = reinterpret_cast<vector_type &>(m_velocity[icl]);
        for (index_type it=0; it<NDIM; ++it) {
            tvel[it] = (soln[it+1] + dsoln[it+1].dot(sft)) / rho;
        }
    }
}

template< size_t NDIM >
void Quantity<NDIM>::update_vorticity() {
    for (index_type icl=-block().ngstcell(); icl<block().ncell(); ++icl) {
        // input
        auto const & dsoln = so1n(icl);
        // output
        auto & tvor = reinterpret_cast<vector_type &>(m_vorticity[icl]);
        auto & tvorm = m_vorticity_magnitude[icl];
        tvor = detail::compute_vorticity(
            dsoln
          , reinterpret_cast<vector_type &>(m_velocity[icl])
          , m_density[icl]);
        if (NDIM == 3) { tvorm = tvor.length(); }
        else           { tvorm = fabs(tvor[0]); }
    }
}

template< size_t NDIM >
void Quantity<NDIM>::update_schlieren() {
    real_type rhogmax = 0;
    for (index_type icl=-block().ngstcell(); icl<block().ncell(); ++icl) {
        // input
        auto const & dsoln = so1n(icl);
        // output
        auto & tsch = m_schlieren[icl];
        tsch = dsoln[0].square();
        rhogmax = std::max(rhogmax, tsch);
    }
    real_type const fac0 = schlieren_k0() * rhogmax;
    real_type const fac1 = -schlieren_k() / ((schlieren_k1()-schlieren_k0()) * rhogmax + ALMOST_ZERO);
    for (index_type icl=-block().ngstcell(); icl<block().ncell(); ++icl) {
        auto & tsch = m_schlieren[icl];
        tsch = std::exp((tsch-fac0)*fac1);
    }
}

template< size_t NDIM >
void Quantity<NDIM>::update_misc() {
    for (index_type icl=-block().ngstcell(); icl<block().ncell(); ++icl) {
        // input
        auto const sft = get_shift(icl);
        auto const & soln = so0n(icl);
        auto const & dsoln = so1n(icl);
        const real_type ga = m_solver.sol().gamma(icl);
        const real_type ga1 = ga - 1;
        auto const & tvel = reinterpret_cast<vector_type &>(m_velocity[icl]);
        auto const rho = m_density[icl];
        // output
        auto & tpre = m_pressure[icl];
        auto & ttem = m_temperature[icl];
        auto & tke = m_ke[icl];
        auto & tss = m_soundspeed[icl];
        auto & tmach = m_mach[icl];
        // kinetic energy.
        tke = tvel.square() * rho / 2;
        // pressure.
        tpre = soln[NDIM+1] + dsoln[NDIM+1].dot(sft);
        tpre = (tpre - tke) * ga1;
        tpre = (tpre + fabs(tpre)) / 2; // make sure it's positive.
        // temperature.
        ttem = tpre / (rho*gasconst());
        // speed of sound.
        tss = sqrt(ga*tpre/rho);
        // Mach number.
        tmach = sqrt(tke/rho*2);
        tmach *= tss / (tss*tss + ALMOST_ZERO); // prevent nan/inf.
    }
}

template< size_t NDIM >
std::shared_ptr<Quantity<NDIM>> const & Solver<NDIM>::make_qty(bool throw_on_exist) {
    if (m_qty) {
        if (throw_on_exist) { throw std::runtime_error("Quantity already exists"); }
    } else {
        m_qty = Quantity<NDIM>::construct(*this);
    }
    return m_qty;
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
