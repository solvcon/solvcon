#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>
#include <limits>
#include <memory>

#include "march/core.hpp"
#include "march/mesh.hpp"

#include "march/gas/Solution.hpp"

namespace march {

namespace gas {

/**
 * Parameters for the CESE method.  Don't change once set.
 */
class Parameter
{

public:

    using int_type = int32_t;

    Parameter() = default;
    Parameter(Parameter const & ) = default;
    Parameter(Parameter       &&) = delete;
    Parameter & operator=(Parameter const & ) = default;
    Parameter & operator=(Parameter       &&) = delete;

    real_type   sigma0() const { return m_sigma0; }
    real_type & sigma0()       { return m_sigma0; }
    real_type   taumin() const { return m_taumin; }
    real_type & taumin()       { return m_taumin; }
    real_type   tauscale() const { return m_tauscale; }
    real_type & tauscale()       { return m_tauscale; }

private:

    real_type m_sigma0=3;
    real_type m_taumin=0.0;
    real_type m_tauscale=1.0;

#define DECL_MARCH_DEBUG(TYPE, NAME, DEFAULT) \
public: \
    TYPE   NAME() const { return m_##NAME; } \
    TYPE & NAME()       { return m_##NAME; } \
private: \
    TYPE m_##NAME = DEFAULT;

DECL_MARCH_DEBUG(real_type, stop_on_negative_density, 1.e-50)
DECL_MARCH_DEBUG(real_type, stop_on_negative_energy, 1.e-50)
DECL_MARCH_DEBUG(bool, stop_on_cfl_adjustment, true)
DECL_MARCH_DEBUG(bool, stop_on_cfl_overflow, true)

#undef DECL_MARCH_DEBUG_BOOL

}; /* end class Parameter */

struct State {
    using int_type = int32_t;

    real_type time=0.0;
    real_type time_increment=0.0;
    int_type step_current=0;
    int_type step_global=0;
    int_type substep_run=2;
    int_type substep_current=0;
    int_type report_interval=0;

    real_type cfl_min=std::numeric_limits<real_type>::quiet_NaN();
    real_type cfl_max=std::numeric_limits<real_type>::quiet_NaN();
    int_type cfl_nadjusted=-1;
    int_type cfl_nadjusted_accumulated=-1;

    std::string step_info_string() const {
        return string::format("step=%d substep=%d", step_current, substep_current);
    }
}; /* end struct State */

template< size_t NDIM > class Quantity;

template< size_t NDIM > class TrimBase;

template< size_t NDIM > class AnchorChain;

template< size_t NDIM >
class Solver
  : public InstanceCounter<Solver<NDIM>>
  , public std::enable_shared_from_this<Solver<NDIM>>
{

public:

    using int_type = State::int_type;
    using block_type = UnstructuredBlock<NDIM>;
    using anchor_chain_type = AnchorChain<NDIM>;
    using vector_type = Vector<NDIM>;
    using solution_type = Solution<NDIM>;

    static constexpr size_t ndim = solution_type::ndim;
    static constexpr size_t neq = solution_type::neq;
    static constexpr real_type TINY = 1.e-60;
    static constexpr real_type ALMOST_ZERO = 1.e-200;

    static constexpr index_type FCMND = block_type::FCMND;
    static constexpr index_type CLMND = block_type::CLMND;
    static constexpr index_type CLMFC = block_type::CLMFC;
    static constexpr index_type FCNCL = block_type::FCNCL;
    static constexpr index_type FCREL = block_type::FCREL;
    static constexpr index_type BFREL = block_type::BFREL;

    class ctor_passkey {
        ctor_passkey() = default;
        friend Solver<NDIM>;
    };

    Solver(const ctor_passkey &, const std::shared_ptr<block_type> & block);

    Solver() = delete;
    Solver(Solver const & ) = delete;
    Solver(Solver       &&) = delete;
    Solver & operator=(Solver const & ) = delete;
    Solver & operator=(Solver       &&) = delete;

    static std::shared_ptr<Solver<NDIM>> construct(const std::shared_ptr<block_type> & block) {
        return std::make_shared<Solver<NDIM>>(ctor_passkey(), block);
    }

    std::shared_ptr<block_type> const & block() const { return m_block; }
    std::vector<std::unique_ptr<TrimBase<NDIM>>> const & trims() const { return m_trims; }
    std::vector<std::unique_ptr<TrimBase<NDIM>>>       & trims()       { return m_trims; }
    AnchorChain<NDIM> const & anchors() const { return m_anchors; }
    AnchorChain<NDIM>       & anchors()       { return m_anchors; }

    LookupTable<real_type, NDIM> const & cecnd() const { return m_cecnd; }
    Parameter const & param() const { return m_param; }
    Parameter       & param()       { return m_param; }
    State const & state() const { return m_state; }
    State       & state()       { return m_state; } 
    solution_type const & sol() const { return m_sol; }
    solution_type       & sol()       { return m_sol; }
    std::shared_ptr<Quantity<NDIM>> const & qty() const { return m_qty; }
    /* no setter for m_qty */
    std::shared_ptr<Quantity<NDIM>> const & make_qty(bool throw_on_exist=false);

    // TODO: move to UnstructuredBlock.
    // @[
    void locate_point(const real_type (& crd)[NDIM]) const;

    // moved to mesh: void prepare_ce();
    // moved to mesh: void prepare_sf();
    // @]

    // marching core.
    // @[
    void update(real_type time, real_type time_increment);
    void calc_so0t();
    void calc_so0n();
    void trim_do0();
    void calc_cfl();
    void trim_do1();
    void calc_so1n();
    // @]

    void march(real_type time_current, real_type time_increment, int_type steps_run);

    void init_solution(
        real_type gas_constant
      , real_type gamma
      , real_type density
      , real_type temperature
    );

private:

    void throw_on_negative_density(const std::string & srcloc, index_type icl) const;
    void throw_on_negative_energy(const std::string & srcloc, index_type icl) const;
    void throw_on_cfl_adjustment(const std::string & srcloc, index_type icl) const;
    void throw_on_cfl_overflow(const std::string & srcloc, index_type icl) const;

private:

    std::shared_ptr<block_type> m_block;
    std::vector<std::unique_ptr<TrimBase<NDIM>>> m_trims;
    AnchorChain<NDIM> m_anchors;
    LookupTable<real_type, NDIM> m_cecnd;
    Parameter m_param;
    State m_state;
    solution_type m_sol;
    std::shared_ptr<Quantity<NDIM>> m_qty;

}; /* end class Solver */

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
