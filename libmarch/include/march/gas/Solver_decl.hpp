#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>
#include <limits>
#include <memory>

#include "march/core/core.hpp"
#include "march/mesh/mesh.hpp"

#include "march/gas/Jacobian.hpp"

namespace march {

namespace gas {

template< size_t NDIM > class Quantity;

/**
 * Solution variables.
 */
template< size_t NDIM, size_t NEQ >
struct SolutionHouse {

    LookupTable<real_type, NEQ> sol;
    LookupTable<real_type, NEQ> soln;
    LookupTable<real_type, NEQ> solt;
    LookupTable<real_type, NEQ*NDIM> dsol;
    LookupTable<real_type, NEQ*NDIM> dsoln;
    LookupTable<real_type, NEQ> stm;
    LookupTable<real_type, 0> cfl;
    LookupTable<real_type, 0> ocfl;

    SolutionHouse() = delete;
    SolutionHouse(SolutionHouse const & ) = delete;
    SolutionHouse(SolutionHouse       &&) = delete;
    SolutionHouse operator=(SolutionHouse const & ) = delete;
    SolutionHouse operator=(SolutionHouse       &&) = delete;

    SolutionHouse(index_type ngstcell, index_type ncell)
      : sol(ngstcell, ncell), soln(ngstcell, ncell), solt(ngstcell, ncell)
      , dsol(ngstcell, ncell), dsoln (ngstcell, ncell)
      , stm(ngstcell, ncell), cfl(ngstcell, ncell), ocfl(ngstcell, ncell)
    {}

}; /* end struct SolutionHouse */

class SolverConstructorAgent; /* backdoor for pybind11 */

template< size_t NDIM >
class Solver
  : public std::enable_shared_from_this<Solver<NDIM>>
{

public:
    typedef int32_t int_type;
    typedef UnstructuredBlock<NDIM> block_type;
    typedef Vector<NDIM> vector_type;

    static constexpr size_t NEQ=NDIM+2;
    static constexpr size_t NSCA=1;
    static constexpr real_type TINY=1.e-60;

    static constexpr index_type FCMND = block_type::FCMND;
    static constexpr index_type CLMND = block_type::CLMND;
    static constexpr index_type CLMFC = block_type::CLMFC;
    static constexpr index_type FCNCL = block_type::FCNCL;
    static constexpr index_type FCREL = block_type::FCREL;
    static constexpr index_type BFREL = block_type::BFREL;

    struct Parameter {
        int_type sigma0=3;
        real_type taumin=0.0;
        real_type tauscale=1.0;
    }; /* end struct Parameter */

    struct State {
        real_type time=0.0;
        real_type time_increment=0.0;
    }; /* end struct State */

    struct Supplement {
        LookupTable<real_type, NSCA> amsca;
        Supplement() = delete;
        Supplement(Supplement const & ) = delete;
        Supplement(Supplement       &&) = delete;
        Supplement operator=(Supplement const & ) = delete;
        Supplement operator=(Supplement       &&) = delete;
        Supplement(index_type ngstcell, index_type ncell)
          : amsca(ngstcell, ncell)
        {}
    }; /* end struct Supplement */

    class ctor_passkey {
    private:
        ctor_passkey() = default;
        friend Solver<NDIM>;
        friend SolverConstructorAgent; /* backdoor for pybind11 */
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
    LookupTable<real_type, NDIM> const & cecnd() const { return m_cecnd; }
    Parameter const & param() const { return m_param; }
    Parameter       & param()       { return m_param; }
    State const & state() const { return m_state; }
    State       & state()       { return m_state; } 
    SolutionHouse<NDIM, NEQ> const & sol() const { return m_sol; }
    SolutionHouse<NDIM, NEQ>       & sol()       { return m_sol; }
    Supplement const & sup() const { return m_sup; }
    Supplement       & sup()       { return m_sup; }
    Quantity<NDIM> const & qty() const { return m_qty; }
    Quantity<NDIM>       & qty()       { return m_qty; }

    // TODO: move to UnstructuredBlock.
    // @[
    void locate_point(const real_type (& crd)[NDIM]) const;

    // moved to mesh: void prepare_ce();
    // moved to mesh: void prepare_sf();
    // @]

    // marching core.
    // @[
    void update(real_type time, real_type time_increment);
    void calc_cfl();
    void calc_solt();
    void calc_soln();
    void calc_dsoln();
    // @]

    // TODO: organize the methods to new boundary condition treatment types.
    // @[
    void bound_nonrefl_soln(const BoundaryData & bcd);
    void bound_nonrefl_dsoln(const BoundaryData & bcd);

    void bound_wall_soln(const BoundaryData & bcd);
    void bound_wall_dsoln(const BoundaryData & bcd);

    void bound_inlet_soln(const BoundaryData & bcd);
    void bound_inlet_dsoln(const BoundaryData & bcd);
    // @]

private:

    std::shared_ptr<block_type> m_block;
    LookupTable<real_type, NDIM> m_cecnd;
    Parameter m_param;
    State m_state;
    SolutionHouse<NDIM, NEQ> m_sol;
    Supplement m_sup;
    Quantity<NDIM> m_qty;

}; /* end class Solver */

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
