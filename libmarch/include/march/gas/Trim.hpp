#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <memory>

#include "march/core/core.hpp"
#include "march/mesh/mesh.hpp"

#include "march/gas/Solver_decl.hpp"

namespace march {

namespace gas {

/**
 * Boundary-condition treatment.
 */
template< size_t NDIM >
class TrimBase {

public:

    using solver_type = Solver<NDIM>;
    using block_type = typename solver_type::block_type;
    using pointer = std::unique_ptr<TrimBase<NDIM>>;

    TrimBase(solver_type & solver, BoundaryData & boundary)
      : m_solver(solver)
      , m_block(*solver.block())
      , m_boundary(boundary)
    {}

    TrimBase() = delete;
    TrimBase(TrimBase const & ) = delete;
    TrimBase(TrimBase       &&) = delete;
    TrimBase & operator=(TrimBase const & ) = delete;
    TrimBase & operator=(TrimBase       &&) = delete;

    virtual ~TrimBase() {}

    virtual pointer clone() = 0;

    virtual void apply_do0() = 0;
    virtual void apply_do1() = 0;

    solver_type const & solver() const { return m_solver; }
    solver_type       & solver()       { return m_solver; }
    block_type const & block() const { return m_block; }
    block_type       & block()       { return m_block; }
    BoundaryData const & boundary() const { return m_boundary; }
    BoundaryData       & boundary()       { return m_boundary; }

private:

    solver_type & m_solver;
    block_type & m_block;
    BoundaryData & m_boundary;

}; /* end class TrimBase */


template< size_t NDIM >
class TrimNoOp : public TrimBase<NDIM> {

public:

    using base_type = TrimBase<NDIM>;
    using solver_type = typename base_type::solver_type;
    using block_type = typename base_type::block_type;
    using pointer = typename base_type::pointer;

    TrimNoOp(solver_type & solver, BoundaryData & boundary)
      : base_type(solver, boundary)
    {}

    ~TrimNoOp() override {}

    pointer clone() override {
        return pointer(new TrimNoOp<NDIM>(this->solver(), this->boundary()));
    }

    void apply_do0() override {}
    void apply_do1() override {}

}; /* end class TrimNoOp */


template< size_t NDIM >
class TrimNonRefl : public TrimBase<NDIM> {

public:

    using base_type = TrimBase<NDIM>;
    using solver_type = typename base_type::solver_type;
    using block_type = typename base_type::block_type;
    using pointer = typename base_type::pointer;

    TrimNonRefl(solver_type & solver, BoundaryData & boundary)
      : base_type(solver, boundary)
    {}

    ~TrimNonRefl() override {}

    pointer clone() override {
        return pointer(new TrimNonRefl<NDIM>(this->solver(), this->boundary()));
    }

    void apply_do0() override;
    void apply_do1() override;

}; /* end class TrimNonRefl */

template< size_t NDIM >
void TrimNonRefl<NDIM>::apply_do0() {
    auto & soln = this->solver().sol().soln;
    index_type const nbnd = this->boundary().nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        const index_type ifc = this->boundary().facn()[ibnd][0];
        auto const & tfccls = this->block().fccls()[ifc];
        const index_type icl = tfccls[0];
        const index_type jcl = tfccls[1];
        for (index_type ieq=0; ieq<solver_type::NEQ; ++ieq) {
            soln[jcl][ieq] = soln[icl][ieq];
        }
    }
}

template< size_t NDIM >
void TrimNonRefl<NDIM>::apply_do1() {
    index_type const nbnd = this->boundary().nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        const index_type ifc = this->boundary().facn()[ibnd][0];
        auto const & tfccls = this->block().fccls()[ifc];
        const index_type icl = tfccls[0];
        const index_type jcl = tfccls[1];

        auto const & tidsol  = reinterpret_cast<Vector<NDIM> const (&)[solver_type::NEQ]>(this->solver().sol().dsol [icl]);
        auto       & tjdsoln = reinterpret_cast<Vector<NDIM>       (&)[solver_type::NEQ]>(this->solver().sol().dsoln[jcl]);

        // set perpendicular gradient to zero.
        Matrix<NDIM> const mat = this->block().get_normal_matrix(ifc);
        Vector<NDIM> vec[solver_type::NEQ];
        for (index_type ieq=0; ieq<solver_type::NEQ; ++ieq) {
            vec[ieq][0] = 0.0;
            Vector<NDIM> dif = tidsol[ieq];
            for (index_type it=1; it<NDIM; ++it) {
                vec[ieq][it] = mat[it].dot(dif);
            }
        }

        // inversely transform the coordinate and set ghost gradient.
        Matrix<NDIM> const matinv = mat.transpose();
        for (index_type ieq=0; ieq<solver_type::NEQ; ++ieq) {
            tjdsoln[ieq] = product(matinv, vec[ieq]);
        }
    }
}

template< size_t NDIM >
class TrimSlipWall : public TrimBase<NDIM> {

public:

    using base_type = TrimBase<NDIM>;
    using solver_type = typename base_type::solver_type;
    using block_type = typename base_type::block_type;
    using pointer = typename base_type::pointer;

    TrimSlipWall(solver_type & solver, BoundaryData & boundary)
      : base_type(solver, boundary)
    {}

    ~TrimSlipWall() override {}

    pointer clone() override {
        return pointer(new TrimSlipWall<NDIM>(this->solver(), this->boundary()));
    }

    void apply_do0() override;
    void apply_do1() override;

}; /* end class TrimSlipWall */

template< size_t NDIM >
void TrimSlipWall<NDIM>::apply_do0() {
    auto & soln = this->solver().sol().soln;
    index_type const nbnd = this->boundary().nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        const index_type ifc = this->boundary().facn()[ibnd][0];
        auto const & tfccls = this->block().fccls()[ifc];
        const index_type icl = tfccls[0];
        const index_type jcl = tfccls[1];

        // load the original momentum vector.
        auto const & momi = *reinterpret_cast<Vector<NDIM> const *>(&soln[icl][1]);
        auto       & momj = *reinterpret_cast<Vector<NDIM>       *>(&soln[jcl][1]);

        // get rotation matrix.
        Matrix<NDIM> const mat = this->block().get_normal_matrix(ifc);
        // calculate the rotated momentum vector.
        Vector<NDIM> mom = product(mat, momi);
        // negate the normal component of the momentum vector.
        mom[0] = -mom[0];
        // rotate and set back to the outside momentum vector.
        momj = product(mat.transpose(), mom);
    }
}

template< size_t NDIM >
void TrimSlipWall<NDIM>::apply_do1() {
    index_type const nbnd = this->boundary().nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        const index_type ifc = this->boundary().facn()[ibnd][0];
        auto const & tfccls = this->block().fccls()[ifc];
        const index_type icl = tfccls[0];
        const index_type jcl = tfccls[1];

        auto const & tidsoln = reinterpret_cast<Vector<NDIM> const (&)[solver_type::NEQ]>(this->solver().sol().dsoln[icl]);
        auto       & tjdsoln = reinterpret_cast<Vector<NDIM>       (&)[solver_type::NEQ]>(this->solver().sol().dsoln[jcl]);
        auto const & titen = *reinterpret_cast<Matrix<NDIM> const *>(&tidsoln[1]);
        auto       & tjten = *reinterpret_cast<Matrix<NDIM>       *>(&tjdsoln[1]);

        Matrix<NDIM> const mat = this->block().get_normal_matrix(ifc);
        Matrix<NDIM> const matinv = mat.transpose();

        // rotate the derivatives to the normal coordinate system.
        Vector<NDIM> u1 = product(mat, tidsoln[0]);
        Vector<NDIM> um = product(mat, tidsoln[NDIM+1]);
        Matrix<NDIM> uv = product(product(mat, titen), matinv);

        // set wall condition in the rotated coordinate;
        u1[0] = -u1[0];
        um[0] = -um[0];
        for (index_type it=1; it<NDIM; ++it) {
            uv[0][it] = -uv[0][it];
            uv[it][0] = -uv[it][0];
        }

        // rotate the derivatives back to the original coordinate system.
        tjdsoln[0]      = product(matinv, u1);
        tjdsoln[NDIM+1] = product(matinv, um);
        tjten = product(product(matinv, uv), mat);

    }
}

template< size_t NDIM >
class TrimInlet : public TrimBase<NDIM> {

public:

    using base_type = TrimBase<NDIM>;
    using solver_type = typename base_type::solver_type;
    using block_type = typename base_type::block_type;
    using pointer = typename base_type::pointer;

    TrimInlet(solver_type & solver, BoundaryData & boundary)
      : base_type(solver, boundary)
    {}

    ~TrimInlet() override {}

    pointer clone() override {
        return pointer(new TrimInlet<NDIM>(this->solver(), this->boundary()));
    }

    void apply_do0() override;
    void apply_do1() override;

}; /* end class TrimInlet */

template< size_t NDIM >
void TrimInlet<NDIM>::apply_do0() {
    auto & soln = this->solver().sol().soln;
    index_type const nbnd = this->boundary().nbound();
    const auto & values = this->boundary().template values<6>();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        // load the boundary value.
        const auto & value = values[ibnd];
        const real_type density = value[0];
        const auto & vel = *reinterpret_cast<Vector<NDIM> const *>(&value[1]);
        const real_type pressure = value[4];
        const real_type gamma = value[5];
        const real_type ke = vel.square() * density / 2;

        // set the ghost value.
        const index_type ifc = this->boundary().facn()[ibnd][0];
        auto const & tfccls = this->block().fccls()[ifc];
        const index_type jcl = tfccls[1];
        auto & tjsoln = soln[jcl];

        tjsoln[0] = density;
        for (index_type it=0; it<NDIM; ++it) {
            tjsoln[1+it] = vel[it] * density;
        }
        tjsoln[1+NDIM] = pressure/(gamma-1.0) + ke;
    }
}

template< size_t NDIM >
void TrimInlet<NDIM>::apply_do1() {
    index_type const nbnd = this->boundary().nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        const index_type ifc = this->boundary().facn()[ibnd][0];
        auto const & tfccls = this->block().fccls()[ifc];
        const index_type jcl = tfccls[1];
        auto & tjdsoln = reinterpret_cast<Vector<NDIM> (&)[solver_type::NEQ]>(this->solver().sol().dsoln[jcl]);
        // set to zero gradient.
        for (index_type it=0; it<solver_type::NEQ; ++it) {
            tjdsoln[it] = 0;
        }
    }
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
