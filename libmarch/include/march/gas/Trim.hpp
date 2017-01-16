#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <memory>
#include <tuple>

#include "march/core/core.hpp"
#include "march/mesh/mesh.hpp"

#include "march/gas/Solver_decl.hpp"

namespace march {

namespace gas {

template< size_t NDIM >
class TrimInternal {

public:

    using solver_type = Solver<NDIM>;
    using block_type = typename solver_type::block_type;
    using o0hand_type = Order0Hand<NDIM, solver_type::NEQ>;
    using o1hand_type = Order1Hand<NDIM, solver_type::NEQ>;

    constexpr static index_type FCNCL = block_type::FCNCL;

    using fccls_row_const_reference = index_type const (&)[FCNCL];
    template< size_t NVALUE >
    using boundary_value_type = real_type[NVALUE];

    TrimInternal(solver_type & solver, BoundaryData & boundary)
      : m_solver(solver)
      , m_block(*solver.block())
      , m_boundary(boundary)
    {}

    index_type nbound() const { return m_boundary.nbound(); }
    index_type iface(index_type ibnd) const { return m_boundary.facn()[ibnd][0]; }
    fccls_row_const_reference tfccls(index_type ifc) const { return m_block.fccls()[ifc]; }
    Matrix<NDIM> get_normal_matrix(index_type ifc) const { return m_block.get_normal_matrix(ifc); }
    template< size_t NVALUE >
    boundary_value_type<NVALUE> const & value(index_type ibnd) const { return m_boundary.template values<NVALUE>()[ibnd]; }

    o0hand_type       so0n(index_type irow)       { return m_solver.sol().so0n(irow); }
    o0hand_type const so0n(index_type irow) const { return m_solver.sol().so0n(irow); }
    o1hand_type       so1c(index_type irow)       { return m_solver.sol().so1c(irow); }
    o1hand_type const so1c(index_type irow) const { return m_solver.sol().so1c(irow); }
    o1hand_type       so1n(index_type irow)       { return m_solver.sol().so1n(irow); }
    o1hand_type const so1n(index_type irow) const { return m_solver.sol().so1n(irow); }

private:

    solver_type & m_solver;
    block_type & m_block;
    BoundaryData & m_boundary;

}; /* end class TrimInternal */

/**
 * Boundary-condition treatment.
 */
template< size_t NDIM >
class TrimBase {

public:

    using pointer = std::unique_ptr<TrimBase<NDIM>>;

    using internal_type = TrimInternal<NDIM>;
    using solver_type = typename internal_type::solver_type;
    using block_type = typename internal_type::block_type;

    TrimBase(solver_type & solver, BoundaryData & boundary): m_internal(solver, boundary) {}

    TrimBase() = delete;
    TrimBase(TrimBase const & ) = delete;
    TrimBase(TrimBase       &&) = delete;
    TrimBase & operator=(TrimBase const & ) = delete;
    TrimBase & operator=(TrimBase       &&) = delete;

    virtual ~TrimBase() {}

    virtual void apply_do0() = 0;
    virtual void apply_do1() = 0;

    internal_type       & internal()       { return m_internal; }
    internal_type const & internal() const { return m_internal; }

    solver_type       & solver()       { return m_internal.solver(); }
    solver_type const & solver() const { return m_internal.solver(); }
    block_type       & block()       { return m_internal.block(); }
    block_type const & block() const { return m_internal.block(); }
    BoundaryData       & boundary()       { return m_internal.boundary(); }
    BoundaryData const & boundary() const { return m_internal.boundary(); }

private:

    TrimInternal<NDIM> m_internal;

}; /* end class TrimBase */


template< size_t NDIM >
class TrimNoOp : public TrimBase<NDIM> {

public:

    using base_type = TrimBase<NDIM>;
    using solver_type = typename base_type::solver_type;
    using block_type = typename base_type::block_type;
    using pointer = typename base_type::pointer;

    TrimNoOp(solver_type & solver, BoundaryData & boundary): base_type(solver, boundary) {}
    ~TrimNoOp() override {}
    void apply_do0() override {}
    void apply_do1() override {}

}; /* end class TrimNoOp */


template< size_t NDIM >
class TrimNonRefl : public TrimBase<NDIM> {

public:

    using base_type = TrimBase<NDIM>;
    using pointer = typename base_type::pointer;
    using solver_type = typename base_type::solver_type;
    using block_type = typename base_type::block_type;

    TrimNonRefl(solver_type & solver, BoundaryData & boundary): base_type(solver, boundary) {}

    ~TrimNonRefl() override {}

    void apply_do0() override;
    void apply_do1() override;

}; /* end class TrimNonRefl */

template< size_t NDIM >
void TrimNonRefl<NDIM>::apply_do0() {
    auto & impl = this->internal();
    index_type const nbnd = impl.nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        auto const & tfccls = impl.tfccls(impl.iface(ibnd));
        impl.so0n(tfccls[0]) = impl.so0n(tfccls[1]);
    }
}

template< size_t NDIM >
void TrimNonRefl<NDIM>::apply_do1() {
    using row_type = typename base_type::internal_type::o1hand_type::row_type;
    auto & impl = this->internal();
    index_type const nbnd = impl.nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        const index_type ifc = impl.iface(ibnd);
        auto const & tfccls = impl.tfccls(ifc);
        auto const tiso1c = impl.so1c(tfccls[0]);
        auto       pjso1n = impl.so1n(tfccls[1]);
        // set perpendicular gradient to zero.
        Matrix<NDIM> const mat = impl.get_normal_matrix(ifc);
        row_type vec;
        for (index_type ieq=0; ieq<solver_type::NEQ; ++ieq) {
            vec[ieq][0] = 0.0;
            Vector<NDIM> dif = tiso1c[ieq];
            for (index_type it=1; it<NDIM; ++it) {
                vec[ieq][it] = mat[it].dot(dif);
            }
        }
        // inversely transform the coordinate and set ghost gradient.
        Matrix<NDIM> const matinv = mat.transpose();
        for (index_type ieq=0; ieq<solver_type::NEQ; ++ieq) {
            pjso1n[ieq] = product(matinv, vec[ieq]);
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

    TrimSlipWall(solver_type & solver, BoundaryData & boundary): base_type(solver, boundary) {}

    ~TrimSlipWall() override {}

    void apply_do0() override;
    void apply_do1() override;

}; /* end class TrimSlipWall */

template< size_t NDIM >
void TrimSlipWall<NDIM>::apply_do0() {
    auto & impl = this->internal();
    index_type const nbnd = impl.nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        const index_type ifc = impl.iface(ibnd);
        auto const & tfccls = impl.tfccls(ifc);
        auto const & momi = impl.so0n(tfccls[0]).momentum();
        auto       & momj = impl.so0n(tfccls[1]).momentum();
        // get rotation matrix.
        Matrix<NDIM> const mat = impl.get_normal_matrix(ifc);
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
    auto & impl = this->internal();
    index_type const nbnd = impl.nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        const index_type ifc = impl.iface(ibnd);
        auto const & tfccls = impl.tfccls(ifc);
        auto const piso1n = impl.so1c(tfccls[0]);
        auto       pjso1n = impl.so1n(tfccls[1]);
        Matrix<NDIM> const mat = impl.get_normal_matrix(ifc);
        Matrix<NDIM> const matinv = mat.transpose();
        // rotate the derivatives to the normal coordinate system.
        Vector<NDIM> u1 = product(mat, piso1n.density());
        Vector<NDIM> um = product(mat, piso1n.energy());
        Matrix<NDIM> uv = product(product(mat, piso1n.momentum()), matinv);
        // set wall condition in the rotated coordinate;
        u1[0] = -u1[0];
        um[0] = -um[0];
        for (index_type it=1; it<NDIM; ++it) {
            uv[0][it] = -uv[0][it];
            uv[it][0] = -uv[it][0];
        }
        // rotate the derivatives back to the original coordinate system.
        pjso1n.density() = product(matinv, u1);
        pjso1n.energy() = product(matinv, um);
        pjso1n.momentum() = product(product(matinv, uv), mat);
    }
}

template< size_t NDIM >
class TrimInlet : public TrimBase<NDIM> {

public:

    constexpr static size_t NVALUE = 6;

    using base_type = TrimBase<NDIM>;
    using solver_type = typename base_type::solver_type;
    using block_type = typename base_type::block_type;
    using pointer = typename base_type::pointer;

    TrimInlet(solver_type & solver, BoundaryData & boundary): base_type(solver, boundary) {}
    ~TrimInlet() override {}
    void apply_do0() override;
    void apply_do1() override;

}; /* end class TrimInlet */

template< size_t NDIM >
void TrimInlet<NDIM>::apply_do0() {
    using boundary_value_type = typename base_type::internal_type::template boundary_value_type<NVALUE>;
    using vector_type = Vector<NDIM>;
    struct BoundaryValue {
        BoundaryValue(boundary_value_type const & ref_in) : ref(ref_in) {}
        boundary_value_type const & ref;
        // translate the array.
        real_type density() const { return ref[0]; }
        vector_type const & velocity() const { return *reinterpret_cast<vector_type const *>(&ref[1]); }
        real_type pressure() const { return ref[4]; }
        real_type gamma() const { return ref[5]; }
        real_type kinetic_energy() const { return velocity().square() * density() / 2; }
    };

    auto & impl = this->internal();
    index_type const nbnd = impl.nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        auto const & tfccls = impl.tfccls(impl.iface(ibnd));
        auto pjso0n = impl.so0n(tfccls[1]);
        BoundaryValue const val(impl.template value<NVALUE>(ibnd));
        pjso0n.density() = val.density();
        pjso0n.momentum() = val.velocity() * val.density();
        pjso0n.energy() = val.pressure()/(val.gamma()-1.0) + val.kinetic_energy();
    }
}

template< size_t NDIM >
void TrimInlet<NDIM>::apply_do1() {
    auto & impl = this->internal();
    index_type const nbnd = impl.nbound();
    for (index_type ibnd=0; ibnd<nbnd; ++ibnd) {
        auto const & tfccls = impl.tfccls(impl.iface(ibnd));
        impl.so1n(tfccls[1]) = 0;
    }
}

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
