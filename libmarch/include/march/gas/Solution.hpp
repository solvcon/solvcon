#pragma once

/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>
#include <limits>
#include <memory>

#include "march/core/core.hpp"
#include "march/mesh/mesh.hpp"

namespace march {

namespace gas {

template< size_t NDIM >
struct SolutionTableTraits {
    constexpr static size_t ndim = NDIM;
    constexpr static size_t neq = NDIM+2;
}; /* end struct SolutionTableTraits */

template< size_t NDIM > class Order0Hand;

template< typename ElemType, size_t NDIM >
struct SolutionOrder0Table : public LookupTable< ElemType, SolutionTableTraits<NDIM>::neq >
{
    using table_traits = SolutionTableTraits<NDIM>;
    constexpr static size_t ndim = table_traits::ndim;
    constexpr static size_t neq = table_traits::neq;
    using base_type = LookupTable<ElemType, neq>;
    using hand_type = Order0Hand<ndim>;
    SolutionOrder0Table(index_type nghost, index_type nbody) : base_type(nghost, nbody) {}
    hand_type       hat(index_type irow)       { return hand_type(*this, irow); }
    hand_type const hat(index_type irow) const { return hand_type(*this, irow); }
}; /* end struct SolutionOrder0Table */

template< size_t NDIM > class Order1Hand;

template< typename ElemType, size_t NDIM >
struct SolutionOrder1Table : public LookupTable< ElemType, NDIM*SolutionTableTraits<NDIM>::neq >
{
    using table_traits = SolutionTableTraits<NDIM>;
    constexpr static size_t ndim = table_traits::ndim;
    constexpr static size_t neq = table_traits::neq;
    using base_type = LookupTable<ElemType, ndim*neq>;
    using hand_type = Order1Hand<ndim>;
    SolutionOrder1Table(index_type nghost, index_type nbody) : base_type(nghost, nbody) {}
    hand_type       hat(index_type irow)       { return hand_type(*this, irow); }
    hand_type const hat(index_type irow) const { return hand_type(*this, irow); }
}; /* end struct SolutionOrder0Table */

class HandBase {
public:
    HandBase() : m_ptr(nullptr) {}
    operator bool() const { return m_ptr; }
protected:
    template<class T> HandBase(T * ptr) : m_ptr((void *)ptr) {}
    template<class T> T * ptr()       { return reinterpret_cast<T *>(m_ptr); }
    template<class T> T * ptr() const { return reinterpret_cast<T *>(m_ptr); }
private:
    void * m_ptr;
}; /* end class HandBase */

/**
 * Hand is a handle to solution arrays.  The copy and move constructors copy
 * the pointer inside the other object, but assignment operators deep copy the
 * contents of the array.
 */
template< class Derived, class Traits >
class HandCRTP : public HandBase {

public:

    using derived_type = Derived;
    using trait_type = Traits;
    using table_type = typename trait_type::table_type;
    using item_reference = typename trait_type::item_reference;
    using item_const_reference = typename trait_type::item_const_reference;
    using row_type = typename trait_type::row_type;
    using row_pointer = typename trait_type::row_pointer;
    using row_reference = typename trait_type::row_reference;
    using row_const_reference = typename trait_type::row_const_reference;

    HandCRTP(table_type       & table, index_type irow) : HandCRTP(&table[irow]) {}
    HandCRTP(table_type const & table, index_type irow) : HandCRTP(&table[irow]) {}

    derived_type & assign(derived_type && other) { this->m_ptr = other.m_ptr; }

    HandCRTP(HandCRTP       && other) : HandCRTP(other.ptr()) {}
    HandCRTP(HandCRTP const &  other) : HandCRTP(other.ptr()) {}
 
    HandCRTP & operator=(HandCRTP       && other) {
        for (index_type it=0; it<table_type::neq; ++it) { (**this)[it] = (*other)[it]; }
        return *this;
    }
    HandCRTP & operator=(HandCRTP const &  other) {
        for (index_type it=0; it<table_type::neq; ++it) { (**this)[it] = (*other)[it]; }
        return *this;
    }

    row_reference       operator*()       { return *ptr(); }
    row_const_reference operator*() const { return *ptr(); }

    item_reference       operator[](index_type it)       { return (*ptr())[it]; }
    item_const_reference operator[](index_type it) const { return (*ptr())[it]; }

private:

    template<class T> HandCRTP(T * ptr) : HandBase(ptr) {
        static_assert(sizeof(Derived) == sizeof(void*), "wrong size of HandCRTP::derived_type");
    }

    row_pointer ptr()       { return HandBase::template ptr<row_type>(); }
    row_pointer ptr() const { return HandBase::template ptr<row_type>(); }

    ~HandCRTP() {}
    friend Derived;

}; /* end class HandCRTP */

template< class TableType, class ItemType >
struct HandTraits {
    using table_type = TableType;
    constexpr static size_t ndim = table_type::ndim;
    constexpr static size_t neq = table_type::neq;
    using item_type = ItemType;
    using item_reference = item_type &;
    using item_const_reference = item_type const &;
    using row_type = item_type[neq];
    using row_pointer = row_type *;
    using row_reference = row_type &;
    using row_const_reference = row_type const &;
}; /* end struct HandTraits */

template< size_t NDIM >
class Order0Hand : public HandCRTP< Order0Hand<NDIM>, HandTraits<SolutionOrder0Table<real_type, NDIM>, real_type> >
{
public:
    using base_type = HandCRTP< Order0Hand<NDIM>, HandTraits<SolutionOrder0Table<real_type, NDIM>, real_type> >;
    using base_type::base_type;
    using trait_type = HandTraits<SolutionOrder0Table<real_type, NDIM>, real_type>;
    using vector_type = Vector<NDIM>;
    Order0Hand & operator=(real_type value) {
        for (index_type it=0; it<trait_type::neq; ++it) { (**this)[it] = value; }
        return *this;
    }
    // accessors to solution quantities.
    real_type       & density()       { return (**this)[0]; }
    real_type const & density() const { return (**this)[0]; }
    vector_type       & momentum()       { return *reinterpret_cast<vector_type       *>(&(**this)[1]); }
    vector_type const & momentum() const { return *reinterpret_cast<vector_type const *>(&(**this)[1]); }
    real_type       & energy()       { return (**this)[NDIM+1]; }
    real_type const & energy() const { return (**this)[NDIM+1]; }
    // accessors physics values.
    real_type pressure(real_type gamma /* ratio of specific heat */) const {
        const real_type ke = momentum().square()/(2.0*density());
        return (gamma - 1.0) * (energy() - ke);
    }
    real_type max_wavespeed(real_type gamma /* ratio of specific heat */) const {
        const real_type density = this->density();
        const real_type momsq = momentum().square();
        const real_type ke = momsq/(2.0*density);
        real_type pr = (gamma - 1.0) * (energy() - ke);
        pr = (pr+fabs(pr))/2.0; // make the sqrt happy even with negative pressure.
        return sqrt(gamma*pr/density) + sqrt(momsq)/density;
    }
    Order0Hand set_by(
        real_type gas_constant
      , real_type gamma /* ratio of specific heat */
      , real_type density
      , real_type temperature
    ) {
        this->density() = density;
        momentum() = 0;
        energy() = density * gas_constant / (gamma-1) * temperature;
        return *this;
    }
}; /* end class Order0Hand */

template< size_t NDIM >
class Order1Hand : public HandCRTP< Order1Hand<NDIM>, HandTraits<SolutionOrder1Table<real_type, NDIM>, Vector<NDIM>> >
{
public:
    using base_type = HandCRTP< Order1Hand<NDIM>, HandTraits<SolutionOrder1Table<real_type, NDIM>, Vector<NDIM>> >;
    using base_type::base_type;
    using trait_type = HandTraits<SolutionOrder1Table<real_type, NDIM>, Vector<NDIM>>;
    using vector_type = Vector<NDIM>;
    using matrix_type = Matrix<NDIM>;
    Order1Hand & operator=(vector_type const & value) {
        for (index_type it=0; it<trait_type::neq; ++it) { (**this)[it] = value; }
        return *this;
    }
    Order1Hand & operator=(real_type value) {
        for (index_type it=0; it<trait_type::neq; ++it) { (**this)[it] = value; }
        return *this;
    }
    vector_type       & density()       { return (**this)[0]; }
    vector_type const & density() const { return (**this)[0]; }
    matrix_type       & momentum()       { return *reinterpret_cast<matrix_type       *>(&(**this)[1]); }
    matrix_type const & momentum() const { return *reinterpret_cast<matrix_type const *>(&(**this)[1]); }
    vector_type       & energy()       { return (**this)[NDIM+1]; }
    vector_type const & energy() const { return (**this)[NDIM+1]; }
}; /* end class Order1Hand */

/**
 * Solution arrays.
 */
template< size_t NDIM >
class Solution {

public:

    using table_traits = SolutionTableTraits<NDIM>;
    static constexpr size_t ndim=table_traits::ndim;
    static constexpr size_t neq=table_traits::neq;

    using o0table_type = SolutionOrder0Table<real_type, ndim>;
    using o1table_type = SolutionOrder1Table<real_type, ndim>;
    using o0hand_type = typename o0table_type::hand_type;
    using o1hand_type = typename o1table_type::hand_type;

    Solution(index_type ngstcell, index_type ncell)
      : m_so0c(ngstcell, ncell), m_so0n(ngstcell, ncell), m_so0t(ngstcell, ncell)
      , m_so1c(ngstcell, ncell), m_so1n(ngstcell, ncell)
      , m_stm(ngstcell, ncell), m_cflo(ngstcell, ncell), m_cflc(ngstcell, ncell)
    {}

    Solution() = delete;
    Solution(Solution const & ) = delete;
    Solution(Solution       &&) = delete;
    Solution operator=(Solution const & ) = delete;
    Solution operator=(Solution       &&) = delete;

    o0hand_type       so0c(index_type irow)       { return m_so0c.hat(irow); }
    o0hand_type const so0c(index_type irow) const { return m_so0c.hat(irow); }
    o0hand_type       so0n(index_type irow)       { return m_so0n.hat(irow); }
    o0hand_type const so0n(index_type irow) const { return m_so0n.hat(irow); }
    o0hand_type       so0t(index_type irow)       { return m_so0t.hat(irow); }
    o0hand_type const so0t(index_type irow) const { return m_so0t.hat(irow); }

    o1hand_type       so1c(index_type irow)       { return m_so1c.hat(irow); }
    o1hand_type const so1c(index_type irow) const { return m_so1c.hat(irow); }
    o1hand_type       so1n(index_type irow)       { return m_so1n.hat(irow); }
    o1hand_type const so1n(index_type irow) const { return m_so1n.hat(irow); }

    o0hand_type       stm(index_type irow)       { return m_stm.hat(irow); }
    o0hand_type const stm(index_type irow) const { return m_stm.hat(irow); }

    real_type & cflo(index_type irow)       { return m_cflo[irow]; }
    real_type   cflo(index_type irow) const { return m_cflo[irow]; }
    real_type & cflc(index_type irow)       { return m_cflc[irow]; }
    real_type   cflc(index_type irow) const { return m_cflc[irow]; }

    void update() {
        std::swap(m_so0c, m_so0n);
        std::swap(m_so1c, m_so1n);
    }

    struct array_access {
        Solution & sol;
        array_access(Solution & sol_in) : sol(sol_in) {}
        SolutionOrder0Table<real_type, NDIM> so0c() { return sol.m_so0c; }
        SolutionOrder0Table<real_type, NDIM> so0n() { return sol.m_so0n; }
        SolutionOrder0Table<real_type, NDIM> so0t() { return sol.m_so0t; }
        SolutionOrder1Table<real_type, NDIM> so1c() { return sol.m_so1c; }
        SolutionOrder1Table<real_type, NDIM> so1n() { return sol.m_so1n; }
        SolutionOrder0Table<real_type, NDIM> stm() { return sol.m_stm; }
        LookupTable<real_type, 0> cflo() { return sol.m_cflo; }
        LookupTable<real_type, 0> cflc() { return sol.m_cflc; }
    };

    array_access arrays() { return array_access(*this); }

private:

    SolutionOrder0Table<real_type, NDIM> m_so0c;
    SolutionOrder0Table<real_type, NDIM> m_so0n;
    SolutionOrder0Table<real_type, NDIM> m_so0t;
    SolutionOrder1Table<real_type, NDIM> m_so1c;
    SolutionOrder1Table<real_type, NDIM> m_so1n;
    SolutionOrder0Table<real_type, NDIM> m_stm;
    LookupTable<real_type, 0> m_cflo;
    LookupTable<real_type, 0> m_cflc;

}; /* end class Solution */

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
