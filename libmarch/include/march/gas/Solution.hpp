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
template< class Derived, class Traits, size_t NDIM, size_t NEQ >
class HandCRTP : public HandBase {

public:

    using derived_type = Derived;
    using trait_type = Traits;
    using table_type = typename trait_type::table_type;
    using elem_reference = typename trait_type::elem_reference;
    using elem_const_reference = typename trait_type::elem_const_reference;
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
        for (index_type it=0; it<NEQ; ++it) { (**this)[it] = (*other)[it]; }
        return *this;
    }
    HandCRTP & operator=(HandCRTP const &  other) {
        for (index_type it=0; it<NEQ; ++it) { (**this)[it] = (*other)[it]; }
        return *this;
    }

    row_reference       operator*()       { return *ptr(); }
    row_const_reference operator*() const { return *ptr(); }

    elem_reference       operator[](index_type it)       { return (*ptr())[it]; }
    elem_const_reference operator[](index_type it) const { return (*ptr())[it]; }

private:

    template<class T> HandCRTP(T * ptr) : HandBase(ptr) {
        static_assert(sizeof(Derived) == sizeof(void*), "wrong size of HandCRTP::derived_type");
    }

    row_pointer ptr()       { return HandBase::template ptr<row_type>(); }
    row_pointer ptr() const { return HandBase::template ptr<row_type>(); }

    ~HandCRTP() {}
    friend Derived;

}; /* end class HandCRTP */

template< class TableType, class ElemType, size_t NDIM, size_t NEQ >
struct HandTraits {
    using vector_type = Vector<NDIM>;
    using matrix_type = Matrix<NDIM>;
    using table_type = TableType;
    using elem_type = ElemType;
    using elem_reference = elem_type &;
    using elem_const_reference = elem_type const &;
    using row_type = elem_type[NEQ];
    using row_pointer = row_type *;
    using row_reference = row_type &;
    using row_const_reference = row_type const &;
}; /* end struct HandTraits */

#define MARCH_DECL_HAND_TRAITS_BODY \
    using vector_type = typename base_trait_t::vector_type; \
    using matrix_type = typename base_trait_t::matrix_type; \
    using table_type = typename base_trait_t::table_type; \
    using elem_type = typename base_trait_t::elem_type; \
    using elem_reference = typename base_trait_t::elem_reference; \
    using elem_const_reference = typename base_trait_t::elem_const_reference; \
    using row_type = typename base_trait_t::row_type; \
    using row_pointer = typename base_trait_t::row_pointer; \
    using row_reference = typename base_trait_t::row_reference; \
    using row_const_reference = typename base_trait_t::row_const_reference;
// end MARCH_DECL_HAND_TRAITS_BODY
template< size_t NDIM, size_t NEQ >
struct Order0HandTraits : HandTraits<LookupTable<real_type, NEQ>, real_type, NDIM, NEQ> {
    using base_trait_t = HandTraits<LookupTable<real_type, NEQ>, real_type, NDIM, NEQ>;
    MARCH_DECL_HAND_TRAITS_BODY
}; /* end struct Order0HandTraits */
template< size_t NDIM, size_t NEQ >
struct Order1HandTraits : HandTraits<LookupTable<real_type, NEQ*NDIM>, Vector<NDIM>, NDIM, NEQ> {
    using base_trait_t = HandTraits<LookupTable<real_type, NEQ*NDIM>, Vector<NDIM>, NDIM, NEQ>;
    MARCH_DECL_HAND_TRAITS_BODY
}; /* end struct Order1HandTraits */
#undef MARCH_DECL_HAND_TRAITS_BODY

template< size_t NDIM, size_t NEQ >
class Order0Hand : public HandCRTP< Order0Hand<NDIM, NEQ>, Order0HandTraits<NDIM, NEQ>, NDIM, NEQ >
{
public:
    using base_type = HandCRTP< Order0Hand<NDIM, NEQ>, Order0HandTraits<NDIM, NEQ>, NDIM, NEQ >;
    using base_type::base_type;
    using trait_type = Order0HandTraits<NDIM, NEQ>;
    using vector_type = typename trait_type::vector_type;
    Order0Hand & operator=(real_type value) {
        for (index_type it=0; it<NEQ; ++it) { (**this)[it] = value; }
        return *this;
    }
    real_type       & density()       { return (**this)[0]; }
    real_type const & density() const { return (**this)[0]; }
    vector_type       & momentum()       { return *reinterpret_cast<vector_type       *>(&(**this)[1]); }
    vector_type const & momentum() const { return *reinterpret_cast<vector_type const *>(&(**this)[1]); }
    real_type       & energy()       { return (**this)[NDIM+1]; }
    real_type const & energy() const { return (**this)[NDIM+1]; }
}; /* end class Order0Hand */

template< size_t NDIM, size_t NEQ >
class Order1Hand : public HandCRTP< Order1Hand<NDIM, NEQ>, Order1HandTraits<NDIM, NEQ>, NDIM, NEQ >
{
public:
    using base_type = HandCRTP< Order1Hand<NDIM, NEQ>, Order1HandTraits<NDIM, NEQ>, NDIM, NEQ >;
    using base_type::base_type;
    using trait_type = Order1HandTraits<NDIM, NEQ>;
    using vector_type = typename trait_type::vector_type;
    using matrix_type = typename trait_type::matrix_type;
    Order1Hand & operator=(vector_type const & value) {
        for (index_type it=0; it<NEQ; ++it) { (**this)[it] = value; }
        return *this;
    }
    Order1Hand & operator=(real_type value) {
        for (index_type it=0; it<NEQ; ++it) { (**this)[it] = value; }
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
template< size_t NDIM, size_t NEQ >
class Solution {

public:

    Solution() = delete;
    Solution(Solution const & ) = delete;
    Solution(Solution       &&) = delete;
    Solution operator=(Solution const & ) = delete;
    Solution operator=(Solution       &&) = delete;

    Solution(index_type ngstcell, index_type ncell)
      : m_so0c(ngstcell, ncell), m_so0n(ngstcell, ncell), m_so0t(ngstcell, ncell)
      , m_so1c(ngstcell, ncell), m_so1n(ngstcell, ncell)
      , m_stm(ngstcell, ncell), m_cflo(ngstcell, ncell), m_cflc(ngstcell, ncell)
    {}

    using o0hand_type = Order0Hand<NDIM, NEQ>;
    using o1hand_type = Order1Hand<NDIM, NEQ>;

    o0hand_type       so0c(index_type irow)       { return o0hand_type(m_so0c, irow); }
    o0hand_type const so0c(index_type irow) const { return o0hand_type(m_so0c, irow); }
    o0hand_type       so0n(index_type irow)       { return o0hand_type(m_so0n, irow); }
    o0hand_type const so0n(index_type irow) const { return o0hand_type(m_so0n, irow); }
    o0hand_type       so0t(index_type irow)       { return o0hand_type(m_so0t, irow); }
    o0hand_type const so0t(index_type irow) const { return o0hand_type(m_so0t, irow); }

    o1hand_type       so1c(index_type irow)       { return o1hand_type(m_so1c, irow); }
    o1hand_type const so1c(index_type irow) const { return o1hand_type(m_so1c, irow); }
    o1hand_type       so1n(index_type irow)       { return o1hand_type(m_so1n, irow); }
    o1hand_type const so1n(index_type irow) const { return o1hand_type(m_so1n, irow); }

    o0hand_type       stm(index_type irow)       { return o0hand_type(m_stm, irow); }
    o0hand_type const stm(index_type irow) const { return o0hand_type(m_stm, irow); }
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
        LookupTable<real_type, NEQ> so0c() { return sol.m_so0c; }
        LookupTable<real_type, NEQ> so0n() { return sol.m_so0n; }
        LookupTable<real_type, NEQ> so0t() { return sol.m_so0t; }
        LookupTable<real_type, NEQ*NDIM> so1c() { return sol.m_so1c; }
        LookupTable<real_type, NEQ*NDIM> so1n() { return sol.m_so1n; }
        LookupTable<real_type, NEQ> stm() { return sol.m_stm; }
        LookupTable<real_type, 0> cflo() { return sol.m_cflo; }
        LookupTable<real_type, 0> cflc() { return sol.m_cflc; }
    };

    array_access arrays() { return array_access(*this); }

private:

    LookupTable<real_type, NEQ> m_so0c;
    LookupTable<real_type, NEQ> m_so0n;
    LookupTable<real_type, NEQ> m_so0t;
    LookupTable<real_type, NEQ*NDIM> m_so1c;
    LookupTable<real_type, NEQ*NDIM> m_so1n;
    LookupTable<real_type, NEQ> m_stm;
    LookupTable<real_type, 0> m_cflo;
    LookupTable<real_type, 0> m_cflc;

}; /* end class Solution */

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
