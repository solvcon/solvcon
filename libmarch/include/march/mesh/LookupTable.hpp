#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <stdexcept>

#include "march/core/core.hpp"

namespace march
{

namespace mesh
{

/**
 * Two-dimensional array serving as a lookup table.
 */
template< 
    typename ValueType,
    size_t NCOLUMN
>
class LookupTable
{

public:

    using value_type = ValueType;

    LookupTable(const index_type nghost, const index_type nbody)
        : m_buffer()
        , m_nghost(nghost)
        , m_nbody(nbody)
    {
        check_negative(nghost, nbody);
        m_buffer = Buffer((nghost+nbody) * NCOLUMN * sizeof(value_type));
    }

    LookupTable() = delete;
    LookupTable(const LookupTable & other) = delete;
    LookupTable & operator=(const LookupTable & other) = delete;

    const index_type & nghost() const { return m_nghost; }
    const index_type & nbody() const { return m_nbody; }
    index_type nelem() const { return (nghost()+nbody()) * NCOLUMN; }

    size_t bytes() const { return m_buffer.bytes(); }

    const value_type (& row(const index_type loc) const) [NCOLUMN] {
        return *reinterpret_cast<value_type(*)[NCOLUMN]>(data()+(m_nghost+loc)*NCOLUMN);
    }

    value_type (& row(const index_type loc)) [NCOLUMN] {
        return *reinterpret_cast<value_type(*)[NCOLUMN]>(data()+(m_nghost+loc)*NCOLUMN);
    }

    const value_type (& get_row(const index_type loc) const) [NCOLUMN] {
        check_range(loc); return row(loc);
    }

    value_type (& get_row(const index_type loc)) [NCOLUMN] {
        check_range(loc); return row(loc);
    }

    /** Backdoor */
    value_type * data() const { return m_buffer.data<value_type>(); }

private:

    void check_negative(const index_type nghost, const index_type nbody) const {
        if (nghost < 0) {
            if (nbody < 0) {
                throw std::invalid_argument("negative nghost and nbody");
            } else {
                throw std::invalid_argument("negative nghost");
            }
        } else if (nbody < 0) {
            throw std::invalid_argument("negative nbody");
        }
    }

    void check_range(const index_type loc) const {
        if (loc < -m_nghost || loc >= m_nbody) {
            throw std::out_of_range("LookupTable location out of range");
        }
    }

    void check_range(const index_type loc) {
        const_cast< const LookupTable< ValueType, NCOLUMN >* >(this)->check_range(loc);
    }

    Buffer m_buffer;
    index_type m_nghost, m_nbody;

}; /* end class LookupTable */

} /* end namespace mesh */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
