#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <stdexcept>

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
        : m_data(nullptr)
        , m_nghost(nghost)
        , m_nbody(nbody)
    {
        if (nghost < 0) {
            if (nbody < 0) {
                throw std::invalid_argument("negative nghost and nbody");
            } else {
                throw std::invalid_argument("negative nghost");
            }
        } else if (nbody < 0) {
            throw std::invalid_argument("negative nbody");
        }
        m_data = new value_type[(nghost+nbody)*NCOLUMN];
    }

    ~LookupTable() {
        delete[] m_data;
        m_data = nullptr;
        m_nghost = m_nbody = 0;
    }

    LookupTable() = delete;
    LookupTable(const LookupTable & other) = delete;
    LookupTable & operator=(const LookupTable & other) = delete;

    const index_type & nghost() const { return m_nghost; }
    const index_type & nbody() const { return m_nbody; }
    index_type nelem() const { return (nghost()+nbody()) * NCOLUMN; }

    const value_type (& row(const index_type loc) const) [NCOLUMN] {
        return *reinterpret_cast<value_type(*)[NCOLUMN]>(m_data+(m_nghost+loc)*NCOLUMN);
    }

    value_type (& row(const index_type loc)) [NCOLUMN] {
        return *reinterpret_cast<value_type(*)[NCOLUMN]>(m_data+(m_nghost+loc)*NCOLUMN);
    }

    const value_type (& getRow(const index_type loc) const) [NCOLUMN] {
        check_range(loc); return row(loc);
    }

    value_type (& getRow(const index_type loc)) [NCOLUMN] {
        check_range(loc); return row(loc);
    }

    /** Backdoor */
    value_type * data() const { return m_data; }

private:

    void check_range(const index_type loc) const {
        if (loc < -m_nghost || loc >= m_nbody) {
            throw std::out_of_range("LookupTable location out of range");
        }
    }

    void check_range(const index_type loc) {
        const_cast< const LookupTable< ValueType, NCOLUMN >* >(this)->check_range(loc);
    }

    value_type * m_data;
    index_type m_nghost, m_nbody;

}; /* end class LookupTable */

} /* end namespace mesh */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
