#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>
#include <stdexcept>

#include "march/core/core.hpp"

namespace march
{

namespace mesh
{

class LookupTableCore {

private:

    Buffer m_buffer;
    index_type m_nghost = 0;
    index_type m_nbody = 0;
    index_type m_ncolumn = 0;
    index_type m_elsize = 1; ///< Element size in bytes.

protected:

    const Buffer & buffer() const { return m_buffer; }

public:

    LookupTableCore() : m_buffer() {}

    LookupTableCore(
        index_type nghost
      , index_type nbody
      , index_type ncolumn
      , index_type elsize
    )
        : m_buffer()
        , m_nghost(nghost)
        , m_nbody(nbody)
        , m_ncolumn(ncolumn)
        , m_elsize(elsize)
    {
        if (nghost < 0) { throw std::invalid_argument("negative nghost"); }
        if (nbody < 0) { throw std::invalid_argument("negative nbody"); }
        if (ncolumn < 0) { throw std::invalid_argument("negative ncolumn"); }
        if (elsize < 0) { throw std::invalid_argument("negative elsize"); }
        m_buffer = Buffer((nghost+nbody) * ncolumn * elsize);
    }

    LookupTableCore(const LookupTableCore &) = delete;

    LookupTableCore(LookupTableCore &&) = delete;

    LookupTableCore & operator=(const LookupTableCore &) = delete;

    LookupTableCore & operator=(LookupTableCore &&) = delete;

    index_type nghost() const { return m_nghost; }

    index_type nbody() const { return m_nbody; }

    index_type ncolumn() const { return m_ncolumn; }

    index_type nelem() const { return (nghost()+nbody()) * ncolumn(); }

    index_type elsize() const { return m_elsize; }

    index_type nbyte() const { return m_buffer.nbyte(); }

    char * row(index_type loc) {
        return data()+(nghost()+loc)*ncolumn()*elsize();
    }

    const char * row(index_type loc) const {
        return data()+(nghost()+loc)*ncolumn()*elsize();
    }

    /** Backdoor */
    char * data() const { return buffer().template data<char>(); }

}; /* end class LookupTableCore */

/**
 * Two-dimensional array serving as a lookup table.
 */
template< 
    typename ValueType,
    size_t NCOLUMN
>
class LookupTable: public LookupTableCore
{

public:

    using value_type = ValueType;

    LookupTable(index_type nghost, index_type nbody)
        : LookupTableCore(nghost, nbody, NCOLUMN, sizeof(ValueType))
    {}

    value_type (& operator[](index_type loc)) [NCOLUMN] {
        return *reinterpret_cast<value_type(*)[NCOLUMN]>(row(loc));
    }

    const value_type (& operator[](index_type loc) const) [NCOLUMN] {
        return *reinterpret_cast<value_type(*)[NCOLUMN]>(row(loc));
    }

    value_type (& at(index_type loc)) [NCOLUMN] {
        check_range(loc); return (*this)[loc];
    }

    const value_type (& at(index_type loc) const ) [NCOLUMN] {
        check_range(loc); return (*this)[loc];
    }

    /** Backdoor */
    value_type * data() const { return buffer().template data<value_type>(); }

private:

    void check_range(index_type loc) const {
        if (loc < -nghost() || loc >= nbody()) {
            throw std::out_of_range("LookupTable location out of range");
        }
    }

}; /* end class LookupTable */

} /* end namespace mesh */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
