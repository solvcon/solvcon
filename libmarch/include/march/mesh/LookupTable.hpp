#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>
#include <stdexcept>
#include <iterator>
#include <vector>

#include "march/core/core.hpp"

namespace march
{

namespace mesh
{

/**
 * Untyped unresizeable lookup table.
 */
class LookupTableCore {

private:

    Buffer m_buffer;
    std::vector<index_type> m_dims;
    index_type m_nghost = 0;
    index_type m_nbody = 0;
    index_type m_ncolumn = 1;
    index_type m_elsize = 1; ///< Element size in bytes.

protected:

    const Buffer & buffer() const { return m_buffer; }

public:

    LookupTableCore() : m_buffer(), m_dims() {
        static_assert(sizeof(LookupTableCore) == 64, "LookupTableCore size changes");
    }

    /**
     * \param[in] nghost  Number of ghost (negative index) rows.
     * \param[in] nbody   Number of body (non-negative index) rows.
     * \param[in] dims    The shape of the table, including the combined row
     *                    number.
     * \param[in] elsize  Number of bytes per data element.
     */
    LookupTableCore(
        index_type nghost
      , index_type nbody
      , const std::vector<index_type> & dims
      , index_type elsize
    )
        : m_buffer()
        , m_dims(dims)
        , m_nghost(nghost)
        , m_nbody(nbody)
        , m_elsize(elsize)
    {
        m_ncolumn = verify(nghost, nbody, dims, elsize);
        m_buffer = Buffer((nghost+nbody) * m_ncolumn * elsize);
    }

    /**
     * When given an allocated memory block (\p data) from outside, its Buffer
     * object doesn't manage its own memory.
     *
     * This constructor allows the ownership of the memory block can be
     * transferred to an outside system, like NumPy.
     *
     * \param[in] nghost  Number of ghost (negative index) rows.
     * \param[in] nbody   Number of body (non-negative index) rows.
     * \param[in] dims    The shape of the table, including the combined row
     *                    number.
     * \param[in] elsize  Number of bytes per data element.
     * \param[in] data    The memory block.
     */
    LookupTableCore(
        index_type nghost
      , index_type nbody
      , const std::vector<index_type> & dims
      , index_type elsize
      , char * data
    )
        : m_buffer()
        , m_dims(dims)
        , m_nghost(nghost)
        , m_nbody(nbody)
        , m_elsize(elsize)
    {
        m_ncolumn = verify(nghost, nbody, dims, elsize);
        m_buffer = Buffer((nghost+nbody) * m_ncolumn * elsize, data);
    }

    LookupTableCore(const LookupTableCore &) = delete;

    LookupTableCore(LookupTableCore &&) = delete;

    LookupTableCore & operator=(const LookupTableCore &) = delete;

    LookupTableCore & operator=(LookupTableCore &&) = delete;

    const std::vector<index_type> & dims() const { return m_dims; }

    index_type ndim() const { return m_dims.size(); }

    index_type nghost() const { return m_nghost; }

    index_type nbody() const { return m_nbody; }

    index_type ncolumn() const { return m_ncolumn; }

    index_type nelem() const { return (nghost()+nbody()) * ncolumn(); }

    index_type elsize() const { return m_elsize; }

    size_t nbyte() const { return m_buffer.nbyte(); }

    /**
     * Pointer at the beginning of the row.
     */
    char * row(index_type loc) {
        return data()+(nghost()+loc)*ncolumn()*elsize();
    }

    /**
     * Pointer at the beginning of the row.
     */
    const char * row(index_type loc) const {
        return data()+(nghost()+loc)*ncolumn()*elsize();
    }

    /** Backdoor */
    char * data() const { return buffer().template data<char>(); }

private:

    /**
     * Verify the shape.
     *
     * \param[in] nghost  Number of ghost (negative index) rows.
     * \param[in] nbody   Number of body (non-negative index) rows.
     * \param[in] dims    The shape of the table, including the combined row
     *                    number.
     * \param[in] elsize  Number of bytes per data element.
     */
    index_type verify(
        index_type nghost
      , index_type nbody
      , const std::vector<index_type> & dims
      , index_type elsize
    ) const {
        if (nghost < 0) { throw std::invalid_argument("negative nghost"); }
        if (nbody < 0) { throw std::invalid_argument("negative nbody"); }
        if (dims.size() == 0) { throw std::invalid_argument("empty dims"); }
        if (dims[0] != (nghost + nbody)) { throw std::invalid_argument("dims[0] != nghost + nbody"); }
        index_type ncolumn = 1;
        if (dims.size() > 1) {
            for (auto it=std::next(dims.begin()); it!=dims.end(); ++it) {
                ncolumn *= *it;
            }
        }
        if (ncolumn < 0) { throw std::invalid_argument("negative ncolumn"); }
        if (elsize < 0) { throw std::invalid_argument("negative elsize"); }
        return ncolumn;
    }

}; /* end class LookupTableCore */

/**
 * Typed unresizeable lookup table.
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
        : LookupTableCore(nghost, nbody, std::vector<index_type>({nghost+nbody, NCOLUMN}), sizeof(ValueType))
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
