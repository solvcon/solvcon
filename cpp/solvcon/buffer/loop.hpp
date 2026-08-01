#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/small_vector.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace solvcon
{

namespace detail
{

/**
 * @brief Describe a shared runtime-rank iteration space.
 *
 * A higher-level operation plan creates one LoopDomain for every set of
 * coordinates traversed together. The domain owns only the extents and
 * defines their rank, bounds, and empty-domain behavior. Operand-specific
 * strides remain in OperandMapping, while operation-specific axes such as
 * M, N, K, or kept and reduced axes remain in the higher-level plan.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, a broadcast-capable MatmulPlan uses
 * `LoopDomain({2,5})` for the ten result-batch coordinates. M=3, N=6, and
 * K=4 remain matrix metadata rather than becoming axes of this domain.
 */
class LoopDomain
{
public:
    using shape_type = small_vector<ssize_t>;

    explicit LoopDomain(shape_type shape)
        : m_shape(std::move(shape))
    {
    }

    shape_type const & shape() const noexcept { return m_shape; }
    ssize_t extent(size_t axis) const noexcept { return m_shape[axis]; }
    size_t rank() const noexcept { return m_shape.size(); }
    size_t size() const noexcept { return std::ranges::fold_left(m_shape, size_t{1}, std::multiplies<size_t>{}); }

private:
    shape_type m_shape;
}; /* end class LoopDomain */

/**
 * @brief Describe how one operand advances over a LoopDomain.
 *
 * Each OperandMapping records one signed stride for every domain axis. This
 * separates the shared coordinate space from each operand layout: positive
 * and negative strides traverse supplied layouts, while a zero stride reuses
 * one operand position along a broadcast axis. contiguous_blocks() constructs
 * the mapping for adjacent fixed-size blocks.
 *
 * For `(2,1,3,4) @ (1,5,4,6)`, the domain is `(2,5)`. The output, lhs, and
 * rhs mappings are `{90,18}`, `{12,0}`, and `{0,24}`. The zero lhs stride
 * reuses it across axis 1, and the zero rhs stride reuses it across axis 0.
 */
class OperandMapping
{
public:
    using stride_type = small_vector<ssize_t>;

    OperandMapping() = default;
    explicit OperandMapping(stride_type strides)
        : m_strides(std::move(strides))
    {
    }

    static OperandMapping contiguous_blocks(LoopDomain const & domain, ssize_t block_size);

    size_t rank() const noexcept { return m_strides.size(); }
    ssize_t stride(size_t axis) const noexcept { return m_strides[axis]; }

private:
    stride_type m_strides;
}; /* end class OperandMapping */

/**
 * @brief Advance all operand offsets with one runtime-rank cursor.
 *
 * MappedOffsetCursor owns one coordinate counter for the LoopDomain and one
 * relative offset for each OperandMapping. advance() updates every offset
 * together, so executors do not need rank-specific nested loops or separate
 * coordinate traversal for each operand. The cursor does not dereference
 * data or execute an operation.
 *
 * With domain `(2,5)` and mappings `{90,18}`, `{12,0}`, and `{0,24}`, the
 * output, lhs, and rhs offsets start at `(0,0,0)`. They advance to
 * `(18,0,24)` for coordinate `(0,1)` and `(90,12,0)` for coordinate `(1,0)`.
 *
 * @note The cursor borrows its domain and mappings, which must outlive it.
 */
class MappedOffsetCursor
{
public:
    using mapping_type = small_vector<OperandMapping>;

    MappedOffsetCursor(LoopDomain const & domain, mapping_type const & mappings);

    explicit operator bool() const noexcept { return m_valid; }
    ssize_t offset(size_t operand) const noexcept { return m_offsets[operand]; }

    template <typename Operand>
    ssize_t offset(Operand operand) const noexcept
    {
        static_assert(std::is_enum_v<Operand>, "cursor operand must be an enum");
        return offset(static_cast<size_t>(std::to_underlying(operand)));
    }

    void advance();

private:
    LoopDomain const & m_domain;
    mapping_type const & m_mappings;
    LoopDomain::shape_type m_index;
    small_vector<ssize_t> m_offsets;
    bool m_valid = false;
}; /* end class MappedOffsetCursor */

inline OperandMapping OperandMapping::contiguous_blocks(LoopDomain const & domain, ssize_t block_size)
{
    stride_type strides(domain.rank(), block_size);
    for (size_t axis = domain.rank(); axis > 1; --axis)
    {
        strides[axis - 2] = strides[axis - 1] * domain.extent(axis - 1);
    }
    return OperandMapping(std::move(strides));
}

inline MappedOffsetCursor::MappedOffsetCursor(LoopDomain const & domain, mapping_type const & mappings)
    : m_domain(domain)
    , m_mappings(mappings)
    , m_index(domain.rank(), 0)
    , m_offsets(mappings.size(), 0)
    , m_valid(domain.size() != 0)
{
    for (OperandMapping const & mapping : mappings)
    {
        if (mapping.rank() != domain.rank())
        {
            throw std::invalid_argument(
                "operand mapping rank does not match its loop domain");
        }
    }
}

inline void MappedOffsetCursor::advance()
{
    for (size_t axis_plus_one = m_domain.rank(); axis_plus_one > 0; --axis_plus_one)
    {
        size_t const axis = axis_plus_one - 1;
        ssize_t const extent = m_domain.extent(axis);
        ++m_index[axis];
        if (m_index[axis] < extent)
        {
            for (size_t operand = 0; operand < m_mappings.size(); ++operand)
            {
                m_offsets[operand] += m_mappings[operand].stride(axis);
            }
            return;
        }

        m_index[axis] = 0;
        for (size_t operand = 0; operand < m_mappings.size(); ++operand)
        {
            m_offsets[operand] -= m_mappings[operand].stride(axis) * (extent - 1);
        }
    }
    m_valid = false;
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
