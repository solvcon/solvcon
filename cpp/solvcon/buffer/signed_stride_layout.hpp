#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <array>
#include <cstddef>
#include <type_traits>

namespace solvcon
{

namespace detail
{

/**
 * Internal mdspan layout for unique mappings with signed strides.
 *
 * The mapping shifts the logical origin so that every valid index maps to a
 * non-negative offset from the beginning of the storage span. The caller must
 * provide strides that form a unique mapping and whose normalized offsets are
 * representable by the index type.
 *
 * References:
 * https://eel.is/c++draft/mdspan.layout.reqmts
 * https://eel.is/c++draft/mdspan.layout.stride
 */
struct SignedStrideLayout
{
    template <typename Extents>
    class mapping;
};

template <typename Extents>
class SignedStrideLayout::mapping
{
public:

    using extents_type = Extents;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using layout_type = SignedStrideLayout;

    static_assert(std::is_signed_v<index_type>, "SignedStrideLayout requires a signed index type");

    constexpr mapping() noexcept;
    constexpr mapping(extents_type const & extents, std::array<index_type, extents_type::rank()> const & strides) noexcept;

    friend constexpr bool operator==(mapping const &, mapping const &) noexcept = default;

    constexpr extents_type const & extents() const noexcept { return m_extents; }

    template <typename... Indices>
    requires(
        sizeof...(Indices) == Extents::rank() &&
        (std::is_convertible_v<Indices, typename Extents::index_type> && ...) &&
        (std::is_nothrow_constructible_v<typename Extents::index_type, Indices> && ...))
    constexpr typename Extents::index_type operator()(Indices... indices) const noexcept;

    constexpr index_type required_span_size() const noexcept { return m_required_span_size; }

    constexpr bool is_unique() const noexcept { return true; }
    constexpr bool is_exhaustive() const noexcept;
    constexpr bool is_strided() const noexcept { return true; }
    constexpr index_type stride(rank_type rank) const noexcept { return m_strides[rank]; }

    static constexpr bool is_always_unique() noexcept { return true; }
    static constexpr bool is_always_exhaustive() noexcept { return false; }
    static constexpr bool is_always_strided() noexcept { return true; }

    constexpr index_type origin_offset() const noexcept { return m_origin_offset; }
    constexpr std::array<index_type, extents_type::rank()> const & strides() const noexcept { return m_strides; }

private:

    extents_type m_extents{};
    std::array<index_type, extents_type::rank()> m_strides{};
    index_type m_origin_offset{0};
    index_type m_required_span_size{0};

    constexpr void initialize_offsets() noexcept;
};

template <typename Extents>
constexpr SignedStrideLayout::mapping<Extents>::mapping() noexcept
{
    index_type stride = 1;
    for (rank_type rank = extents_type::rank(); rank > 0; --rank)
    {
        m_strides[rank - 1] = stride;
        stride *= m_extents.extent(rank - 1);
    }
    initialize_offsets();
}

template <typename Extents>
constexpr SignedStrideLayout::mapping<Extents>::mapping(
    extents_type const & extents,
    std::array<index_type, extents_type::rank()> const & strides) noexcept
    : m_extents(extents)
    , m_strides(strides)
{
    initialize_offsets();
}

template <typename Extents>
template <typename... Indices>
requires(
    sizeof...(Indices) == Extents::rank() &&
    (std::is_convertible_v<Indices, typename Extents::index_type> && ...) &&
    (std::is_nothrow_constructible_v<typename Extents::index_type, Indices> && ...))
constexpr typename Extents::index_type
SignedStrideLayout::mapping<Extents>::operator()(Indices... indices) const noexcept
{
    index_type offset = m_origin_offset;
    rank_type rank = 0;
    ((offset += static_cast<index_type>(indices) * m_strides[rank++]), ...);
    return offset;
}

template <typename Extents>
constexpr bool SignedStrideLayout::mapping<Extents>::is_exhaustive() const noexcept
{
    if (m_required_span_size == 0)
    {
        return true;
    }

    index_type element_count = 1;
    for (rank_type rank = 0; rank < extents_type::rank(); ++rank)
    {
        element_count *= m_extents.extent(rank);
    }
    return element_count == m_required_span_size;
}

template <typename Extents>
constexpr void SignedStrideLayout::mapping<Extents>::initialize_offsets() noexcept
{
    index_type min_offset = 0;
    index_type max_offset = 0;

    for (rank_type rank = 0; rank < extents_type::rank(); ++rank)
    {
        if (m_extents.extent(rank) == 0)
        {
            m_origin_offset = 0;
            m_required_span_size = 0;
            return;
        }

        index_type const axis_offset = (m_extents.extent(rank) - 1) * m_strides[rank];
        if (axis_offset < 0)
        {
            min_offset += axis_offset;
        }
        else
        {
            max_offset += axis_offset;
        }
    }

    m_origin_offset = -min_offset;
    m_required_span_size = max_offset - min_offset + 1;
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
