/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/signed_stride_layout.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <mdspan>
#include <type_traits>

#include <gtest/gtest.h>

TEST(SignedStrideLayout, positive_strides)
{
    namespace mm = solvcon;

    using extents_type = std::dextents<ssize_t, 2>;
    using mapping_type = mm::detail::SignedStrideLayout::mapping<extents_type>;

    mapping_type const mapping(extents_type(2, 3), std::array<ssize_t, 2>{4, 1});

    EXPECT_EQ(mapping.origin_offset(), 0);
    EXPECT_EQ(mapping.required_span_size(), 7);
    EXPECT_EQ(mapping(0, 0), 0);
    EXPECT_EQ(mapping(1, 2), 6);
    EXPECT_FALSE(mapping.is_exhaustive());
}

TEST(SignedStrideLayout, negative_strides)
{
    namespace mm = solvcon;

    using extents_type = std::dextents<ssize_t, 2>;
    using layout_type = mm::detail::SignedStrideLayout;
    using mapping_type = layout_type::mapping<extents_type>;

    static_assert(std::regular<mapping_type>);
    static_assert(std::is_nothrow_move_constructible_v<mapping_type>);
    static_assert(std::is_nothrow_move_assignable_v<mapping_type>);
    static_assert(std::is_nothrow_swappable_v<mapping_type>);

    mapping_type const mapping(extents_type(3, 4), std::array<ssize_t, 2>{-4, 1});

    static_assert(mapping_type::is_always_unique());
    static_assert(mapping_type::is_always_strided());
    EXPECT_EQ(mapping.origin_offset(), 8);
    EXPECT_EQ(mapping.required_span_size(), 12);
    EXPECT_EQ(mapping(0, 0), 8);
    EXPECT_EQ(mapping(0, 3), 11);
    EXPECT_EQ(mapping(1, 0), 4);
    EXPECT_EQ(mapping(2, 3), 3);
    EXPECT_TRUE(mapping.is_exhaustive());
}

TEST(SignedStrideLayout, mixed_strides_with_padding)
{
    namespace mm = solvcon;

    using extents_type = std::dextents<ssize_t, 3>;
    using mapping_type = mm::detail::SignedStrideLayout::mapping<extents_type>;

    mapping_type const mapping(extents_type(2, 3, 4), std::array<ssize_t, 3>{-16, 4, -1});

    EXPECT_EQ(mapping.origin_offset(), 19);
    EXPECT_EQ(mapping.required_span_size(), 28);
    EXPECT_EQ(mapping(0, 0, 0), 19);
    EXPECT_EQ(mapping(0, 2, 3), 24);
    EXPECT_EQ(mapping(1, 0, 0), 3);
    EXPECT_EQ(mapping(1, 2, 3), 8);
    EXPECT_FALSE(mapping.is_exhaustive());
}

TEST(SignedStrideLayout, empty_extent)
{
    namespace mm = solvcon;

    using extents_type = std::dextents<ssize_t, 3>;
    using mapping_type = mm::detail::SignedStrideLayout::mapping<extents_type>;

    mapping_type const mapping(extents_type(2, 0, 3), std::array<ssize_t, 3>{-3, 3, 1});

    EXPECT_EQ(mapping.origin_offset(), 0);
    EXPECT_EQ(mapping.required_span_size(), 0);
    EXPECT_TRUE(mapping.is_exhaustive());
}

TEST(SignedStrideLayout, scalar)
{
    namespace mm = solvcon;

    using extents_type = std::extents<ssize_t>;
    using mapping_type = mm::detail::SignedStrideLayout::mapping<extents_type>;

    mapping_type const mapping;

    EXPECT_EQ(mapping.origin_offset(), 0);
    EXPECT_EQ(mapping.required_span_size(), 1);
    EXPECT_EQ(mapping(), 0);
    EXPECT_TRUE(mapping.is_exhaustive());
}

TEST(SignedStrideLayout, mdspan_with_positive_strides)
{
    namespace mm = solvcon;

    using extents_type = std::dextents<ssize_t, 2>;
    using layout_type = mm::detail::SignedStrideLayout;
    using mapping_type = layout_type::mapping<extents_type>;

    mapping_type const mapping(extents_type(2, 3), std::array<ssize_t, 2>{4, 1});

    std::array<int, 7> storage{};
    std::mdspan<int, extents_type, layout_type> view(storage.data(), mapping);
    view[0, 0] = 10;
    view[1, 2] = 20;
    EXPECT_EQ(storage[0], 10);
    EXPECT_EQ(storage[6], 20);
}

TEST(SignedStrideLayout, mdspan_with_negative_strides)
{
    namespace mm = solvcon;

    using extents_type = std::dextents<ssize_t, 2>;
    using layout_type = mm::detail::SignedStrideLayout;
    using mapping_type = layout_type::mapping<extents_type>;

    mapping_type const mapping(extents_type(3, 4), std::array<ssize_t, 2>{-4, 1});

    std::array<int, 12> storage{};
    std::mdspan<int, extents_type, layout_type> view(storage.data(), mapping);
    view[0, 0] = 8;
    view[2, 3] = 3;
    EXPECT_EQ(storage[8], 8);
    EXPECT_EQ(storage[3], 3);
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:
