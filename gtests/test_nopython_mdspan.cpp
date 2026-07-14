/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/buffer.hpp>

#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(SimpleArray, mdspan_1d)
{
    namespace mm = solvcon;

    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{6});
    for (size_t i = 0; i < 6; ++i) { arr(i) = static_cast<double>(i); }

    auto ms = arr.as_mdspan<1>();
    EXPECT_EQ(ms.extent(0), 6);
    for (size_t i = 0; i < 6; ++i) { EXPECT_EQ(ms[i], arr(i)); }

    auto sp = arr.as_span();
    EXPECT_EQ(sp.size(), 6u);
    for (size_t i = 0; i < 6; ++i) { EXPECT_EQ(sp[i], arr(i)); }

    // Write through mdspan is visible via span and the underlying SimpleArray.
    ms[3] = 99.0;
    EXPECT_EQ(arr(3), 99.0);
    EXPECT_EQ(sp[3], 99.0);
}

TEST(SimpleArray, mdspan_1d_const)
{
    namespace mm = solvcon;

    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{6}, 5.0);
    const auto & carr = arr;

    auto ms = carr.as_mdspan<1>();
    static_assert(std::is_same_v<decltype(ms)::element_type, const double>);
    EXPECT_EQ(ms.extent(0), 6);
    for (size_t i = 0; i < 6; ++i) { EXPECT_EQ(ms[i], 5.0); }

    auto sp = carr.as_span();
    static_assert(std::is_same_v<decltype(sp)::element_type, const double>);
    EXPECT_EQ(sp.size(), 6u);
    for (size_t i = 0; i < 6; ++i) { EXPECT_EQ(sp[i], 5.0); }
}

TEST(SimpleArray, mdspan_1d_ghost)
{
    namespace mm = solvcon;

    // 1D array: 5 total elements, 1 ghost at the front.
    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{5});
    arr.set_nghost(1);
    for (size_t i = 0; i < 5; ++i) { arr.data(i) = static_cast<double>(i); }

    // Both views span all 5 elements via data().
    auto ms = arr.as_mdspan<1>();
    EXPECT_EQ(ms.extent(0), 5);
    for (size_t i = 0; i < 5; ++i) { EXPECT_EQ(ms[i], arr.data(i)); }

    auto sp = arr.as_span();
    EXPECT_EQ(sp.size(), 5u);
    for (size_t i = 0; i < 5; ++i) { EXPECT_EQ(sp[i], arr.data(i)); }
}

TEST(SimpleArray, mdspan_2d)
{
    namespace mm = solvcon;

    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{3, 4});
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            arr(i, j) = static_cast<double>(i * 4 + j);
        }
    }

    auto ms = arr.as_mdspan<2>();
    EXPECT_EQ(ms.extent(0), 3);
    EXPECT_EQ(ms.extent(1), 4);
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_EQ((ms[i, j]), arr(i, j));
        }
    }

    // Linear view over the C-contiguous buffer; sp[i*4+j] == arr(i, j).
    auto sp = arr.as_span();
    EXPECT_EQ(sp.size(), 12u);
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_EQ(sp[i * 4 + j], arr(i, j));
        }
    }

    // Write through mdspan is visible via span and the underlying SimpleArray.
    ms[1, 2] = 99.0;
    EXPECT_EQ(arr(1, 2), 99.0);
    EXPECT_EQ(sp[1 * 4 + 2], 99.0);
}

TEST(SimpleArray, mdspan_2d_const)
{
    namespace mm = solvcon;

    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{2, 3}, 7.0);
    const auto & carr = arr;

    auto ms = carr.as_mdspan<2>();
    static_assert(std::is_same_v<decltype(ms)::element_type, const double>);
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            EXPECT_EQ((ms[i, j]), 7.0);
        }
    }

    auto sp = carr.as_span();
    static_assert(std::is_same_v<decltype(sp)::element_type, const double>);
    EXPECT_EQ(sp.size(), 6u);
    for (size_t i = 0; i < 6; ++i) { EXPECT_EQ(sp[i], 7.0); }
}

TEST(SimpleArray, mdspan_2d_ghost)
{
    namespace mm = solvcon;

    // shape {5, 4}: 5 rows (1 ghost + 4 body), 4 columns.
    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{5, 4});
    arr.set_nghost(1);
    for (size_t idx = 0; idx < arr.size(); ++idx) { arr.data(idx) = static_cast<double>(idx); }

    auto ms = arr.as_mdspan<2>();
    EXPECT_EQ(ms.extent(0), 5);
    EXPECT_EQ(ms.extent(1), 4);

    // ms origin is data(), so ms[i, j] == data()[i * 4 + j].
    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_EQ((ms[i, j]), arr.data(i * 4 + j));
        }
    }

    // Body rows start at ms row 1 (== nghost); ms[i+1, j] == arr(i, j).
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_EQ((ms[i + 1, j]), arr(i, j));
        }
    }

    auto sp = arr.as_span();
    EXPECT_EQ(sp.size(), 20u);
    for (size_t k = 0; k < sp.size(); ++k) { EXPECT_EQ(sp[k], arr.data(k)); }
}

TEST(SimpleArray, mdspan_3d)
{
    namespace mm = solvcon;

    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{2, 3, 4});
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 4; ++k)
            {
                arr(i, j, k) = static_cast<double>((i * 3 + j) * 4 + k);
            }
        }
    }

    auto ms = arr.as_mdspan<3>();
    EXPECT_EQ(ms.extent(0), 2);
    EXPECT_EQ(ms.extent(1), 3);
    EXPECT_EQ(ms.extent(2), 4);
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 4; ++k)
            {
                EXPECT_EQ((ms[i, j, k]), arr(i, j, k));
            }
        }
    }

    auto sp = arr.as_span();
    EXPECT_EQ(sp.size(), 24u);
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 4; ++k)
            {
                EXPECT_EQ(sp[(i * 3 + j) * 4 + k], arr(i, j, k));
            }
        }
    }

    ms[1, 2, 3] = 99.0;
    EXPECT_EQ(arr(1, 2, 3), 99.0);
    EXPECT_EQ(sp[(1 * 3 + 2) * 4 + 3], 99.0);
}

TEST(SimpleArray, mdspan_3d_ghost)
{
    namespace mm = solvcon;

    // shape {4, 3, 2}: 4 slices (2 ghost + 2 body), 3 rows, 2 columns.
    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{4, 3, 2});
    arr.set_nghost(2);
    for (size_t idx = 0; idx < arr.size(); ++idx) { arr.data(idx) = static_cast<double>(idx); }

    auto ms = arr.as_mdspan<3>();
    EXPECT_EQ(ms.extent(0), 4);
    EXPECT_EQ(ms.extent(1), 3);
    EXPECT_EQ(ms.extent(2), 2);

    // ms origin is data(), so ms[i, j, k] == data()[(i * 3 + j) * 2 + k].
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 2; ++k)
            {
                EXPECT_EQ((ms[i, j, k]), arr.data((i * 3 + j) * 2 + k));
            }
        }
    }

    // Body slices start at ms index 2 (== nghost); ms[i+2, j, k] == arr(i, j, k).
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 2; ++k)
            {
                EXPECT_EQ((ms[i + 2, j, k]), arr(i, j, k));
            }
        }
    }

    auto sp = arr.as_span();
    EXPECT_EQ(sp.size(), 24u);
    for (size_t k = 0; k < sp.size(); ++k) { EXPECT_EQ(sp[k], arr.data(k)); }
}

TEST(SimpleArray, mdspan_4d)
{
    namespace mm = solvcon;

    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{2, 3, 4, 5});
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 4; ++k)
            {
                for (size_t l = 0; l < 5; ++l)
                {
                    arr(i, j, k, l) = static_cast<double>(((i * 3 + j) * 4 + k) * 5 + l);
                }
            }
        }
    }

    auto ms = arr.as_mdspan<4>();
    EXPECT_EQ(ms.extent(0), 2);
    EXPECT_EQ(ms.extent(1), 3);
    EXPECT_EQ(ms.extent(2), 4);
    EXPECT_EQ(ms.extent(3), 5);
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 4; ++k)
            {
                for (size_t l = 0; l < 5; ++l)
                {
                    EXPECT_EQ((ms[i, j, k, l]), arr(i, j, k, l));
                }
            }
        }
    }

    auto sp = arr.as_span();
    EXPECT_EQ(sp.size(), 120u);
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 4; ++k)
            {
                for (size_t l = 0; l < 5; ++l)
                {
                    EXPECT_EQ(sp[((i * 3 + j) * 4 + k) * 5 + l], arr(i, j, k, l));
                }
            }
        }
    }

    ms[1, 2, 3, 4] = 99.0;
    EXPECT_EQ(arr(1, 2, 3, 4), 99.0);
    EXPECT_EQ(sp[((1 * 3 + 2) * 4 + 3) * 5 + 4], 99.0);
}

TEST(SimpleArray, mdspan_4d_ghost)
{
    namespace mm = solvcon;

    // shape {4, 3, 2, 2}: 4 slices (1 ghost + 3 body), 3 rows, 2 columns, 2 depth.
    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{4, 3, 2, 2});
    arr.set_nghost(1);
    for (size_t idx = 0; idx < arr.size(); ++idx) { arr.data(idx) = static_cast<double>(idx); }

    auto ms = arr.as_mdspan<4>();
    EXPECT_EQ(ms.extent(0), 4);
    EXPECT_EQ(ms.extent(1), 3);
    EXPECT_EQ(ms.extent(2), 2);
    EXPECT_EQ(ms.extent(3), 2);

    // ms origin is data(), so ms[i, j, k, l] == data()[((i * 3 + j) * 2 + k) * 2 + l].
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 2; ++k)
            {
                for (size_t l = 0; l < 2; ++l)
                {
                    EXPECT_EQ((ms[i, j, k, l]), arr.data(((i * 3 + j) * 2 + k) * 2 + l));
                }
            }
        }
    }

    // Body slices start at ms index 1 (== nghost); ms[i+1, j, k, l] == arr(i, j, k, l).
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 2; ++k)
            {
                for (size_t l = 0; l < 2; ++l)
                {
                    EXPECT_EQ((ms[i + 1, j, k, l]), arr(i, j, k, l));
                }
            }
        }
    }

    auto sp = arr.as_span();
    EXPECT_EQ(sp.size(), 48u);
    for (size_t k = 0; k < sp.size(); ++k) { EXPECT_EQ(sp[k], arr.data(k)); }
}

TEST(SimpleArray, mdspan_rank_mismatch)
{
    namespace mm = solvcon;

    mm::SimpleArray<double> arr(mm::small_vector<ssize_t>{3, 4});
    EXPECT_THROW(arr.as_mdspan<3>(), std::out_of_range);
}

TEST(SimpleArray, mdspan_non_contiguous)
{
    namespace mm = solvcon;

    // Build a 3x4 view whose stride differs from the row-major layout, so the
    // array is neither C- nor F-contiguous over the underlying buffer.
    mm::small_vector<ssize_t> shape{3, 4};
    mm::small_vector<ssize_t> stride{8, 1};
    auto buffer = mm::ConcreteBuffer::construct(3 * 8 * sizeof(double));
    mm::SimpleArray<double> arr(shape, stride, buffer);
    for (size_t i = 0; i < 24; ++i) { arr.data(i) = static_cast<double>(i); }

    EXPECT_FALSE(arr.is_c_contiguous());

    auto ms = arr.as_mdspan<2>();
    EXPECT_EQ(ms.extent(0), 3);
    EXPECT_EQ(ms.extent(1), 4);
    EXPECT_EQ(ms.mapping().stride(0), 8);
    EXPECT_EQ(ms.mapping().stride(1), 1);
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            EXPECT_EQ((ms[i, j]), arr.data(i * 8 + j));
        }
    }

    auto sp = arr.as_span();
    EXPECT_EQ(sp.size(), 24u);
    for (size_t i = 0; i < sp.size(); ++i) { EXPECT_EQ(sp[i], arr.data(i)); }

    ms[2, 3] = 99.0;
    EXPECT_EQ(arr(2, 3), 99.0);
    EXPECT_EQ(sp[2 * 8 + 3], 99.0);

    sp[8] = 42.0;
    EXPECT_EQ((ms[1, 0]), 42.0);

    const auto & carr = arr;
    auto cms = carr.as_mdspan<2>();
    static_assert(std::is_same_v<decltype(cms)::element_type, const double>);
    EXPECT_EQ((cms[1, 0]), 42.0);

    auto csp = carr.as_span();
    static_assert(std::is_same_v<decltype(csp)::element_type, const double>);
    EXPECT_EQ(csp.size(), 24u);
    EXPECT_EQ(csp[2 * 8 + 3], 99.0);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
