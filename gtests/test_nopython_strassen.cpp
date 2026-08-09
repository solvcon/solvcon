/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/Strassen.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace
{

namespace detail = solvcon::detail;
namespace strassen = solvcon::detail::strassen;

template <typename T>
void fill_operands(ssize_t rows, ssize_t columns, ssize_t inner_size, std::vector<T> & lhs, std::vector<T> & rhs)
{
    lhs.resize(static_cast<size_t>(rows * inner_size));
    rhs.resize(static_cast<size_t>(inner_size * columns));
    for (size_t index = 0; index < lhs.size(); ++index)
    {
        lhs[index] = static_cast<T>((index * 7) % 19) / T{8};
    }
    for (size_t index = 0; index < rhs.size(); ++index)
    {
        rhs[index] = static_cast<T>((index * 5 + 3) % 23) / T{16};
    }
}

template <typename T>
strassen::Gemm<T> make_gemm(ssize_t rows, ssize_t columns, ssize_t inner_size, T const * lhs, T const * rhs, T * output)
{
    solvcon::BlasMatrixView<T> const lhs_view{lhs, inner_size, solvcon::BlasTranspose::None};
    solvcon::BlasMatrixView<T> const rhs_view{rhs, columns, solvcon::BlasTranspose::None};
    return {rows, columns, inner_size, lhs_view, rhs_view, output};
}

template <typename T>
void reference_gemm(strassen::Gemm<T> const & gemm)
{
    for (ssize_t row = 0; row < gemm.rows; ++row)
    {
        for (ssize_t column = 0; column < gemm.columns; ++column)
        {
            T total{};
            for (ssize_t inner = 0; inner < gemm.inner_size; ++inner)
            {
                total += gemm.lhs.m_data[row * gemm.lhs.m_leading_dimension + inner] *
                         gemm.rhs.m_data[inner * gemm.rhs.m_leading_dimension + column];
            }
            gemm.output[row * gemm.columns + column] = total;
        }
    }
}

template <typename T>
void expect_near(std::vector<T> const & output, std::vector<T> const & expected)
{
    ASSERT_EQ(output.size(), expected.size());
    T const epsilon = std::numeric_limits<T>::epsilon();
    for (size_t index = 0; index < output.size(); ++index)
    {
        T const tolerance = T{128} * epsilon * std::max(T{1}, std::abs(expected[index]));
        EXPECT_NEAR(output[index], expected[index], tolerance) << "index " << index;
    }
}

template <typename T, size_t Depth>
size_t run_strassen(
    ssize_t rows, ssize_t columns, ssize_t inner_size, strassen::Workspace<T> & workspace)
{
    std::vector<T> lhs;
    std::vector<T> rhs;
    fill_operands(rows, columns, inner_size, lhs, rhs);
    std::vector<T> output(static_cast<size_t>(rows * columns));
    std::vector<T> expected(static_cast<size_t>(rows * columns));
    strassen::Gemm<T> const gemm = make_gemm(rows, columns, inner_size, lhs.data(), rhs.data(), output.data());
    strassen::Gemm<T> expected_gemm = gemm;
    expected_gemm.output = expected.data();
    reference_gemm(expected_gemm);
    size_t leaf_calls = 0;

    auto const leaf = [&leaf_calls](strassen::Gemm<T> const & leaf_gemm)
    {
        ++leaf_calls;
        reference_gemm(leaf_gemm);
    };
    strassen::multiply<Depth>(gemm, workspace, leaf);

    expect_near(output, expected);
    return leaf_calls;
}

template <typename T, size_t Depth>
void check_depth(size_t expected_leaf_calls)
{
    strassen::Workspace<T> workspace;
    size_t const leaf_calls = run_strassen<T, Depth>(8, 12, 16, workspace);
    EXPECT_EQ(leaf_calls, expected_leaf_calls);
}

} /* end namespace */

TEST(StrassenKernel, matches_reference_at_each_depth)
{
    strassen::Workspace<double> workspace;
    size_t const leaf_calls = run_strassen<double, 0>(3, 5, 7, workspace);
    EXPECT_EQ(leaf_calls, 1);
    check_depth<float, 1>(7);
    check_depth<double, 1>(7);
    check_depth<float, 2>(49);
    check_depth<double, 2>(49);
}

TEST(StrassenKernel, rejects_invalid_gemm)
{
    std::vector<double> lhs(128);
    std::vector<double> rhs(192);
    std::vector<double> output(96);
    strassen::Workspace<double> workspace;
    strassen::Gemm<double> gemm = make_gemm(8, 12, 16, lhs.data(), rhs.data(), output.data());
    auto const leaf = [](strassen::Gemm<double> const &) {};

    gemm.rows = 0;
    EXPECT_THAT(
        [&]
        { strassen::multiply<1>(gemm, workspace, leaf); },
        testing::ThrowsMessage<std::invalid_argument>("Strassen GEMM dimensions must be positive"));
    gemm.rows = 7;
    EXPECT_THAT(
        [&]
        { strassen::multiply<1>(gemm, workspace, leaf); },
        testing::ThrowsMessage<std::invalid_argument>("Strassen GEMM dimensions must be divisible by 2^depth"));
    gemm.rows = 8;
    gemm.lhs.m_transpose = solvcon::BlasTranspose::Transpose;
    EXPECT_THAT(
        [&]
        { strassen::multiply<1>(gemm, workspace, leaf); },
        testing::ThrowsMessage<std::invalid_argument>("Strassen GEMM does not support transposed input views"));
    gemm.lhs.m_transpose = solvcon::BlasTranspose::None;
    gemm.lhs.m_leading_dimension = 15;
    EXPECT_THAT(
        [&]
        { strassen::multiply<1>(gemm, workspace, leaf); },
        testing::ThrowsMessage<std::invalid_argument>("Strassen GEMM input leading dimensions are too small"));
}

TEST(StrassenKernel, reuses_workspace)
{
    strassen::Workspace<double> workspace;
    run_strassen<double, 2>(8, 12, 16, workspace);
    size_t const capacity = workspace.capacity();
    run_strassen<double, 1>(6, 10, 8, workspace);
    EXPECT_EQ(workspace.capacity(), capacity);
    run_strassen<double, 2>(8, 12, 16, workspace);
    EXPECT_EQ(workspace.capacity(), capacity);
}

TEST(StrassenKernel, blas_leaf)
{
    double lhs = 2;
    double rhs = 3;
    double output = 0;
    strassen::Workspace<double> workspace;
    strassen::Gemm<double> const gemm = make_gemm(1, 1, 1, &lhs, &rhs, &output);
#if (defined(__APPLE__) && defined(__arm64__)) || defined(SC_HAS_CBLAS)
    detail::gemm_strassen<0>(gemm, workspace);
    EXPECT_EQ(output, 6);
#else
    EXPECT_THAT(
        [&]
        { detail::gemm_strassen<0>(gemm, workspace); },
        testing::ThrowsMessage<std::runtime_error>("solvcon BLAS wrapper: CBLAS backend is unavailable"));
#endif
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
