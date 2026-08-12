/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/matmul.hpp>
#include <solvcon/math/Winograd.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace solvcon
{

namespace detail
{

namespace winograd
{

namespace
{

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
void fill_operands(
    ssize_t rows,
    ssize_t columns,
    ssize_t inner_size,
    std::vector<Complex<T>> & lhs,
    std::vector<Complex<T>> & rhs)
{
    lhs.resize(static_cast<size_t>(rows * inner_size));
    rhs.resize(static_cast<size_t>(inner_size * columns));
    for (size_t index = 0; index < lhs.size(); ++index)
    {
        lhs[index] = {static_cast<T>((index * 7) % 19) / T{8},
                      static_cast<T>((index * 11 + 1) % 17) / T{16}};
    }
    for (size_t index = 0; index < rhs.size(); ++index)
    {
        rhs[index] = {static_cast<T>((index * 5 + 3) % 23) / T{16},
                      static_cast<T>((index * 13 + 2) % 29) / T{32}};
    }
}

template <typename T>
BlasGemmOperation<T> make_gemm(
    ssize_t rows,
    ssize_t columns,
    ssize_t inner_size,
    T const * lhs,
    T const * rhs,
    T * output)
{
    return {
        .rows = rows,
        .columns = columns,
        .inner_size = inner_size,
        .lhs = {
            .m_data = lhs,
            .m_leading_dimension = inner_size,
            .m_transpose = BlasTranspose::None,
        },
        .rhs = {
            .m_data = rhs,
            .m_leading_dimension = columns,
            .m_transpose = BlasTranspose::None,
        },
        .output = {
            .m_data = output,
            .m_leading_dimension = columns,
        },
        .alpha = T{1},
        .beta = T{0},
    };
}

template <typename T>
void reference_gemm(BlasGemmOperation<T> const & gemm)
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
            T & output = gemm.output.m_data[row * gemm.output.m_leading_dimension + column];
            output = gemm.alpha * total + gemm.beta * output;
        }
    }
}

template <typename T>
auto magnitude(T const & value)
{
    if constexpr (is_complex_v<T>)
    {
        return std::abs(value.to_std_complex());
    }
    else
    {
        return std::abs(value);
    }
}

template <typename T>
void expect_near(std::vector<T> const & output, std::vector<T> const & expected)
{
    ASSERT_EQ(output.size(), expected.size());
    using scalar_type = decltype(magnitude(T{}));
    scalar_type const epsilon = std::numeric_limits<scalar_type>::epsilon();
    for (size_t index = 0; index < output.size(); ++index)
    {
        scalar_type const tolerance = scalar_type{128} * epsilon * std::max(scalar_type{1}, magnitude(expected[index]));
        EXPECT_LE(magnitude(output[index] - expected[index]), tolerance) << "index " << index;
    }
}

template <typename T>
size_t run_winograd(
    ssize_t rows,
    ssize_t columns,
    ssize_t inner_size,
    std::vector<T> const & lhs,
    std::vector<T> const & rhs)
{
    std::vector<T> output(static_cast<size_t>(rows * columns), T{13});
    std::vector<T> expected(static_cast<size_t>(rows * columns), T{-7});
    BlasGemmOperation<T> const gemm = make_gemm(rows, columns, inner_size, lhs.data(), rhs.data(), output.data());
    BlasGemmOperation<T> reference = gemm;
    reference.output.m_data = expected.data();
    reference_gemm(reference);
    size_t product_calls = 0;

    auto const multiply_product = [&product_calls](BlasGemmOperation<T> const & product)
    {
        ++product_calls;
        reference_gemm(product);
    };
    multiply(gemm, multiply_product);

    expect_near(output, expected);
    return product_calls;
}

template <typename T>
size_t run_winograd(ssize_t rows, ssize_t columns, ssize_t inner_size)
{
    std::vector<T> lhs;
    std::vector<T> rhs;
    fill_operands(rows, columns, inner_size, lhs, rhs);
    return run_winograd(rows, columns, inner_size, lhs, rhs);
}

template <typename T>
void check_blas_winograd(ssize_t rows, ssize_t columns, ssize_t inner_size)
{
    std::vector<T> lhs;
    std::vector<T> rhs;
    fill_operands(rows, columns, inner_size, lhs, rhs);
    std::vector<T> output(static_cast<size_t>(rows * columns), T{13});
    std::vector<T> expected(static_cast<size_t>(rows * columns), T{-7});
    BlasGemmOperation<T> const gemm = make_gemm(rows, columns, inner_size, lhs.data(), rhs.data(), output.data());
    BlasGemmOperation<T> reference = gemm;
    reference.output.m_data = expected.data();
    reference_gemm(reference);

    gemm_winograd(rows, columns, inner_size, gemm.lhs, gemm.rhs, gemm.output);
    expect_near(output, expected);
}

template <typename T>
void check_winograd_padded_output(ssize_t rows, ssize_t columns, ssize_t inner_size)
{
    std::vector<T> lhs;
    std::vector<T> rhs;
    fill_operands(rows, columns, inner_size, lhs, rhs);
    ssize_t const output_stride = columns + 3;
    T const sentinel = T{-12345};
    std::vector<T> output(static_cast<size_t>(rows * output_stride), sentinel);
    std::vector<T> actual(static_cast<size_t>(rows * columns));
    std::vector<T> expected(static_cast<size_t>(rows * columns));

    BlasGemmOperation<T> gemm = make_gemm(rows, columns, inner_size, lhs.data(), rhs.data(), output.data());
    gemm.output.m_leading_dimension = output_stride;
    BlasGemmOperation<T> reference = make_gemm(rows, columns, inner_size, lhs.data(), rhs.data(), expected.data());
    reference_gemm(reference);

    multiply(gemm, reference_gemm<T>);

    for (ssize_t row = 0; row < rows; ++row)
    {
        T const * output_row = output.data() + row * output_stride;
        std::copy_n(output_row, columns, actual.data() + row * columns);
        for (ssize_t column = columns; column < output_stride; ++column)
        {
            EXPECT_EQ(output_row[column], sentinel);
        }
    }
    expect_near(actual, expected);
}

template <typename T>
void check_winograd_cancellation()
{
    std::vector<T> const lhs{T{4096}, T{1}, T{4095}, T{1}};
    std::vector<T> const rhs{T{1} / T{4096}, T{2} / T{4096}, T{2} / T{4096}, T{3} / T{4096}};
    EXPECT_EQ(run_winograd(2, 2, 2, lhs, rhs), 7);
}

} /* end namespace */

TEST(WinogradKernel, matches_rectangular_reference)
{
    constexpr std::array<std::array<ssize_t, 3>, 4> shapes{{
        {8, 12, 16},
        {16, 8, 12},
        {8, 16, 12},
        {16, 16, 8},
    }};
    for (auto const & shape : shapes)
    {
        EXPECT_EQ(run_winograd<float>(shape[0], shape[1], shape[2]), 7);
        EXPECT_EQ(run_winograd<double>(shape[0], shape[1], shape[2]), 7);
        EXPECT_EQ(run_winograd<Complex<float>>(shape[0], shape[1], shape[2]), 7);
        EXPECT_EQ(run_winograd<Complex<double>>(shape[0], shape[1], shape[2]), 7);
    }
}

TEST(WinogradKernel, rejects_invalid_gemm)
{
    std::vector<double> lhs(128);
    std::vector<double> rhs(192);
    std::vector<double> output(96);
    BlasGemmOperation<double> gemm = make_gemm(8, 12, 16, lhs.data(), rhs.data(), output.data());
    auto const multiply_product = [](BlasGemmOperation<double> const &) {};
    auto const multiply_gemm = [&]
    { multiply(gemm, multiply_product); };

    gemm.rows = 0;
    EXPECT_THAT(
        multiply_gemm,
        testing::ThrowsMessage<std::invalid_argument>("Winograd GEMM dimensions must be positive"));
    gemm.rows = 7;
    EXPECT_THAT(
        multiply_gemm,
        testing::ThrowsMessage<std::invalid_argument>("Winograd GEMM dimensions must be even"));
    gemm.rows = 8;
    gemm.lhs.m_transpose = BlasTranspose::Transpose;
    EXPECT_THAT(
        multiply_gemm,
        testing::ThrowsMessage<std::invalid_argument>("Winograd GEMM does not support transposed input views"));
    gemm.lhs.m_transpose = BlasTranspose::None;
    gemm.lhs.m_leading_dimension = 15;
    EXPECT_THAT(
        multiply_gemm,
        testing::ThrowsMessage<std::invalid_argument>("Winograd GEMM input leading dimensions are too small"));
    gemm.lhs.m_leading_dimension = 16;
    gemm.rhs.m_leading_dimension = 11;
    EXPECT_THAT(
        multiply_gemm,
        testing::ThrowsMessage<std::invalid_argument>("Winograd GEMM input leading dimensions are too small"));
    gemm.rhs.m_leading_dimension = 12;
    gemm.output.m_leading_dimension = 11;
    EXPECT_THAT(
        multiply_gemm,
        testing::ThrowsMessage<std::invalid_argument>("Winograd GEMM output leading dimension is too small"));
    gemm.output.m_leading_dimension = 12;
    gemm.alpha = 2;
    EXPECT_THAT(
        multiply_gemm,
        testing::ThrowsMessage<std::invalid_argument>("Winograd GEMM requires alpha=1 and beta=0"));
    gemm.alpha = 1;
    gemm.beta = 1;
    EXPECT_THAT(
        multiply_gemm,
        testing::ThrowsMessage<std::invalid_argument>("Winograd GEMM requires alpha=1 and beta=0"));
}

TEST(WinogradKernel, handles_cancellation)
{
    check_winograd_cancellation<float>();
    check_winograd_cancellation<double>();
}

TEST(WinogradKernel, preserves_output_padding)
{
    check_winograd_padded_output<float>(8, 12, 16);
    check_winograd_padded_output<double>(16, 8, 12);
}

TEST(WinogradKernel, blas_products)
{
#if (defined(__APPLE__) && defined(__arm64__)) || defined(SC_HAS_CBLAS)
    check_blas_winograd<float>(8, 12, 16);
    check_blas_winograd<double>(16, 8, 12);
    check_blas_winograd<Complex<float>>(8, 16, 12);
    check_blas_winograd<Complex<double>>(16, 12, 8);
#else
    std::vector<double> lhs(128);
    std::vector<double> rhs(192);
    std::vector<double> output(96);
    BlasGemmOperation<double> const gemm = make_gemm(8, 12, 16, lhs.data(), rhs.data(), output.data());
    auto const run_winograd = [&]
    { gemm_winograd(8, 12, 16, gemm.lhs, gemm.rhs, gemm.output); };
    EXPECT_THAT(
        run_winograd,
        testing::ThrowsMessage<std::runtime_error>("solvcon BLAS wrapper: CBLAS backend is unavailable"));
#endif
}

TEST(WinogradDispatch, selects_square_threshold)
{
    auto const & tuning = WINOGRAD_TUNING;

    EXPECT_FALSE(meets_winograd_threshold(tuning, 16382, 16382, 16382));
    EXPECT_TRUE(meets_winograd_threshold(tuning, 16384, 16384, 16384));
    EXPECT_FALSE(meets_winograd_threshold(tuning, 16385, 16385, 16385));
    EXPECT_TRUE(meets_winograd_threshold(tuning, 16386, 16386, 16386));
    EXPECT_FALSE(meets_winograd_threshold(tuning, 16386, 16384, 16384));
    EXPECT_FALSE(meets_winograd_threshold(tuning, 16384, 16386, 16384));
    EXPECT_FALSE(meets_winograd_threshold(tuning, 16384, 16384, 16386));
}

} /* end namespace winograd */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
