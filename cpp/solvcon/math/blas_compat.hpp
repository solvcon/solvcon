#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>
#include <solvcon/math/Complex.hpp>

#include <cstdint>
#include <stdexcept>

namespace solvcon
{

/**
 * Whether this build has a BLAS backend behind the wrappers below.
 *
 * A constant rather than an `#if` at each call site, so a caller can discard
 * its BLAS branch through `if constexpr` instead of having the preprocessor
 * delete it. That leaves no unreachable statement behind the branch, which
 * MSVC reports as C4702 from its code generator.
 */
#if (defined(__APPLE__) && defined(__arm64__)) || defined(SC_HAS_CBLAS)
inline constexpr bool has_blas_backend = true;
#else
inline constexpr bool has_blas_backend = false;
#endif

/**
 * @brief Select whether BLAS reads a matrix descriptor as stored or transposed.
 */
enum class BlasTranspose : std::uint8_t
{
    None,
    Transpose,
}; /* end enum class BlasTranspose */

/**
 * @brief Describe a non-owning BLAS vector with a positive element increment.
 */
template <typename T>
struct BlasVectorView
{
    T const * m_data;
    ssize_t m_increment;
}; /* end struct BlasVectorView */

/**
 * @brief Describe a non-owning row-major BLAS matrix operand.
 *
 * A column-major logical matrix is represented by its row-major transposed
 * storage and `BlasTranspose::Transpose`.
 */
template <typename T>
struct BlasMatrixView
{
    T const * m_data;
    ssize_t m_leading_dimension;
    BlasTranspose m_transpose;
}; /* end struct BlasMatrixView */

/**
 * @brief Describe a non-owning writable row-major BLAS matrix view.
 *
 * The leading dimension is measured in elements.
 *
 * @tparam T Element type.
 */
template <typename T>
struct BlasOutputView
{
    T * m_data;
    ssize_t m_leading_dimension;
}; /* end struct BlasOutputView */

/**
 * @brief Describe one non-owning row-major GEMM operation.
 *
 * The operation evaluates `output = alpha * lhs * rhs + beta * output`.
 * The logical shapes of `lhs`, `rhs`, and `output` are `(rows, inner_size)`,
 * `(inner_size, columns)`, and `(rows, columns)`. All views must remain valid
 * until evaluation completes.
 *
 * @tparam T Element type.
 */
template <typename T>
struct BlasGemmOperation
{
    ssize_t rows;
    ssize_t columns;
    ssize_t inner_size;
    BlasMatrixView<T> lhs;
    BlasMatrixView<T> rhs;
    BlasOutputView<T> output;
    T alpha;
    T beta;
}; /* end struct BlasGemmOperation */

namespace detail
{

float dot(ssize_t size, BlasVectorView<float> lhs, BlasVectorView<float> rhs);
double dot(ssize_t size, BlasVectorView<double> lhs, BlasVectorView<double> rhs);
Complex<float> dot(ssize_t size, BlasVectorView<Complex<float>> lhs, BlasVectorView<Complex<float>> rhs);
Complex<double> dot(ssize_t size, BlasVectorView<Complex<double>> lhs, BlasVectorView<Complex<double>> rhs);

void gemv(ssize_t m, ssize_t n, BlasMatrixView<float> matrix, BlasVectorView<float> vector, float * result, BlasTranspose requested_transpose);
void gemv(ssize_t m, ssize_t n, BlasMatrixView<double> matrix, BlasVectorView<double> vector, double * result, BlasTranspose requested_transpose);
void gemv(ssize_t m, ssize_t n, BlasMatrixView<Complex<float>> matrix, BlasVectorView<Complex<float>> vector, Complex<float> * result, BlasTranspose requested_transpose);
void gemv(ssize_t m, ssize_t n, BlasMatrixView<Complex<double>> matrix, BlasVectorView<Complex<double>> vector, Complex<double> * result, BlasTranspose requested_transpose);

void gemm(BlasGemmOperation<float> const & operation);
void gemm(BlasGemmOperation<double> const & operation);
void gemm(BlasGemmOperation<Complex<float>> const & operation);
void gemm(BlasGemmOperation<Complex<double>> const & operation);

} /* end namespace detail */

template <typename T>
T dot_blas(ssize_t size, BlasVectorView<T> lhs, BlasVectorView<T> rhs)
{
#if (defined(__APPLE__) && defined(__arm64__)) || defined(SC_HAS_CBLAS)
    return detail::dot(size, lhs, rhs);
#else
    throw std::runtime_error("solvcon BLAS wrapper: CBLAS backend is unavailable");
#endif
}

template <typename T>
void gemv_blas(ssize_t m, ssize_t n, BlasMatrixView<T> matrix, BlasVectorView<T> vector, T * result, BlasTranspose requested_transpose)
{
#if (defined(__APPLE__) && defined(__arm64__)) || defined(SC_HAS_CBLAS)
    detail::gemv(m, n, matrix, vector, result, requested_transpose);
#else
    throw std::runtime_error("solvcon BLAS wrapper: CBLAS backend is unavailable");
#endif
}

template <typename T>
void gemm_blas(BlasGemmOperation<T> const & operation)
{
#if (defined(__APPLE__) && defined(__arm64__)) || defined(SC_HAS_CBLAS)
    detail::gemm(operation);
#else
    throw std::runtime_error("solvcon BLAS wrapper: CBLAS backend is unavailable");
#endif
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
