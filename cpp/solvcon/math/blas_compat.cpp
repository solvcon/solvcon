/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/blas_compat.hpp>

#if defined(__APPLE__) && defined(__arm64__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#ifndef ACCELERATE_LAPACK_ILP64
#define ACCELERATE_LAPACK_ILP64
#endif
#include <vecLib/cblas_new.h>

#elifdef SC_HAS_MKL
#include <mkl_cblas.h>
#elifdef SC_HAS_CBLAS
#include <cblas.h>
#endif

#include <complex>
#include <format>
#include <limits>
#include <stdexcept>

namespace solvcon
{

#if (defined(__APPLE__) && defined(__arm64__)) || defined(SC_HAS_CBLAS)
#if defined(__APPLE__) && defined(__arm64__)
using blas_int_type = __LAPACK_int;
#else
using blas_int_type = int;
#endif

[[noreturn]] static void throw_blas_int_range_error(ssize_t value, char const * name)
{
    if (value < 0)
    {
        throw std::out_of_range(
            std::format("solvcon BLAS wrapper: {}={} must be non-negative",
                        name,
                        value));
    }

    throw std::out_of_range(
        std::format("solvcon BLAS wrapper: {}={} exceeds BLAS integer range",
                    name,
                    value));
}

static blas_int_type to_blas_int(ssize_t value, char const * name)
{
    if (value < 0 || value > static_cast<ssize_t>(std::numeric_limits<blas_int_type>::max()))
    {
        throw_blas_int_range_error(value, name);
    }
    return static_cast<blas_int_type>(value);
}

static CBLAS_TRANSPOSE to_cblas_transpose(BlasTranspose transpose)
{
    return transpose == BlasTranspose::Transpose ? CblasTrans : CblasNoTrans;
}

static BlasTranspose compose_transposes(BlasTranspose storage, BlasTranspose requested)
{
    return storage == requested ? BlasTranspose::None : BlasTranspose::Transpose;
}

namespace detail
{

float dot(ssize_t size, BlasVectorView<float> lhs, BlasVectorView<float> rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    blas_int_type const bincx = to_blas_int(lhs.m_increment, "incx");
    blas_int_type const bincy = to_blas_int(rhs.m_increment, "incy");
    return cblas_sdot(bsize, lhs.m_data, bincx, rhs.m_data, bincy);
}

double dot(ssize_t size, BlasVectorView<double> lhs, BlasVectorView<double> rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    blas_int_type const bincx = to_blas_int(lhs.m_increment, "incx");
    blas_int_type const bincy = to_blas_int(rhs.m_increment, "incy");
    return cblas_ddot(bsize, lhs.m_data, bincx, rhs.m_data, bincy);
}

Complex<float> dot(ssize_t size, BlasVectorView<Complex<float>> lhs, BlasVectorView<Complex<float>> rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    blas_int_type const bincx = to_blas_int(lhs.m_increment, "incx");
    blas_int_type const bincy = to_blas_int(rhs.m_increment, "incy");
    std::complex<float> value;
    cblas_cdotu_sub(
        bsize, as_std_complex_pointer(lhs.m_data), bincx, as_std_complex_pointer(rhs.m_data), bincy, &value);
    return value;
}

Complex<double> dot(ssize_t size, BlasVectorView<Complex<double>> lhs, BlasVectorView<Complex<double>> rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    blas_int_type const bincx = to_blas_int(lhs.m_increment, "incx");
    blas_int_type const bincy = to_blas_int(rhs.m_increment, "incy");
    std::complex<double> value;
    cblas_zdotu_sub(
        bsize, as_std_complex_pointer(lhs.m_data), bincx, as_std_complex_pointer(rhs.m_data), bincy, &value);
    return value;
}

void gemv(ssize_t m, ssize_t n, BlasMatrixView<float> matrix, BlasVectorView<float> vector, float * result, BlasTranspose requested_transpose)
{
    bool const transposed_storage = matrix.m_transpose == BlasTranspose::Transpose;
    blas_int_type const bm = to_blas_int(transposed_storage ? n : m, "m");
    blas_int_type const bn = to_blas_int(transposed_storage ? m : n, "n");
    blas_int_type const blda = to_blas_int(matrix.m_leading_dimension, "lda");
    blas_int_type const bincx = to_blas_int(vector.m_increment, "incx");
    CBLAS_TRANSPOSE const bop = to_cblas_transpose(compose_transposes(matrix.m_transpose, requested_transpose));
    cblas_sgemv(CblasRowMajor, bop, bm, bn, 1.0F, matrix.m_data, blda, vector.m_data, bincx, 0.0F, result, 1);
}

void gemv(ssize_t m, ssize_t n, BlasMatrixView<double> matrix, BlasVectorView<double> vector, double * result, BlasTranspose requested_transpose)
{
    bool const transposed_storage = matrix.m_transpose == BlasTranspose::Transpose;
    blas_int_type const bm = to_blas_int(transposed_storage ? n : m, "m");
    blas_int_type const bn = to_blas_int(transposed_storage ? m : n, "n");
    blas_int_type const blda = to_blas_int(matrix.m_leading_dimension, "lda");
    blas_int_type const bincx = to_blas_int(vector.m_increment, "incx");
    CBLAS_TRANSPOSE const bop = to_cblas_transpose(compose_transposes(matrix.m_transpose, requested_transpose));
    cblas_dgemv(CblasRowMajor, bop, bm, bn, 1.0, matrix.m_data, blda, vector.m_data, bincx, 0.0, result, 1);
}

void gemv(ssize_t m, ssize_t n, BlasMatrixView<Complex<float>> matrix, BlasVectorView<Complex<float>> vector, Complex<float> * result, BlasTranspose requested_transpose)
{
    bool const transposed_storage = matrix.m_transpose == BlasTranspose::Transpose;
    blas_int_type const bm = to_blas_int(transposed_storage ? n : m, "m");
    blas_int_type const bn = to_blas_int(transposed_storage ? m : n, "n");
    blas_int_type const blda = to_blas_int(matrix.m_leading_dimension, "lda");
    blas_int_type const bincx = to_blas_int(vector.m_increment, "incx");
    CBLAS_TRANSPOSE const bop = to_cblas_transpose(compose_transposes(matrix.m_transpose, requested_transpose));
    std::complex<float> const alpha{1.0F, 0.0F};
    std::complex<float> const beta{0.0F, 0.0F};
    auto const * matrix_data = as_std_complex_pointer(matrix.m_data);
    auto const * vector_data = as_std_complex_pointer(vector.m_data);
    auto * result_data = as_std_complex_pointer(result);
    cblas_cgemv(CblasRowMajor, bop, bm, bn, &alpha, matrix_data, blda, vector_data, bincx, &beta, result_data, 1);
}

void gemv(ssize_t m, ssize_t n, BlasMatrixView<Complex<double>> matrix, BlasVectorView<Complex<double>> vector, Complex<double> * result, BlasTranspose requested_transpose)
{
    bool const transposed_storage = matrix.m_transpose == BlasTranspose::Transpose;
    blas_int_type const bm = to_blas_int(transposed_storage ? n : m, "m");
    blas_int_type const bn = to_blas_int(transposed_storage ? m : n, "n");
    blas_int_type const blda = to_blas_int(matrix.m_leading_dimension, "lda");
    blas_int_type const bincx = to_blas_int(vector.m_increment, "incx");
    CBLAS_TRANSPOSE const bop = to_cblas_transpose(compose_transposes(matrix.m_transpose, requested_transpose));
    std::complex<double> const alpha{1.0, 0.0};
    std::complex<double> const beta{0.0, 0.0};
    auto const * matrix_data = as_std_complex_pointer(matrix.m_data);
    auto const * vector_data = as_std_complex_pointer(vector.m_data);
    auto * result_data = as_std_complex_pointer(result);
    cblas_zgemv(CblasRowMajor, bop, bm, bn, &alpha, matrix_data, blda, vector_data, bincx, &beta, result_data, 1);
}

void gemm(BlasGemmOperation<float> const & operation)
{
    blas_int_type const bm = to_blas_int(operation.rows, "m");
    blas_int_type const bn = to_blas_int(operation.columns, "n");
    blas_int_type const bk = to_blas_int(operation.inner_size, "k");
    blas_int_type const blda = to_blas_int(operation.lhs.m_leading_dimension, "lda");
    blas_int_type const bldb = to_blas_int(operation.rhs.m_leading_dimension, "ldb");
    blas_int_type const bldc = to_blas_int(operation.output.m_leading_dimension, "ldc");
    CBLAS_TRANSPOSE const opa = to_cblas_transpose(operation.lhs.m_transpose);
    CBLAS_TRANSPOSE const opb = to_cblas_transpose(operation.rhs.m_transpose);
    cblas_sgemm(
        CblasRowMajor,
        opa,
        opb,
        bm,
        bn,
        bk,
        operation.alpha,
        operation.lhs.m_data,
        blda,
        operation.rhs.m_data,
        bldb,
        operation.beta,
        operation.output.m_data,
        bldc);
}

void gemm(BlasGemmOperation<double> const & operation)
{
    blas_int_type const bm = to_blas_int(operation.rows, "m");
    blas_int_type const bn = to_blas_int(operation.columns, "n");
    blas_int_type const bk = to_blas_int(operation.inner_size, "k");
    blas_int_type const blda = to_blas_int(operation.lhs.m_leading_dimension, "lda");
    blas_int_type const bldb = to_blas_int(operation.rhs.m_leading_dimension, "ldb");
    blas_int_type const bldc = to_blas_int(operation.output.m_leading_dimension, "ldc");
    CBLAS_TRANSPOSE const opa = to_cblas_transpose(operation.lhs.m_transpose);
    CBLAS_TRANSPOSE const opb = to_cblas_transpose(operation.rhs.m_transpose);
    cblas_dgemm(
        CblasRowMajor,
        opa,
        opb,
        bm,
        bn,
        bk,
        operation.alpha,
        operation.lhs.m_data,
        blda,
        operation.rhs.m_data,
        bldb,
        operation.beta,
        operation.output.m_data,
        bldc);
}

void gemm(BlasGemmOperation<Complex<float>> const & operation)
{
    blas_int_type const bm = to_blas_int(operation.rows, "m");
    blas_int_type const bn = to_blas_int(operation.columns, "n");
    blas_int_type const bk = to_blas_int(operation.inner_size, "k");
    blas_int_type const blda = to_blas_int(operation.lhs.m_leading_dimension, "lda");
    blas_int_type const bldb = to_blas_int(operation.rhs.m_leading_dimension, "ldb");
    blas_int_type const bldc = to_blas_int(operation.output.m_leading_dimension, "ldc");
    CBLAS_TRANSPOSE const opa = to_cblas_transpose(operation.lhs.m_transpose);
    CBLAS_TRANSPOSE const opb = to_cblas_transpose(operation.rhs.m_transpose);
    std::complex<float> const alpha = operation.alpha.to_std_complex();
    std::complex<float> const beta = operation.beta.to_std_complex();
    auto const * lhs_data = as_std_complex_pointer(operation.lhs.m_data);
    auto const * rhs_data = as_std_complex_pointer(operation.rhs.m_data);
    auto * output_data = as_std_complex_pointer(operation.output.m_data);
    cblas_cgemm(
        CblasRowMajor,
        opa,
        opb,
        bm,
        bn,
        bk,
        &alpha,
        lhs_data,
        blda,
        rhs_data,
        bldb,
        &beta,
        output_data,
        bldc);
}

void gemm(BlasGemmOperation<Complex<double>> const & operation)
{
    blas_int_type const bm = to_blas_int(operation.rows, "m");
    blas_int_type const bn = to_blas_int(operation.columns, "n");
    blas_int_type const bk = to_blas_int(operation.inner_size, "k");
    blas_int_type const blda = to_blas_int(operation.lhs.m_leading_dimension, "lda");
    blas_int_type const bldb = to_blas_int(operation.rhs.m_leading_dimension, "ldb");
    blas_int_type const bldc = to_blas_int(operation.output.m_leading_dimension, "ldc");
    CBLAS_TRANSPOSE const opa = to_cblas_transpose(operation.lhs.m_transpose);
    CBLAS_TRANSPOSE const opb = to_cblas_transpose(operation.rhs.m_transpose);
    std::complex<double> const alpha = operation.alpha.to_std_complex();
    std::complex<double> const beta = operation.beta.to_std_complex();
    auto const * lhs_data = as_std_complex_pointer(operation.lhs.m_data);
    auto const * rhs_data = as_std_complex_pointer(operation.rhs.m_data);
    auto * output_data = as_std_complex_pointer(operation.output.m_data);
    cblas_zgemm(
        CblasRowMajor,
        opa,
        opb,
        bm,
        bn,
        bk,
        &alpha,
        lhs_data,
        blda,
        rhs_data,
        bldb,
        &beta,
        output_data,
        bldc);
}

} /* end namespace detail */
#endif

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
