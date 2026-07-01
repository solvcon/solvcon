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

#elifdef MM_HAS_CBLAS
#include <cblas.h>
#endif

#include <complex>
#include <format>
#include <limits>
#include <stdexcept>

namespace solvcon
{

#if (defined(__APPLE__) && defined(__arm64__)) || defined(MM_HAS_CBLAS)
#if defined(__APPLE__) && defined(__arm64__)
using blas_int_type = __LAPACK_int;
#else
using blas_int_type = int;
#endif

static blas_int_type to_blas_int(ssize_t value, char const * name)
{
    if (value < 0)
    {
        throw std::out_of_range(
            std::format("solvcon BLAS wrapper: {}={} must be non-negative",
                        name,
                        value));
    }
    if (value <= static_cast<ssize_t>(std::numeric_limits<blas_int_type>::max()))
    {
        return static_cast<blas_int_type>(value);
    }

    throw std::out_of_range(
        std::format("solvcon BLAS wrapper: {}={} exceeds BLAS integer range",
                    name,
                    value));
}

static CBLAS_TRANSPOSE to_cblas_transpose(bool transpose_matrix)
{
    return transpose_matrix ? CblasTrans : CblasNoTrans;
}

float dot_blas(ssize_t size, float const * lhs, float const * rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    return cblas_sdot(bsize, lhs, 1, rhs, 1);
}

double dot_blas(ssize_t size, double const * lhs, double const * rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    return cblas_ddot(bsize, lhs, 1, rhs, 1);
}

Complex<float> dot_blas(ssize_t size,
                        Complex<float> const * lhs,
                        Complex<float> const * rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    std::complex<float> result;
    cblas_cdotu_sub(bsize,
                    as_std_complex_pointer(lhs),
                    1,
                    as_std_complex_pointer(rhs),
                    1,
                    &result);
    return result;
}

Complex<double> dot_blas(ssize_t size,
                         Complex<double> const * lhs,
                         Complex<double> const * rhs)
{
    blas_int_type const bsize = to_blas_int(size, "size");
    std::complex<double> result;
    cblas_zdotu_sub(bsize,
                    as_std_complex_pointer(lhs),
                    1,
                    as_std_complex_pointer(rhs),
                    1,
                    &result);
    return result;
}

void gemv_blas(ssize_t m,
               ssize_t n,
               float const * matrix,
               float const * vector,
               float * result,
               bool transpose_matrix)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    cblas_sgemv(CblasRowMajor,
                to_cblas_transpose(transpose_matrix),
                bm,
                bn,
                1.0F,
                matrix,
                bn,
                vector,
                1,
                0.0F,
                result,
                1);
}

void gemv_blas(ssize_t m,
               ssize_t n,
               double const * matrix,
               double const * vector,
               double * result,
               bool transpose_matrix)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    cblas_dgemv(CblasRowMajor,
                to_cblas_transpose(transpose_matrix),
                bm,
                bn,
                1.0,
                matrix,
                bn,
                vector,
                1,
                0.0,
                result,
                1);
}

void gemv_blas(ssize_t m,
               ssize_t n,
               Complex<float> const * matrix,
               Complex<float> const * vector,
               Complex<float> * result,
               bool transpose_matrix)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    std::complex<float> const alpha{1.0F, 0.0F};
    std::complex<float> const beta{0.0F, 0.0F};
    cblas_cgemv(CblasRowMajor,
                to_cblas_transpose(transpose_matrix),
                bm,
                bn,
                &alpha,
                as_std_complex_pointer(matrix),
                bn,
                as_std_complex_pointer(vector),
                1,
                &beta,
                as_std_complex_pointer(result),
                1);
}

void gemv_blas(ssize_t m,
               ssize_t n,
               Complex<double> const * matrix,
               Complex<double> const * vector,
               Complex<double> * result,
               bool transpose_matrix)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    std::complex<double> const alpha{1.0, 0.0};
    std::complex<double> const beta{0.0, 0.0};
    cblas_zgemv(CblasRowMajor,
                to_cblas_transpose(transpose_matrix),
                bm,
                bn,
                &alpha,
                as_std_complex_pointer(matrix),
                bn,
                as_std_complex_pointer(vector),
                1,
                &beta,
                as_std_complex_pointer(result),
                1);
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               float const * lhs,
               float const * rhs,
               float * result)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                bm,
                bn,
                bk,
                1.0F,
                lhs,
                bk,
                rhs,
                bn,
                0.0F,
                result,
                bn);
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               double const * lhs,
               double const * rhs,
               double * result)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                bm,
                bn,
                bk,
                1.0,
                lhs,
                bk,
                rhs,
                bn,
                0.0,
                result,
                bn);
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               Complex<float> const * lhs,
               Complex<float> const * rhs,
               Complex<float> * result)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    std::complex<float> const alpha{1.0F, 0.0F};
    std::complex<float> const beta{0.0F, 0.0F};
    cblas_cgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                bm,
                bn,
                bk,
                &alpha,
                as_std_complex_pointer(lhs),
                bk,
                as_std_complex_pointer(rhs),
                bn,
                &beta,
                as_std_complex_pointer(result),
                bn);
}

void gemm_blas(ssize_t m,
               ssize_t n,
               ssize_t k,
               Complex<double> const * lhs,
               Complex<double> const * rhs,
               Complex<double> * result)
{
    blas_int_type const bm = to_blas_int(m, "m");
    blas_int_type const bn = to_blas_int(n, "n");
    blas_int_type const bk = to_blas_int(k, "k");
    std::complex<double> const alpha{1.0, 0.0};
    std::complex<double> const beta{0.0, 0.0};
    cblas_zgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                bm,
                bn,
                bk,
                &alpha,
                as_std_complex_pointer(lhs),
                bk,
                as_std_complex_pointer(rhs),
                bn,
                &beta,
                as_std_complex_pointer(result),
                bn);
}
#else
[[noreturn]] static void throw_blas_unavailable()
{
    throw std::runtime_error(
        "solvcon BLAS wrapper: CBLAS backend is unavailable");
}

float dot_blas(ssize_t, float const *, float const *)
{
    throw_blas_unavailable();
}

double dot_blas(ssize_t, double const *, double const *)
{
    throw_blas_unavailable();
}

Complex<float> dot_blas(ssize_t,
                        Complex<float> const *,
                        Complex<float> const *)
{
    throw_blas_unavailable();
}

Complex<double> dot_blas(ssize_t,
                         Complex<double> const *,
                         Complex<double> const *)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               float const *,
               float const *,
               float *,
               bool)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               double const *,
               double const *,
               double *,
               bool)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               Complex<float> const *,
               Complex<float> const *,
               Complex<float> *,
               bool)
{
    throw_blas_unavailable();
}

void gemv_blas(ssize_t,
               ssize_t,
               Complex<double> const *,
               Complex<double> const *,
               Complex<double> *,
               bool)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t, ssize_t, ssize_t, float const *, float const *, float *)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               double const *,
               double const *,
               double *)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               Complex<float> const *,
               Complex<float> const *,
               Complex<float> *)
{
    throw_blas_unavailable();
}

void gemm_blas(ssize_t,
               ssize_t,
               ssize_t,
               Complex<double> const *,
               Complex<double> const *,
               Complex<double> *)
{
    throw_blas_unavailable();
}
#endif

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
