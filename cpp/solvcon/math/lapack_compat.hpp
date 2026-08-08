#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * Compatibility shim for the vendor LAPACK backend.
 *
 * On Apple, pull in Accelerate/vecLib with the modern (non-deprecated) LAPACK
 * signatures.  On other platforms with OpenBLAS, declare the Fortran ABI
 * symbols directly; this avoids pulling in lapacke.h, which distributions
 * package inconsistently.
 *
 * Consumers should include this header and use `lapack_int_t` plus the
 * Fortran-named entry points (e.g. `dgeev_`) without conditional compilation.
 * The header itself requires `SC_HAS_VENDOR_LAPACK` to be defined by the build
 * system.
 */

#ifndef SC_HAS_VENDOR_LAPACK
#error "solvcon/math/lapack_compat.hpp requires a vendor LAPACK (SC_HAS_VENDOR_LAPACK)."
#endif

#include <solvcon/math/Complex.hpp>

#ifdef __APPLE__
// Opt in to the modern (non-deprecated) LAPACK signatures provided by
// Accelerate.  Must be defined before including the Accelerate header.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
// NOLINTNEXTLINE(misc-header-include-cycle)
#include <Accelerate/Accelerate.h>
#else // __APPLE__
// On Linux + OpenBLAS we declare the Fortran ABI symbols directly.  The
// general (non-symmetric) eigensolvers are SGEEV/DGEEV (real) and
// CGEEV/ZGEEV (complex).  The complex buffers are passed as void* so this
// header needs no LAPACK complex type; the real workspace rwork is a plain
// float/double array.
extern "C" void sgeev_(
    char const * jobvl,
    char const * jobvr,
    int const * n,
    float * a,
    int const * lda,
    float * wr,
    float * wi,
    float * vl,
    int const * ldvl,
    float * vr,
    int const * ldvr,
    float * work,
    int const * lwork,
    int * info);
extern "C" void dgeev_(
    char const * jobvl,
    char const * jobvr,
    int const * n,
    double * a,
    int const * lda,
    double * wr,
    double * wi,
    double * vl,
    int const * ldvl,
    double * vr,
    int const * ldvr,
    double * work,
    int const * lwork,
    int * info);
extern "C" void cgeev_(
    char const * jobvl,
    char const * jobvr,
    int const * n,
    void * a,
    int const * lda,
    void * w,
    void * vl,
    int const * ldvl,
    void * vr,
    int const * ldvr,
    void * work,
    int const * lwork,
    float * rwork,
    int * info);
extern "C" void zgeev_(
    char const * jobvl,
    char const * jobvr,
    int const * n,
    void * a,
    int const * lda,
    void * w,
    void * vl,
    int const * ldvl,
    void * vr,
    int const * ldvr,
    void * work,
    int const * lwork,
    double * rwork,
    int * info);
#endif // __APPLE__

namespace solvcon
{

#ifdef __APPLE__
using lapack_int_t = __LAPACK_int;
// Accelerate's new-LAPACK complex element types for CGEEV/ZGEEV.
using lapack_complex_float_t = __LAPACK_float_complex;
using lapack_complex_double_t = __LAPACK_double_complex;
#else
using lapack_int_t = int;
// On Linux the complex symbols above take void*; cast targets are void.
using lapack_complex_float_t = void;
using lapack_complex_double_t = void;
#endif

namespace detail
{

/**
 * Overloaded thin wrappers around the LAPACK *GEEV entry points, presenting a
 * uniform call shape to EigenSystem<T>.  solvcon's Complex<T> is
 * layout-compatible with the LAPACK complex types, so the buffers are
 * reinterpret_cast to the backend pointer type.
 */
inline void lapack_geev(
    char jobvl, char jobvr, lapack_int_t n, float * a, lapack_int_t lda, float * wr, float * wi, float * vl, lapack_int_t ldvl, float * vr, lapack_int_t ldvr, float * work, lapack_int_t lwork, lapack_int_t * info)
{
    sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

inline void lapack_geev(
    char jobvl, char jobvr, lapack_int_t n, double * a, lapack_int_t lda, double * wr, double * wi, double * vl, lapack_int_t ldvl, double * vr, lapack_int_t ldvr, double * work, lapack_int_t lwork, lapack_int_t * info)
{
    dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

inline void lapack_geev(
    char jobvl, char jobvr, lapack_int_t n, Complex<float> * a, lapack_int_t lda, Complex<float> * w, Complex<float> * vl, lapack_int_t ldvl, Complex<float> * vr, lapack_int_t ldvr, Complex<float> * work, lapack_int_t lwork, float * rwork, lapack_int_t * info)
{
    cgeev_(
        &jobvl, &jobvr, &n, reinterpret_cast<lapack_complex_float_t *>(a), &lda, reinterpret_cast<lapack_complex_float_t *>(w), reinterpret_cast<lapack_complex_float_t *>(vl), &ldvl, reinterpret_cast<lapack_complex_float_t *>(vr), &ldvr, reinterpret_cast<lapack_complex_float_t *>(work), &lwork, rwork, info); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

inline void lapack_geev(
    char jobvl, char jobvr, lapack_int_t n, Complex<double> * a, lapack_int_t lda, Complex<double> * w, Complex<double> * vl, lapack_int_t ldvl, Complex<double> * vr, lapack_int_t ldvr, Complex<double> * work, lapack_int_t lwork, double * rwork, lapack_int_t * info)
{
    zgeev_(
        &jobvl, &jobvr, &n, reinterpret_cast<lapack_complex_double_t *>(a), &lda, reinterpret_cast<lapack_complex_double_t *>(w), reinterpret_cast<lapack_complex_double_t *>(vl), &ldvl, reinterpret_cast<lapack_complex_double_t *>(vr), &ldvr, reinterpret_cast<lapack_complex_double_t *>(work), &lwork, rwork, info); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
