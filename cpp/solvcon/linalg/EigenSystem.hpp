#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Eigenvalue and eigenvector computation for general (non-symmetric) matrices
 * using LAPACK *GEEV: SGEEV (float), DGEEV (double), CGEEV (Complex<float>),
 * and ZGEEV (Complex<double>).
 *
 * The LAPACK backend is selected by the build system via MM_HAS_VENDOR_LAPACK:
 * Apple's vecLib (Accelerate framework) on macOS and OpenBLAS on Linux.
 *
 * @ingroup group_numerics
 */

#ifndef MM_HAS_VENDOR_LAPACK
#error "solvcon/linalg/EigenSystem.hpp requires a vendor LAPACK (MM_HAS_VENDOR_LAPACK)."
#endif

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include <solvcon/buffer/buffer.hpp>
#include <solvcon/math/lapack_compat.hpp>

namespace solvcon
{

/**
 * Eigenvalue solver for a general matrix using LAPACK *GEEV.
 *
 * The element type T is one of float, double, Complex<float>, or
 * Complex<double>.  Construction validates the input shape and prepares
 * column-major workspace buffers.  Call run() to invoke the matching *GEEV
 * routine to calculate eigenvalues and eigenvectors.
 *
 * @ingroup group_numerics
 */
template <typename T>
class EigenSystem
    : public std::enable_shared_from_this<EigenSystem<T>>
{

    static_assert(
        is_real_v<T> || is_complex_v<T>,
        "EigenSystem<T> requires T to be a real or complex number type");

private:

    struct ctor_passkey
    {
    };

public:

    using value_type = T;
    using array_type = SimpleArray<T>;
    using real_type = typename detail::select_real_t<T>::type;
    using real_array_type = SimpleArray<real_type>;
    static constexpr bool is_complex = is_complex_v<T>;

    static std::shared_ptr<EigenSystem> construct(array_type const & matrix, bool do_vl = true, bool do_vr = true)
    {
        return std::make_shared<EigenSystem>(matrix, do_vl, do_vr, ctor_passkey());
    }

    explicit EigenSystem(array_type const & matrix, bool do_vl, bool do_vr, ctor_passkey const &);

    EigenSystem() = delete;
    EigenSystem(EigenSystem const &) = delete;
    EigenSystem(EigenSystem &&) = delete;
    EigenSystem & operator=(EigenSystem const &) = delete;
    EigenSystem & operator=(EigenSystem &&) = delete;
    ~EigenSystem() = default;

    void run();

    array_type const & matrix() const { return m_matrix; }
    real_array_type const & wr() const { return m_wr; }
    real_array_type const & wi() const { return m_wi; }
    array_type const & vl(bool suppress_exception = false) const;
    array_type const & vr(bool suppress_exception = false) const;
    bool do_vl() const { return m_do_vl; }
    bool do_vr() const { return m_do_vr; }
    bool done() const { return m_done; }

private:

    static std::string format_shape(array_type const & arr);

    array_type const & m_matrix;
    array_type m_colmajor;
    real_array_type m_wr;
    real_array_type m_wi;
    array_type m_vl;
    array_type m_vr;
    bool const m_do_vl;
    bool const m_do_vr;
    bool m_done = false;

}; /* end class EigenSystem */

template <typename T>
EigenSystem<T>::EigenSystem(array_type const & matrix, bool do_vl, bool do_vr, ctor_passkey const &)
    : m_matrix(matrix)
    // Stage matrix into a column-major workspace for LAPACK.
    , m_colmajor(matrix.to_column_major())
    , m_wr(matrix.shape(0))
    , m_wi(matrix.shape(0))
    , m_vl(do_vl ? array_type(matrix.shape()) : array_type())
    , m_vr(do_vr ? array_type(matrix.shape()) : array_type())
    , m_do_vl(do_vl)
    , m_do_vr(do_vr)
{
    if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1))
    {
        throw std::invalid_argument(std::format(
            "EigenSystem: matrix must be a square 2D SimpleArray, but got shape {}",
            format_shape(matrix)));
    }

    // Result buffers stay uninitialized: *GEEV never reads them on entry and
    // run() overwrites them, so their contents are unspecified until then.
    if (m_do_vl)
    {
        m_vl.transpose();
    }
    if (m_do_vr)
    {
        m_vr.transpose();
    }
}

/**
 * @brief Run the matching *GEEV routine on the prepared workspace.
 */
template <typename T>
void EigenSystem<T>::run()
{
    auto const n = static_cast<lapack_int_t>(m_matrix.shape(0));
    if (n == 0)
    {
        m_done = true;
        return;
    }

    /*
     * LAPACK *GEEV API reference (real DGEEV shown; the complex CGEEV/ZGEEV
     * return a single complex eigenvalue array w and take a real rwork):
     *   https://www.netlib.org/lapack/explore-html/d4/d68/group__geev_ga7d8afe93d23c5862e238626905ee145e.html
     *   https://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f_source.html
     */
    char const jobvl = m_do_vl ? 'V' : 'N';
    char const jobvr = m_do_vr ? 'V' : 'N';
    lapack_int_t const lda = n;
    // *GEEV requires LDVL/LDVR >= 1 and a valid (non-null) pointer even when
    // the matrix is unreferenced; route the unused side to a stack scratch
    // slot.
    lapack_int_t const ldvl = m_do_vl ? n : 1;
    lapack_int_t const ldvr = m_do_vr ? n : 1;
    lapack_int_t info = 0;
    value_type vl_dummy{};
    value_type vr_dummy{};
    value_type * const vl_ptr = m_do_vl ? m_vl.data() : &vl_dummy;
    value_type * const vr_ptr = m_do_vr ? m_vr.data() : &vr_dummy;

    // Phase 1: workspace query.  lwork == -1 tells *GEEV to write the optimal
    // workspace size into work[0] without performing any work.  Phase 2 then
    // allocates max(optimal, documented minimum) and runs the factorization.
    value_type work_query{};
    lapack_int_t lwork = -1;

    if constexpr (is_complex)
    {
        // CGEEV/ZGEEV return eigenvalues as a single complex array and need a
        // real workspace rwork of length 2*n.  Compute into a local complex
        // buffer, then split it into the real m_wr/m_wi parts below.
        array_type w(static_cast<size_t>(n));
        real_array_type rwork(static_cast<size_t>(2 * n));
        detail::lapack_geev(
            jobvl, jobvr, n, m_colmajor.data(), lda, w.data(), vl_ptr, ldvl, vr_ptr, ldvr, &work_query, lwork, rwork.data(), &info);
        if (info != 0)
        {
            throw std::runtime_error(std::format(
                "EigenSystem::run: GEEV workspace query failed with info={}",
                static_cast<int64_t>(info)));
        }

        lapack_int_t const lwork_min = 2 * n;
        lwork = std::max<lapack_int_t>(static_cast<lapack_int_t>(work_query.real()), lwork_min);
        array_type work(static_cast<size_t>(lwork));
        detail::lapack_geev(
            jobvl, jobvr, n, m_colmajor.data(), lda, w.data(), vl_ptr, ldvl, vr_ptr, ldvr, work.data(), lwork, rwork.data(), &info);
        for (lapack_int_t i = 0; i < n; ++i)
        {
            m_wr(i) = w(i).real();
            m_wi(i) = w(i).imag();
        }
    }
    else
    {
        detail::lapack_geev(
            jobvl, jobvr, n, m_colmajor.data(), lda, m_wr.data(), m_wi.data(), vl_ptr, ldvl, vr_ptr, ldvr, &work_query, lwork, &info);
        if (info != 0)
        {
            throw std::runtime_error(std::format(
                "EigenSystem::run: GEEV workspace query failed with info={}",
                static_cast<int64_t>(info)));
        }

        // 4*n minimum when any eigenvectors requested, else 3*n.
        lapack_int_t const lwork_min = (m_do_vl || m_do_vr) ? 4 * n : 3 * n;
        lwork = std::max<lapack_int_t>(static_cast<lapack_int_t>(work_query), lwork_min);
        array_type work(static_cast<size_t>(lwork));
        detail::lapack_geev(
            jobvl, jobvr, n, m_colmajor.data(), lda, m_wr.data(), m_wi.data(), vl_ptr, ldvl, vr_ptr, ldvr, work.data(), lwork, &info);
    }

    if (info < 0)
    {
        throw std::runtime_error(std::format(
            "EigenSystem::run: GEEV illegal argument at position {}",
            -static_cast<int>(info)));
    }
    if (info > 0)
    {
        throw std::runtime_error(std::format(
            "EigenSystem::run: GEEV failed to converge; {} eigenvalues did not converge",
            static_cast<int>(info)));
    }
    m_done = true;
}

/**
 * @brief Left eigenvectors computed by *GEEV.
 *
 * @param suppress_exception
 *      Return the empty placeholder instead of throwing when the matrix was
 *      not computed.
 * @return
 *      Column-major n-by-n matrix; empty when do_vl=false and
 *      suppress_exception=true.
 * @throws std::runtime_error When do_vl=false and suppress_exception=false.
 */
template <typename T>
EigenSystem<T>::array_type const & EigenSystem<T>::vl(bool suppress_exception) const
{
    if (!m_do_vl && !suppress_exception)
    {
        throw std::runtime_error(
            "EigenSystem::vl: left eigenvectors were not computed "
            "(do_vl=false)");
    }
    return m_vl;
}

/**
 * @brief Right eigenvectors computed by *GEEV.
 *
 * @param suppress_exception
 *      Return the empty placeholder instead of throwing when the matrix was
 *      not computed.
 * @return
 *      Column-major n-by-n matrix; empty when do_vr=false and
 *      suppress_exception=true.
 * @throws std::runtime_error When do_vr=false and suppress_exception=false.
 */
template <typename T>
EigenSystem<T>::array_type const & EigenSystem<T>::vr(bool suppress_exception) const
{
    if (!m_do_vr && !suppress_exception)
    {
        throw std::runtime_error(
            "EigenSystem::vr: right eigenvectors were not computed "
            "(do_vr=false)");
    }
    return m_vr;
}

template <typename T>
std::string EigenSystem<T>::format_shape(array_type const & arr)
{
    std::string result = "(";
    for (size_t i = 0; i < arr.ndim(); ++i)
    {
        if (i > 0)
        {
            result += ", ";
        }
        result += std::to_string(arr.shape(i));
    }
    result += ")";
    return result;
}

/**
 * Type-erased surrogate for EigenSystem<T>.
 *
 * Accepts a SimpleArrayPlex and dispatches on its runtime element type to the
 * matching EigenSystem<T> (float, double, Complex<float>, Complex<double>).
 * Results are returned as type-erased SimpleArrayPlex.
 *
 * @ingroup group_numerics
 */
class EigenSystemPlex
{

public:

    explicit EigenSystemPlex(SimpleArrayPlex const & matrix, bool do_vl = true, bool do_vr = true);

    EigenSystemPlex() = delete;
    EigenSystemPlex(EigenSystemPlex const &) = delete;
    EigenSystemPlex(EigenSystemPlex &&) = delete;
    EigenSystemPlex & operator=(EigenSystemPlex const &) = delete;
    EigenSystemPlex & operator=(EigenSystemPlex &&) = delete;
    ~EigenSystemPlex() = default;

    void run();

    SimpleArrayPlex const & matrix() const { return m_matrix; }
    DataType data_type() const { return m_data_type; }

    SimpleArrayPlex wr() const;
    SimpleArrayPlex wi() const;
    SimpleArrayPlex vl(bool suppress_exception = false) const;
    SimpleArrayPlex vr(bool suppress_exception = false) const;
    bool do_vl() const;
    bool do_vr() const;
    bool done() const;

private:

    using solver_variant = std::variant<
        std::shared_ptr<EigenSystem<float>>,
        std::shared_ptr<EigenSystem<double>>,
        std::shared_ptr<EigenSystem<Complex<float>>>,
        std::shared_ptr<EigenSystem<Complex<double>>>>;

    SimpleArrayPlex const & m_matrix;
    solver_variant m_solver;
    DataType m_data_type;

}; /* end class EigenSystemPlex */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
