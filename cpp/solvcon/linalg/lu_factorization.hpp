#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * LU factorization with partial pivoting for general matrices, with
 * linear-solve, inverse, and determinant helpers.
 *
 * @ingroup group_numerics
 */

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

/**
 * Stateful LU decomposition with partial pivoting for general (non-symmetric)
 * matrices.
 *
 * Given an n-by-n matrix A, the constructor computes the factorization
 * PA = LU and stores the result, where:
 *   - P is a permutation matrix (represented as a pivot index vector),
 *   - L is a unit lower triangular matrix (diagonal elements are implicitly 1),
 *   - U is an upper triangular matrix.
 *
 * L and U are stored compactly in a single n-by-n array: the strictly lower
 * triangle holds L (excluding the unit diagonal), and the upper triangle
 * (including the diagonal) holds U.
 *
 * The pivot vector piv[k] records that row k was swapped with row piv[k]
 * during the k-th elimination step.  Swaps are applied sequentially from
 * k = 0 to k = n-1.
 *
 * The expensively computed lu matrix and pivot vector are cached as members
 * so that multiple solve()/inv() calls reuse the O(n^3) factorization instead
 * of redoing it.
 *
 * Supported element types: float, double, Complex<float>, Complex<double>.
 *
 * @ingroup group_numerics
 */
template <typename T>
class LuFactorization
{

    static_assert(
        is_real_v<T> || is_complex_v<T>,
        "LuFactorization<T> requires T to be a real or complex number type");

public:

    using value_type = T;
    using real_type = typename detail::select_real_t<value_type>::type;
    using array_type = SimpleArray<value_type>;
    using pivot_type = SimpleArray<int64_t>;

    /**
     * Factorize a square matrix into PA = LU.
     *
     * @param a  Square 2D SimpleArray (the matrix to factorize).
     * @throws std::invalid_argument if a is not a square 2D array.
     * @throws std::runtime_error    if the matrix is singular or near-singular.
     */
    explicit LuFactorization(array_type const & a);

    LuFactorization() = delete;
    LuFactorization(LuFactorization const &) = default;
    LuFactorization(LuFactorization &&) = default;
    LuFactorization & operator=(LuFactorization const &) = default;
    LuFactorization & operator=(LuFactorization &&) = default;
    ~LuFactorization() = default;

    /**
     * The combined LU matrix: L in the strictly lower triangle and U in the
     * upper triangle (including the diagonal).
     */
    array_type const & lu() const { return m_lu; }

    /**
     * The pivot vector: piv()[k] is the row index that was swapped with
     * row k at step k of the factorization.
     */
    pivot_type const & piv() const { return m_piv; }

    /** Dimension of the factorized square matrix. */
    ssize_t n() const { return static_cast<ssize_t>(m_lu.shape(0)); }

    /**
     * Solve the linear system Ax = b using the cached LU factors.
     *
     * @param b  1D or 2D SimpleArray (the right-hand side).
     *           If 1D, it is treated as a single column vector.
     *           If 2D with shape (n, m), each column is a separate RHS.
     * @return   The solution x with the same shape as b.
     * @throws std::invalid_argument if b has wrong rank or dimensions are
     *         incompatible with the factorized matrix.
     */
    array_type solve(array_type const & b) const;

    /**
     * Compute the matrix inverse A^(-1) using the cached LU factors.
     *
     * Internally solves AX = I where I is the identity matrix.
     */
    array_type inv() const;

    /**
     * Compute the determinant det(A) using the cached LU factors.
     *
     * From PA = LU we have det(A) = det(P)^(-1) * det(L) * det(U).  L is unit
     * lower triangular so det(L) = 1; det(U) is the product of its diagonal;
     * and det(P) = (-1)^s, where s is the number of row swaps recorded in the
     * pivot vector.  The determinant is therefore (-1)^s times the product of
     * the diagonal of U, obtained in O(n) from the already-factorized data.
     *
     * Note: a singular or near-singular matrix is rejected by the constructor,
     * so a successfully constructed factorization never has a (near-)zero
     * determinant.
     */
    value_type det() const;

private:

    // Validate that a is square and 2D; return its shape for member init.
    static small_vector<ssize_t> validate_shape(array_type const & a);

    /**
     * Forward substitution: solve Ly = Pb using the cached factors.
     *
     * Applies the row permutation recorded in m_piv to b, then solves the
     * unit lower triangular system.  L is stored in the strictly lower
     * triangle of m_lu (diagonal of L is implicitly 1).
     */
    array_type forward_substitution(array_type const & b) const;

    /**
     * Backward substitution: solve Ux = y using the cached factors.
     *
     * U is stored in the upper triangle (including diagonal) of m_lu.
     */
    array_type backward_substitution(array_type const & y) const;

    array_type m_lu;
    pivot_type m_piv;

}; /* end class LuFactorization */

template <typename T>
small_vector<ssize_t> LuFactorization<T>::validate_shape(array_type const & a)
{
    if (a.ndim() != 2 || a.shape(0) != a.shape(1))
    {
        throw std::invalid_argument(std::format(
            "LuFactorization: a must be a square 2D SimpleArray, but got shape {}",
            detail::format_shape(a.shape())));
    }
    return a.shape();
}

template <typename T>
LuFactorization<T>::LuFactorization(array_type const & a)
    : m_lu(validate_shape(a), value_type{0})
    , m_piv(small_vector<ssize_t>{a.shape(0)})
{
    const auto n = a.shape(0);

    // Working copy of the input matrix.  The algorithm modifies m_lu
    // in-place, overwriting it with the combined L and U factors.
    std::copy(a.begin(), a.end(), m_lu.begin());

    // piv[k] records which row was swapped with row k at step k.
    // Initialize with std::iota so piv[k] = k (identity permutation).
    std::iota(m_piv.begin(), m_piv.end(), int64_t{0});

    // Main elimination loop.  For each column k, we:
    //   1. Search column k (rows k..n-1) for the largest absolute value
    //      (partial pivoting) to improve numerical stability.
    //   2. Swap the pivot row with row k.
    //   3. Check for singularity.
    //   4. Compute multipliers and update the trailing submatrix.
    for (ssize_t k = 0; k < n; ++k)
    {
        // Step 1: Find the pivot -- the row in [k, n) with the largest
        // absolute value in column k.  Note: abs() returns real_type even
        // for complex value_type, so all comparison variables must be real.
        real_type max_val = abs(m_lu(k, k));
        ssize_t max_row = k;
        for (ssize_t i = k + 1; i < n; ++i)
        {
            real_type const val = abs(m_lu(i, k));
            if (val > max_val)
            {
                max_val = val;
                max_row = i;
            }
        }
        m_piv[k] = static_cast<int64_t>(max_row);

        // Step 2: Swap rows k and max_row in the full working matrix.
        if (max_row != k)
        {
            for (ssize_t j = 0; j < n; ++j)
            {
                std::swap(m_lu(k, j), m_lu(max_row, j));
            }
        }

        // Step 3: Reject both exact singularity (pivot = 0, e.g. duplicate
        // rows) and near-singularity (pivot at noise level).
        real_type const pivot = abs(m_lu(k, k));
        const auto eps = std::numeric_limits<real_type>::epsilon();

        // Absolute threshold (~100 * machine eps); works for well-scaled inputs.
        // TODO: make it relative to matrix/column magnitude for better robustness.
        real_type const singular_tol = real_type(100) * eps;
        if (pivot <= singular_tol)
        {
            throw std::runtime_error("LuFactorization: LU decomposition failed: singular or near-singular matrix.");
        }

        // Step 4: Gaussian elimination.  For each row below the pivot,
        // compute the multiplier l(i,k) = a(i,k) / a(k,k) and store it
        // in the lower triangle.  Then subtract l(i,k) * row_k from row_i
        // in the trailing submatrix (columns k+1..n-1).
        for (ssize_t i = k + 1; i < n; ++i)
        {
            m_lu(i, k) = m_lu(i, k) / m_lu(k, k); // multiplier stored in L
            for (ssize_t j = k + 1; j < n; ++j)
            {
                m_lu(i, j) = m_lu(i, j) - m_lu(i, k) * m_lu(k, j);
            }
        }
    }
}

template <typename T>
typename LuFactorization<T>::array_type LuFactorization<T>::forward_substitution(array_type const & b) const
{
    const auto m = n();
    const auto ncols = b.shape(1);
    small_vector<ssize_t> const y_shape{m, ncols};

    // Copy b into y so we can apply the permutation in-place.
    array_type y(y_shape);
    std::copy(b.begin(), b.end(), y.begin());

    // Apply the row permutation P to b.  The swaps are replayed in the
    // same order they were recorded during factorization (k = 0, 1, ...).
    for (ssize_t i = 0; i < m; ++i)
    {
        if (m_piv[i] != i)
        {
            for (ssize_t k = 0; k < ncols; ++k)
            {
                std::swap(y(i, k), y(m_piv[i], k));
            }
        }
    }

    // Solve Ly = Pb by forward substitution.  L is unit lower triangular
    // (diagonal is implicitly 1), so the formula for each element is:
    //   y(i, k) = y(i, k) - sum_{j=0}^{i-1} L(i,j) * y(j, k)
    for (ssize_t k = 0; k < ncols; ++k)
    {
        for (ssize_t i = 1; i < m; ++i)
        {
            value_type sum{0};
            for (ssize_t j = 0; j < i; ++j)
            {
                sum += m_lu(i, j) * y(j, k);
            }
            y(i, k) = y(i, k) - sum;
        }
    }

    return y;
}

template <typename T>
typename LuFactorization<T>::array_type LuFactorization<T>::backward_substitution(array_type const & y) const
{
    const auto m = n();
    const auto ncols = y.shape(1);
    small_vector<ssize_t> const x_shape{m, ncols};
    array_type x(x_shape);

    // Solve Ux = y by backward substitution.  U occupies the upper triangle
    // (including diagonal) of m_lu.  The formula is:
    //   x(i, k) = (y(i, k) - sum_{j=i+1}^{m-1} U(i,j) * x(j, k)) / U(i,i)
    for (ssize_t k = 0; k < ncols; ++k)
    {
        for (ssize_t i = m - 1; i >= 0; --i)
        {
            value_type sum{0};
            for (ssize_t j = i + 1; j < m; ++j)
            {
                sum += m_lu(i, j) * x(j, k);
            }
            x(i, k) = (y(i, k) - sum) / m_lu(i, i);
        }
    }

    return x;
}

template <typename T>
typename LuFactorization<T>::array_type LuFactorization<T>::solve(array_type const & b) const
{
    if (b.ndim() != 1 && b.ndim() != 2)
    {
        throw std::invalid_argument(std::format(
            "LuFactorization::solve: b must be 1D or 2D, but got {}D",
            b.ndim()));
    }
    if (b.shape(0) != n())
    {
        throw std::invalid_argument(std::format(
            "LuFactorization::solve: dimension mismatch: a.shape[0]={}, b.shape[0]={}",
            n(),
            b.shape(0)));
    }

    // If b is 1D, reshape to a column matrix (n, 1), solve, then reshape
    // the result back to 1D so the caller gets the same shape they passed in.
    bool const was_1d = (b.ndim() == 1);
    if (was_1d)
    {
        auto b_2d = b.reshape(small_vector<ssize_t>{b.shape(0), 1});
        auto y = forward_substitution(b_2d);
        auto x = backward_substitution(y);
        return x.reshape(small_vector<ssize_t>{x.shape(0)});
    }
    else
    {
        auto y = forward_substitution(b);
        auto x = backward_substitution(y);
        return x;
    }
}

template <typename T>
typename LuFactorization<T>::array_type LuFactorization<T>::inv() const
{
    // Matrix inversion is equivalent to solving AX = I for X, where I is the
    // n-by-n identity matrix.  Each column of X is the solution to A * x_j = e_j.
    auto identity = array_type::eye(n());
    auto y = forward_substitution(identity);
    auto x = backward_substitution(y);
    return x;
}

template <typename T>
typename LuFactorization<T>::value_type LuFactorization<T>::det() const
{
    // det(U) is the product of its diagonal; L contributes 1 (unit diagonal).
    value_type result{1};
    for (ssize_t i = 0; i < n(); ++i)
    {
        result *= m_lu(i, i);
    }

    // det(P) flips sign once per row swap (a transposition).  m_piv[k] != k
    // means step k swapped row k with a later row.
    bool negate = false;
    for (ssize_t k = 0; k < n(); ++k)
    {
        if (m_piv[k] != k)
        {
            negate = !negate;
        }
    }
    return negate ? value_type{0} - result : result;
}

/**
 * Free function wrapper for LU factorization with partial pivoting.
 *
 * @param a  Square 2D SimpleArray.
 * @return   (lu_matrix, pivot_vector) pair.  See LuFactorization<T>.
 */
template <typename T>
std::pair<SimpleArray<T>, SimpleArray<int64_t>> lu_factorization(SimpleArray<T> const & a)
{
    LuFactorization<T> const f(a);
    return {f.lu(), f.piv()};
}

/**
 * Free function wrapper for solving Ax = b via LU decomposition.
 *
 * @param a  Square 2D SimpleArray (coefficient matrix).
 * @param b  1D or 2D SimpleArray (right-hand side).
 * @return   The solution x with the same shape as b.
 */
template <typename T>
SimpleArray<T> lu_solve(SimpleArray<T> const & a, SimpleArray<T> const & b)
{
    return LuFactorization<T>(a).solve(b);
}

/**
 * Free function wrapper for computing the matrix inverse via LU decomposition.
 *
 * @param a  Square 2D SimpleArray.
 * @return   The inverse A^(-1).
 */
template <typename T>
SimpleArray<T> lu_inv(SimpleArray<T> const & a)
{
    return LuFactorization<T>(a).inv();
}

/**
 * Free function wrapper for computing the determinant via LU decomposition.
 *
 * @param a  Square 2D SimpleArray.
 * @return   The determinant det(A).
 */
template <typename T>
T lu_det(SimpleArray<T> const & a)
{
    return LuFactorization<T>(a).det();
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
