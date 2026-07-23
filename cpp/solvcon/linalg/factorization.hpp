#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Cholesky (L L^T) factorization and linear solve for symmetric positive
 * definite matrices.
 *
 * @ingroup group_numerics
 */

#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

namespace detail
{

template <typename T>
class Llt
{

private:

    using value_type = T;
    using real_type = typename detail::select_real_t<value_type>::type;
    using array_type = SimpleArray<value_type>;

public:

    // Factorization algorithm
    // Reference: https://www.geeksforgeeks.org/dsa/cholesky-decomposition-matrix-decomposition/
    static array_type factorize(array_type const & a);
    // Solver: solve A x = b where A = L L^T (or L L^H for complex)
    static array_type solve(array_type const & a, array_type const & b);

private:

    // Forward substitution: solve L Y = B
    static array_type forward_substitution(array_type const & l, array_type const & b);
    // Backward substitution: solve L^T X = Y (or L^H X = Y for complex)
    static array_type backward_substitution(array_type const & l, array_type const & y);

}; /* end class Llt */

template <typename T>
Llt<T>::array_type Llt<T>::forward_substitution(array_type const & l, array_type const & b)
{
    if (l.ndim() != 2 || l.shape(0) != l.shape(1))
    {
        std::ostringstream oss;
        oss << "Llt::forward_substitution: The first argument l must be a square 2D SimpleArray, but got shape "
            << detail::format_shape(l.shape());
        throw std::invalid_argument(oss.str());
    }
    if (b.ndim() != 2 || b.shape(0) != l.shape(0))
    {
        std::ostringstream oss;
        oss << "Llt::forward_substitution: The second argument b must be a 2D SimpleArray with first dimension matching l, but got shape "
            << detail::format_shape(b.shape());
        throw std::invalid_argument(oss.str());
    }

    const auto m = l.shape(0);
    const auto n = b.shape(1);
    small_vector<ssize_t> const y_shape{m, n};
    array_type y(y_shape);
    for (ssize_t k = 0; k < n; ++k)
    {
        for (ssize_t i = 0; i < m; ++i)
        {
            T sum = 0;
            for (ssize_t j = 0; j < i; ++j)
            {
                sum += l(i, j) * y(j, k);
            }
            y(i, k) = (b(i, k) - sum) / l(i, i);
        }
    }
    return y;
}

template <typename T>
Llt<T>::array_type Llt<T>::backward_substitution(array_type const & l, array_type const & y)
{
    if (l.ndim() != 2 || l.shape(0) != l.shape(1))
    {
        std::ostringstream oss;
        oss << "Llt::backward_substitution: The first argument l must be a square 2D SimpleArray, but got shape "
            << detail::format_shape(l.shape());
        throw std::invalid_argument(oss.str());
    }
    if (y.ndim() != 2 || y.shape(0) != l.shape(0))
    {
        std::ostringstream oss;
        oss << "Llt::backward_substitution: The second argument y must be a 2D SimpleArray with first dimension matching l, but got shape "
            << detail::format_shape(y.shape());
        throw std::invalid_argument(oss.str());
    }

    const auto m = l.shape(0);
    const auto n = y.shape(1);
    small_vector<ssize_t> const x_shape{m, n};
    array_type x(x_shape);
    for (ssize_t k = 0; k < n; ++k)
    {
        for (ssize_t i = m - 1; i >= 0; --i)
        {
            T sum = 0;
            for (ssize_t j = i + 1; j < m; ++j)
            {
                sum += conj_mul(x(j, k), l(j, i));
            }
            x(i, k) = (y(i, k) - sum) / l(i, i);
        }
    }
    return x;
}

template <typename T>
Llt<T>::array_type Llt<T>::factorize(array_type const & a)
{
    if (a.ndim() != 2 || a.shape(0) != a.shape(1))
    {
        std::ostringstream oss;
        oss << "Llt::factorize: The first argument a must be a square 2D SimpleArray, but got shape "
            << detail::format_shape(a.shape());
        throw std::invalid_argument(oss.str());
    }

    const auto m = a.shape(0);
    small_vector<ssize_t> const shape = {m, m};
    array_type l(shape, value_type(0));
    const real_type eps = std::numeric_limits<real_type>::epsilon();
    for (ssize_t i = 0; i < m; ++i)
    {
        for (ssize_t j = 0; j <= i; ++j)
        {
            value_type sum = 0;
            for (ssize_t k = 0; k < j; ++k)
            {
                sum += conj_mul(l(i, k), l(j, k));
            }
            if (i == j)
            {
                real_type const dr = real(a(j, j) - sum);
                real_type const tol = std::max<real_type>(1, abs(a(j, j))) * 100 * eps;
                if (dr <= tol)
                {
                    throw std::runtime_error("Llt::factorize: Cholesky failed: SimpleArray not (numerically) SPD.");
                }
                l(j, j) = std::sqrt(dr);
            }
            else
            {
                l(i, j) = (a(i, j) - sum) / l(j, j);
            }
        }
    }
    return l;
}

template <typename T>
Llt<T>::array_type Llt<T>::solve(array_type const & a, array_type const & b)
{
    if (a.ndim() != 2 || a.shape(0) != a.shape(1))
    {
        std::ostringstream oss;
        oss << "Llt::solve: The first argument a must be a square 2D SimpleArray, but got shape "
            << detail::format_shape(a.shape());
        throw std::invalid_argument(oss.str());
    }
    if (a.shape(0) != b.shape(0))
    {
        std::ostringstream oss;
        oss << "Llt::solve: The first argument a and the second argument b dimension mismatch: a.shape[0]=" << a.shape(0) << ", b.shape[0]=" << b.shape(0);
        throw std::invalid_argument(oss.str());
    }
    if (b.ndim() != 1 && b.ndim() != 2)
    {
        std::ostringstream oss;
        oss << "Llt::solve: The second argument b must be 1D or 2D, but got " << b.ndim() << "D";
        throw std::invalid_argument(oss.str());
    }

    array_type const l = factorize(a);

    bool const was_1d = (b.ndim() == 1);
    if (was_1d)
    {
        array_type const b_2d = b.reshape(small_vector<ssize_t>{b.shape(0), 1});
        array_type const y = forward_substitution(l, b_2d);
        array_type const x = backward_substitution(l, y);
        return x.reshape(small_vector<ssize_t>{x.shape(0)});
    }
    else
    {
        array_type const y = forward_substitution(l, b);
        array_type x = backward_substitution(l, y);
        return x;
    }
}

} /* end namespace detail */

template <typename T>
SimpleArray<T> llt_factorization(SimpleArray<T> const & a)
{
    return detail::Llt<T>::factorize(a);
}

template <typename T>
SimpleArray<T> llt_solve(SimpleArray<T> const & a, SimpleArray<T> const & b)
{
    return detail::Llt<T>::solve(a, b);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
