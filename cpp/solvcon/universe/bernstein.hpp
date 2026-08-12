#pragma once

/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/math/math.hpp>
#include <vector>

namespace solvcon
{

namespace detail
{

template <typename T>
T calc_bernstein_polynomial_impl(T t, size_t i, size_t n)
{
    T ret = 1.0;
    if (i > 0)
    {
        ret *= static_cast<T>(std::pow(t, i));
    }
    if (n > i)
    {
        ret *= static_cast<T>(std::pow(1.0 - t, n - i));
    }
    for (size_t it = n; it > 1; --it)
    {
        ret *= it;
    }
    for (size_t it = i; it > 1; --it)
    {
        ret /= it;
    }
    for (size_t it = (n - i); it > 1; --it)
    {
        ret /= it;
    }
    return ret;
}

template <typename T>
T interpolate_bernstein_impl(T t, std::vector<T> const & values, size_t n)
{
    T ret = 0.0;
    for (size_t it = 0; it <= n; ++it)
    {
        T v = (it >= values.size()) ? T(1) : values[it];
        v *= calc_bernstein_polynomial_impl(t, it, n);
        ret += v;
    }
    return ret;
}

} /* end namespace detail */

double calc_bernstein_polynomial(double t, size_t i, size_t n);
double interpolate_bernstein(double t, std::vector<double> const & values, size_t n);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
