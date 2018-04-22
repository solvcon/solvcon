#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <array>
#include <string>

#include "march/core/types.hpp"
#include "march/core/utility.hpp"
#include "march/core/string.hpp"

namespace march {

/**
 * Cartesian vector in two or three dimensional space.
 */
template< size_t NDIM > struct Vector {

    static_assert(2 == NDIM || 3 == NDIM, "not 2 or 3 dimensional");

    typedef real_type          value_type;
    typedef value_type &       reference;
    typedef const value_type & const_reference;
    typedef size_t             size_type;

    value_type data[NDIM];

    /* constructors */
    Vector() = default;
    Vector(real_type v) { for (size_t it=0; it<NDIM; ++it) { data[it] = v; } }
    Vector(real_type v0, real_type v1) : data{v0, v1} {
        static_assert(2 == NDIM, "only valid for 2 dimensional");
    }
    Vector(real_type v0, real_type v1, real_type v2) : data{v0, v1, v2} {
        static_assert(3 == NDIM, "only valid for 3 dimensional");
    }
    Vector(const Vector & other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] = other.data[it]; }
    }
    Vector(const real_type (&other)[NDIM]) {
        for (size_t it=0; it<NDIM; ++it) { data[it] = other[it]; }
    }
    Vector(const real_type * other) { // I don't like this danger.
        for (size_t it=0; it<NDIM; ++it) { data[it] = other[it]; }
    }

    /* assignment operators */
    Vector & operator=(Vector const & other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] = other.data[it]; }
        return *this;
    }
    Vector & operator=(real_type const (&other)[NDIM]) {
        for (size_t it=0; it<NDIM; ++it) { data[it] = other[it]; }
        return *this;
    }
    Vector & operator=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] = other; }
        return *this;
    }

    bool operator==(Vector const & other) {
        for (size_t it=0; it<NDIM; ++it) { if (data[it] != other.data[it]) { return false; } }
        return true;
    }

    bool operator!=(Vector const & other) {
        for (size_t it=0; it<NDIM; ++it) { if (data[it] != other.data[it]) { return true; } }
        return false;
    }

    bool is_close_to(Vector const & other, real_type epsilon) {
        for (size_t it=0; it<NDIM; ++it) { if (std::abs(data[it] - other.data[it]) > epsilon) { return false; } }
        return true;
    }

    /* accessors */
    constexpr size_type size() const { return NDIM; }
    reference                 operator[](size_type n)       { return data[n]; }
    constexpr const_reference operator[](size_type n) const { return data[n]; }
    reference       at(size_type n)       {
        if (n < NDIM) { return data[n]; }
        else          { throw std::out_of_range(string::format("Vector%ldD doesn't have %d-th element", NDIM, n)); }
    }
    const_reference at(size_type n) const {
        if (n < NDIM) { return data[n]; }
        else          { throw std::out_of_range(string::format("Vector%ldD doesn't have %d-th element", NDIM, n)); }
    }

    /* arithmetic operators */
    Vector & operator+=(Vector const & other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] += other.data[it]; }
        return *this;
    }
    Vector & operator+=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] += other; }
        return *this;
    }
    Vector & operator-=(Vector const & other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] -= other.data[it]; }
        return *this;
    }
    Vector & operator-=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] -= other; }
        return *this;
    }
    Vector & operator*=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] *= other; }
        return *this;
    }
    Vector & operator/=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] /= other; }
        return *this;
    }
    real_type dot(Vector const & rhs) const {
        if (3 == NDIM) { return this->data[0]*rhs.data[0] + this->data[1]*rhs.data[1] + this->data[2]*rhs.data[2]; }
        else           { return this->data[0]*rhs.data[0] + this->data[1]*rhs.data[1]; }
    }
    real_type square() const {
        if (3 == NDIM) { return data[0]*data[0] + data[1]*data[1] + data[2]*data[2]; }
        else           { return data[0]*data[0] + data[1]*data[1]; }
    }
    real_type length() const { return sqrt(square()); }

    std::string repr(size_t indent=0, size_t precision=0) const;

}; /* end struct Vector */

template< size_t NDIM > std::string Vector<NDIM>::repr(size_t, size_t precision) const {
    std::string ret(NDIM == 3 ? "Vector3D(" : "Vector2D(");
    for (size_t it=0; it<NDIM; ++it) {
        ret += string::from_double(data[it], precision);
        ret += NDIM-1 == it ? ")" : ",";
    }
    return ret;
}

template< size_t NDIM >
Vector<NDIM> operator+(Vector<NDIM> lhs, Vector<NDIM> const & rhs) { lhs += rhs; return lhs; }
template< size_t NDIM >
Vector<NDIM> operator+(Vector<NDIM> lhs, real_type            rhs) { lhs += rhs; return lhs; }
template< size_t NDIM >
Vector<NDIM> operator+(real_type    lhs, Vector<NDIM>         rhs) { rhs += lhs; return rhs; }

template< size_t NDIM >
Vector<NDIM> operator-(Vector<NDIM> lhs, Vector<NDIM> const & rhs) { lhs -= rhs; return lhs; }
template< size_t NDIM >
Vector<NDIM> operator-(Vector<NDIM> lhs, real_type            rhs) { lhs -= rhs; return lhs; }

template< size_t NDIM >
Vector<NDIM> operator*(Vector<NDIM> lhs, real_type            rhs) { lhs *= rhs; return lhs; }
template< size_t NDIM >
Vector<NDIM> operator*(real_type    lhs, Vector<NDIM>         rhs) { rhs *= lhs; return rhs; }

template< size_t NDIM >
Vector<NDIM> operator/(Vector<NDIM> lhs, real_type            rhs) { lhs /= rhs; return lhs; }

inline real_type cross(Vector<2> const & lhs, Vector<2> const & rhs) {
    return lhs[0]*rhs[1] - lhs[1]*rhs[0];
}

inline Vector<3> cross(Vector<3> const & lhs, Vector<3> const & rhs) {
    return Vector<3>(lhs[1]*rhs[2] - lhs[2]*rhs[1], lhs[2]*rhs[0] - lhs[0]*rhs[2], lhs[0]*rhs[1] - lhs[1]*rhs[0]);
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
