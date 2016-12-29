#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <array>

#include "march/core/types.hpp"
#include "march/core/Vector.hpp"

namespace march {

/**
 * Cartesian vector in two or three dimensional space.
 */
template< size_t NDIM > struct Matrix {

    static_assert(2 == NDIM || 3 == NDIM, "not 2 or 3 dimensional");

    typedef Vector<NDIM>       value_type;
    typedef value_type &       reference;
    typedef const value_type & const_reference;
    typedef size_t             size_type;

    value_type data[NDIM];

    /* constructors */
    Matrix() = default;
    Matrix(real_type v) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] = v;
            }
        }
    }
    Matrix(const value_type & v0, const value_type & v1) : data{v0, v1} {
        static_assert(2 == NDIM, "only valid for 2 dimensional");
    }
    Matrix(const value_type & v0, const value_type & v1, const value_type & v2) : data{v0, v1, v2} {
        static_assert(3 == NDIM, "only valid for 3 dimensional");
    }
    Matrix(const Matrix & other) {
        for (size_t it=0; it<NDIM; ++it) { data[it] = other.data[it]; }
    }
    Matrix(const real_type (&other)[NDIM][NDIM]) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] = other[it][jt];
            }
        }
    }

    /* assignment operators */
    Matrix & operator=(Matrix const & other) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] = other[it][jt];
            }
        }
        return *this;
    }
    Matrix & operator=(real_type const (&other)[NDIM][NDIM]) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] = other[it][jt];
            }
        }
        return *this;
    }
    Matrix & operator=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] = other;
            }
        }
        return *this;
    }

    /* accessors */
    constexpr size_type size() const { return NDIM; }
    constexpr size_type nelem() const { return NDIM*NDIM; }
    reference                 operator[](size_type n)       { return data[n]; }
    constexpr const_reference operator[](size_type n) const { return data[n]; }
    Vector<NDIM> column(size_t n) const {
        Vector<NDIM> ret;
        for (size_t it=0; it<NDIM; ++it) { ret[it] = data[it][n]; }
        return ret;
    }

    /* arithmetic operators */
    Matrix & operator+=(Matrix const & other) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] += other[it][jt];
            }
        }
        return *this;
    }
    Matrix & operator+=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] += other;
            }
        }
        return *this;
    }
    Matrix & operator-=(Matrix const & other) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] -= other[it][jt];
            }
        }
        return *this;
    }
    Matrix & operator-=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] -= other;
            }
        }
        return *this;
    }
    Matrix & operator*=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] *= other;
            }
        }
        return *this;
    }
    Matrix & operator/=(real_type other) {
        for (size_t it=0; it<NDIM; ++it) {
            for (size_t jt=0; jt<NDIM; ++jt) {
                data[it][jt] /= other;
            }
        }
        return *this;
    }

}; /* end struct Matrix */

template< size_t NDIM >
Matrix<NDIM> operator+(Matrix<NDIM> lhs, Matrix<NDIM> const & rhs) { lhs += rhs; return lhs; }
template< size_t NDIM >
Matrix<NDIM> operator+(Matrix<NDIM> lhs, real_type            rhs) { lhs += rhs; return lhs; }

template< size_t NDIM >
Matrix<NDIM> operator-(Matrix<NDIM> lhs, Matrix<NDIM> const & rhs) { lhs -= rhs; return lhs; }
template< size_t NDIM >
Matrix<NDIM> operator-(Matrix<NDIM> lhs, real_type            rhs) { lhs -= rhs; return lhs; }

template< size_t NDIM >
Matrix<NDIM> operator*(Matrix<NDIM> lhs, real_type            rhs) { lhs *= rhs; return lhs; }

template< size_t NDIM >
Matrix<NDIM> operator/(Matrix<NDIM> lhs, real_type            rhs) { lhs /= rhs; return lhs; }

template< size_t NDIM >
Vector<NDIM> product(Matrix<NDIM> const & lhs, Vector<NDIM> const & rhs) {
    Vector<NDIM> ret;
    for (size_t it=0; it<NDIM; ++it) { ret[it] = lhs[it].dot(rhs); }
    return ret;
}

inline real_type volume(const Matrix<2> & mat) {
    return fabs(cross(mat[0], mat[1])) / 2;
}

inline real_type volume(const Matrix<3> & mat) {
    return fabs(cross(mat[0], mat[1]).dot(mat[2])) / 6;
}

inline Matrix<2> unnormalized_inverse(const Matrix<2> & mat) {
    Matrix<2> ret;
    ret[0][0] =  mat[1][1]; ret[0][1] = -mat[0][1];
    ret[1][0] = -mat[1][0]; ret[1][1] =  mat[0][0];
    return ret;
}

inline Matrix<3> unnormalized_inverse(const Matrix<3> & mat) {
    Matrix<3> ret;
    ret[0][0] = mat[1][1]*mat[2][2] - mat[1][2]*mat[2][1];
    ret[0][1] = mat[0][2]*mat[2][1] - mat[0][1]*mat[2][2];
    ret[0][2] = mat[0][1]*mat[1][2] - mat[0][2]*mat[1][1];
    ret[1][0] = mat[1][2]*mat[2][0] - mat[1][0]*mat[2][2];
    ret[1][1] = mat[0][0]*mat[2][2] - mat[0][2]*mat[2][0];
    ret[1][2] = mat[0][2]*mat[1][0] - mat[0][0]*mat[1][2];
    ret[2][0] = mat[1][0]*mat[2][1] - mat[1][1]*mat[2][0];
    ret[2][1] = mat[0][1]*mat[2][0] - mat[0][0]*mat[2][1];
    ret[2][2] = mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0];
    return ret;
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
