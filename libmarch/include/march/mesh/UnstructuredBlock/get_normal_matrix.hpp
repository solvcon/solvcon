#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include "march/core.hpp"
#include "march/mesh/UnstructuredBlock/class.hpp"

namespace march {

namespace detail {

template< size_t NDIM >
Matrix<NDIM> outward_rotation(Vector<NDIM> const & nml, Vector<NDIM> const & cnd, Vector<NDIM> const & crd);

template<>
inline
Matrix<2> outward_rotation<2>(Vector<2> const & nml, Vector<2> const &, Vector<2> const &) {
    Matrix<2> ret;
    ret[0] = nml;
    ret[1][0] =  nml[1];
    ret[1][1] = -nml[0];
    return ret;
}

template<>
inline
Matrix<3> outward_rotation<3>(Vector<3> const & nml, Vector<3> const & cnd, Vector<3> const & crd) {
    Matrix<3> ret;
    ret[0] = nml;
    ret[1] = crd - cnd;
    ret[1] /= ret[1].length();
    ret[2] = cross(ret[0], ret[1]);
    return ret;
}

} /* end namespace detail */

template< size_t NDIM >
Matrix<NDIM> UnstructuredBlock<NDIM>::get_normal_matrix(index_type const ifc) const {
    Matrix<NDIM> ret;
    auto const & tfcnds = fcnds()[ifc];
    auto const & tfcnml = reinterpret_cast<Vector<NDIM> const &>(fcnml()[ifc]);
    auto const & tfccnd = reinterpret_cast<Vector<NDIM> const &>(fccnd()[ifc]);
    auto const & tndcrd = reinterpret_cast<Vector<NDIM> const &>(ndcrd()[tfcnds[1]]);
    // set perpendicular gradient to zero.
    ret = detail::outward_rotation(tfcnml, tfccnd, tndcrd);
    return ret;
}

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
