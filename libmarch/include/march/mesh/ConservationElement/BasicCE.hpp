#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <array>

#include "march/core.hpp"

#include "march/mesh/UnstructuredBlock.hpp"

namespace march {

namespace detail {

/**
 * Calculator of BasicCE.
 */
template< size_t NDIM >
struct BasicCECenterVolumeHelper {
    typedef UnstructuredBlock<NDIM> block_type;
    typedef Vector<NDIM> vector_type;
    /**
     * Read the input data from standard mesh.
     *
     * @param[in] block  The block to load from.
     * @param[in] icl    The index of cell of interest.
     * @param[in] ifl    The index of face in the cell, 0-base.
     */
    BasicCECenterVolumeHelper(const block_type & block, index_type icl, index_type ifl)
      : tfcnds(block.fcnds()[block.clfcs()[icl][ifl+1]])
    {
        assert(ifl < block.clfcs()[icl][0]);
        // cell and face centers.
        index_type ifc = block.clfcs()[icl][ifl+1];
        outward_face = block.fccls()[icl][0] == icl ? true : false;
        crdi = block.clcnd()[icl];
        crde = block.clcnd()[block.fcrcl(ifc, icl)];
        if (3 == NDIM) fcnd = block.fccnd()[ifc];
        // node coordinates.
        index_type inf;
        for (inf=0; inf<tfcnds[0]; ++inf) crd[inf] = block.ndcrd()[tfcnds[inf+1]];
        if (3 == NDIM) crd[inf] = crd[0];
    }
    void calc_cnd_vol(vector_type & cnd, real_type & vol) const;
    void calc_subface_cnd(std::array<vector_type, block_type::FCMND> & fccnd) const;
    void calc_subface_nml(std::array<vector_type, block_type::FCMND> & fcnml) const;
    const index_type (&tfcnds)[block_type::FCMND+1]; // "t" for "this".
    bool outward_face; // true if the face normal of icl point outward.
    vector_type crdi;
    vector_type crde;
    vector_type fcnd; // only valid in 3 dimensional.
    vector_type crd[block_type::FCMND+1]; // only in 3 dimensional we need the last vector.
}; /* end struct BasicCECenterVolumeHelper */

template<>
inline void BasicCECenterVolumeHelper<3>::calc_cnd_vol(BasicCECenterVolumeHelper<3>::vector_type & cnd, real_type & vol) const {
    vol = 0.0;
    cnd = 0.0;
    for (index_type inf=0; inf<tfcnds[0]; ++inf) {
        vector_type tvec = crd[inf] + crd[inf+1] + fcnd;
        // base triangle.
        vector_type disu = crd[inf  ] - fcnd;
        vector_type disv = crd[inf+1] - fcnd;
        vector_type dist = cross(disu, disv);
        // inner tetrahedron.
        vector_type disw = crdi - fcnd;
        real_type voli = fabs(dist.dot(disw)) / 6;
        vector_type cndi = (tvec + crdi) / 4;
        // outer tetrahedron.
        disw = crde - fcnd;
        real_type vole = fabs(dist.dot(disw)) / 6;
        vector_type cnde = (tvec + crde) / 4;
        // accumulate volume and centroid for BCE.
        vol += voli + vole;
        cnd += cndi * voli + cnde * vole;
    }
    cnd /= vol;
}

template<>
inline void BasicCECenterVolumeHelper<2>::calc_cnd_vol(BasicCECenterVolumeHelper<2>::vector_type & cnd, real_type & vol) const {
    // triangle formed by cell point and two face nodes.
    vector_type cndi = (crd[0] + crd[1] + crdi) / 3;
    real_type voli = fabs(cross(crd[0]-crdi, crd[1]-crdi)) / 2;
    // triangle formed by neighbor cell point and two face nodes.
    vector_type cnde = (crd[0] + crd[1] + crde) / 3;
    real_type vole = fabs(cross(crd[0]-crde, crd[1]-crde)) / 2;
    // volume of BCE (quadrilateral) formed by the two triangles.
    vol = voli + vole;
    // geometry center of each BCE for cell jcl.
    cnd = (cndi * voli + cnde * vole) / vol;
}

template< size_t NDIM >
inline void BasicCECenterVolumeHelper<NDIM>::calc_subface_cnd(
    std::array<BasicCECenterVolumeHelper<NDIM>::vector_type, BasicCECenterVolumeHelper<NDIM>::block_type::FCMND> & sfcnd
) const {
    for (index_type inf=0; inf<tfcnds[0]; ++inf) {
        sfcnd[inf] = crde;
        for (index_type it=inf; it<inf+(NDIM-1); ++it) sfcnd[inf] += crd[it];
        sfcnd[inf] /= NDIM;
    }
}

template<>
inline void BasicCECenterVolumeHelper<3>::calc_subface_nml(
    std::array<BasicCECenterVolumeHelper<3>::vector_type, BasicCECenterVolumeHelper<3>::block_type::FCMND> & sfnml
) const {
    real_type voe = outward_face ? 0.5 : -0.5;
    for (index_type inf=0; inf<tfcnds[0]; ++inf) {
        vector_type disu = crd[inf  ] - crde;
        vector_type disv = crd[inf+1] - crde;
        sfnml[inf] = cross(disu, disv) * voe;
    }
}

template<>
inline void BasicCECenterVolumeHelper<2>::calc_subface_nml(
    std::array<BasicCECenterVolumeHelper<2>::vector_type, BasicCECenterVolumeHelper<2>::block_type::FCMND> & sfnml
) const {
    real_type voe = (crd[0][0]-crde[0])*(crd[1][1]-crde[1])
                  - (crd[0][1]-crde[1])*(crd[1][0]-crde[0]);
    voe /= fabs(voe);
    sfnml[0][0] = -(crde[1]-crd[0][1]) * voe;
    sfnml[0][1] =  (crde[0]-crd[0][0]) * voe;
    sfnml[1][0] =  (crde[1]-crd[1][1]) * voe;
    sfnml[1][1] = -(crde[0]-crd[1][0]) * voe;
}

} /* end namespace detail */

/**
 * Geometry data for a single basic conservation element (BCE).
 */
template< size_t NDIM >
struct BasicCE {
    typedef UnstructuredBlock<NDIM> block_type;
    typedef Vector<NDIM> vector_type;

    vector_type cnd; // conservation element centroid.
    real_type vol; // conservation element volume.
    std::array<vector_type, block_type::FCMND> sfcnd; // sub-face centroids.
    std::array<vector_type, block_type::FCMND> sfnml; // sub-face normal.

    BasicCE() = default;
    BasicCE(const block_type & block, index_type icl, index_type ifl) {
        detail::BasicCECenterVolumeHelper<NDIM> helper(block, icl, ifl);
        helper.calc_cnd_vol(cnd, vol);
        helper.calc_subface_cnd(sfcnd);
        helper.calc_subface_nml(sfnml);
    }
}; /* end struct BasicCE */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
