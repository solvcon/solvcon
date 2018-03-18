#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 *
 * This file includes code for gradient element.
 */

#include <array>

#include "march/core.hpp"

#include "march/mesh/UnstructuredBlock.hpp"
#include "march/mesh/ConservationElement/BasicCE.hpp"

namespace march {

struct GEType {

    /// Maximum number of FGE.
    static constexpr size_t NFGE_MAX = 8;
    /// Maximum number of faces of a FGE (number of dimension).
    static constexpr size_t FGENFC_MAX = 3;

    GEType(index_type celltype_id, index_type nfge, real_type nfge_inverse)
      : celltype_id(celltype_id)
      , clnfc(::march::celltype(celltype_id).nface())
      , nfge(nfge)
      , nfge_inverse(nfge_inverse)
    {
        faces.fill({-1, -1, -1}); // sentinel.
    }

    GEType() {
        faces.fill({-1, -1, -1}); // sentinel.
    }

    const CellType & celltype() const { return ::march::celltype(celltype_id); }
    index_type fgenfc() const { return celltype().ndim(); }

public: // data members

    using fge_facelist_type = std::array<index_type, FGENFC_MAX>;

    index_type celltype_id = -1; // sentinel
    index_type clnfc = -1; // sentinel
    index_type nfge = -1; // sentinel
    real_type nfge_inverse = 0.0; // sentinel
    /// The face list in the mesh block's clfcs table.
    std::array< fge_facelist_type, NFGE_MAX > faces;

}; /* end struct GEType */

namespace detail {

class GETypeGroup {

public:

    GEType const & operator[](size_t id) const { return m_ge_types[id]; }
    size_t size() const { return sizeof(m_ge_types) / sizeof(GEType); }
 
    static const GETypeGroup & get_instance() {
        static GETypeGroup inst;
        return inst;
    }

private:

    GETypeGroup()
      : m_ge_types{
            GEType(0, 0, 0    ) // point
          , GEType(1, 1, 1    ) // line
          , GEType(2, 4, 1.0/4) // quadrilateral
          , GEType(3, 3, 1.0/3) // triangle
          , GEType(4, 8, 1.0/8) // hexahedron
          , GEType(5, 4, 1.0/4) // tetrahedron
          , GEType(6, 6, 1.0/6) // prism
          , GEType(7, 6, 1.0/6) // pyramid
        }
    {
        m_ge_types[CellType::QUADRILATERAL].faces[0] = {1, 2, -1};
        m_ge_types[CellType::QUADRILATERAL].faces[1] = {2, 3, -1};
        m_ge_types[CellType::QUADRILATERAL].faces[2] = {3, 4, -1};
        m_ge_types[CellType::QUADRILATERAL].faces[3] = {4, 1, -1};
        m_ge_types[CellType::TRIANGLE].faces[0] = {1, 2, -1};
        m_ge_types[CellType::TRIANGLE].faces[1] = {2, 3, -1};
        m_ge_types[CellType::TRIANGLE].faces[2] = {3, 1, -1};
        m_ge_types[CellType::HEXAHEDRON].faces[0] = {2, 3, 5};
        m_ge_types[CellType::HEXAHEDRON].faces[1] = {6, 3, 2};
        m_ge_types[CellType::HEXAHEDRON].faces[2] = {4, 3, 6};
        m_ge_types[CellType::HEXAHEDRON].faces[3] = {5, 3, 4};
        m_ge_types[CellType::HEXAHEDRON].faces[4] = {5, 1, 2};
        m_ge_types[CellType::HEXAHEDRON].faces[5] = {2, 1, 6};
        m_ge_types[CellType::HEXAHEDRON].faces[6] = {6, 1, 4};
        m_ge_types[CellType::HEXAHEDRON].faces[7] = {4, 1, 5};
        m_ge_types[CellType::TETRAHEDRON].faces[0] = {3, 1, 2};
        m_ge_types[CellType::TETRAHEDRON].faces[1] = {2, 1, 4};
        m_ge_types[CellType::TETRAHEDRON].faces[2] = {4, 1, 3};
        m_ge_types[CellType::TETRAHEDRON].faces[3] = {2, 4, 3};
        m_ge_types[CellType::PRISM].faces[0] = {5, 2, 4};
        m_ge_types[CellType::PRISM].faces[1] = {3, 2, 5};
        m_ge_types[CellType::PRISM].faces[2] = {4, 2, 3};
        m_ge_types[CellType::PRISM].faces[3] = {4, 1, 5};
        m_ge_types[CellType::PRISM].faces[4] = {5, 1, 3};
        m_ge_types[CellType::PRISM].faces[5] = {3, 1, 4};
        m_ge_types[CellType::PYRAMID].faces[0] = {1, 5, 2};
        m_ge_types[CellType::PYRAMID].faces[1] = {2, 5, 3};
        m_ge_types[CellType::PYRAMID].faces[2] = {3, 5, 4};
        m_ge_types[CellType::PYRAMID].faces[3] = {4, 5, 1};
        m_ge_types[CellType::PYRAMID].faces[4] = {1, 3, 4};
        m_ge_types[CellType::PYRAMID].faces[5] = {3, 1, 2};
    }

    GEType m_ge_types[CellType::NTYPE];

}; /* end class GETypeGroup */

} /* end namespace detail */

/**
 * Get the GEType object for the specified type id.
 */
inline GEType const & getype(size_t id) { return detail::GETypeGroup::get_instance()[id]; }

/**
 * The geometry shape of a general gradient element that is used to calculate
 * the gradient for a CESE dual mesh.
 */
template< size_t NDIM >
struct GradientElement {

    typedef UnstructuredBlock<NDIM> block_type;

public: // data members

    const index_type icl;
    const GEType & getype;
    const block_type & block;

    /**
     * Indices of neighboring cells.
     */
    index_type rcls[CellType::CLNFC_MAX];
    /**
     * Displacement from the self solution point to the gradient evaluation
     * points.
     */
    Vector<NDIM> idis[CellType::CLNFC_MAX];
    /**
     * Displacement from the neighboring solution points to the gradient
     * evaluation points.
     */
    Vector<NDIM> jdis[CellType::CLNFC_MAX];

public:

    /**
     * Calculate and fill the following vectors using the tau parameter:
     *
     *  1. idis: Gradient evaluation point displacement from the self solution point.
     *  2. jdis: Gradient evaluation point displacement from the neighboring solution points.
     *
     * The GGE needs to be displaced so that the centroid colocates with the
     * solution point (i.e., centroid of the CCE).
     *
     * @param[in] block  The unstructured mesh definition.
     * @param[in] icl    The index of self cell.
     * @param[in] tau    Tau parameter (as in the c-tau scheme) of this cell.
     */
    GradientElement(
        const block_type & block
      , const LookupTable<real_type, NDIM> & cecnd
      , const index_type icl
      , const real_type tau
    )
      : icl(icl)
      , getype(::march::getype(block.cltpn()[icl]))
      , block(block)
    {
#ifdef MH_DEBUG
        fill_sentinel(&rcls[0], CellType::CLNFC_MAX, std::numeric_limits<index_type>::min());
        fill_sentinel(&idis[0][0], CellType::CLNFC_MAX * NDIM);
        fill_sentinel(&jdis[0][0], CellType::CLNFC_MAX * NDIM);
#endif // MH_DEBUG

        const auto & tclfcs = block.clfcs()[icl];
        const auto & icecnd = reinterpret_cast<const Vector<NDIM> &>(cecnd[icl]);

        // Calculate gradient evaluation points by using tau.
        assert(tau >= 0);
        assert(tau <= 1.0);
        for (index_type ifl=0; ifl<getype.clnfc; ++ifl) {
            const auto jcl = block.fcrcl(tclfcs[ifl+1], icl);
            rcls[ifl] = jcl;
            const auto & jcecnd = reinterpret_cast<const Vector<NDIM> &>(cecnd[jcl]);
            const auto midpt = BasicCE<NDIM>(block, icl, ifl).cnd;
            idis[ifl] = (jcecnd - midpt) * tau + midpt;
            jdis[ifl] = idis[ifl] - jcecnd;
        }

        // Calculate average point.
        Vector<NDIM> crd(0.0);
        for (index_type ifl=0; ifl<getype.clnfc; ++ifl) { crd += idis[ifl]; }
        crd /= getype.clnfc;
        // Triangulate the GGE into sub-GE using the average point and then
        // calculate the centroid.
        Vector<NDIM> cnd(0.0);
        real_type voc = 0;
        for (index_type isub=0; isub<getype.nfge; ++isub) {
            Vector<NDIM> subcnd(crd);
            Matrix<NDIM> dst;
            for (index_type ivx=0; ivx<NDIM; ++ivx) {
                const index_type ifl = getype.faces[isub][ivx]-1;
                assert(ifl >= 0);
                subcnd += idis[ifl];
                dst[ivx] = idis[ifl] - crd;
            }
            subcnd /= NDIM+1;
            const auto vob = volume(dst);
            voc += vob;
            cnd += subcnd * vob;
        }
        cnd /= voc;

        // Shift the GGE vertices.
        const auto gsft = icecnd - cnd;
        for (index_type ifl=0; ifl<getype.clnfc; ++ifl) {
            jdis[ifl] += gsft; // Shift it so that the GGE centroid colocate with the self solution point.
            idis[ifl] -= cnd; // Make it relative to the self solution point.
        }
    }

    Matrix<NDIM> calc_displacement_matrix(index_type ifge) const {
        Matrix<NDIM> dst;
        GEType::fge_facelist_type const & tface = getype.faces[ifge];
        for (index_type ivx=0; ivx<NDIM; ++ivx) { dst[ivx] = idis[tface[ivx]-1]; }
        return dst;
    }

}; /* end struct GradientElement */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
