/*
 * Copyright (c) 2016, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multidim/euler.hpp>

#include <array>
#include <cmath>
#include <stdexcept>

namespace solvcon
{

EulerCore::EulerCore(
    std::shared_ptr<StaticMesh> const & mesh,
    real_type time_increment,
    ctor_passkey const &)
    : m_mesh(mesh)
    , m_time_increment(time_increment)
    , m_ndim(mesh->ndim())
    , m_ncell(static_cast<int_type>(mesh->ncell()))
    , m_ngstcell(static_cast<int_type>(mesh->ngstcell()))
    , m_neq(mesh->ndim() + 2)
{
    initialize_arrays();
    initialize_solution();
    prepare_ce();
}

void EulerCore::initialize_arrays()
{
    ssize_t const total = m_ngstcell + m_ncell;
    ssize_t const ndim = m_ndim;

    m_cevol = SimpleArray<real_type>(
        small_vector<ssize_t>{total, StaticMesh::CLMFC + 1}, 0);
    m_cevol.set_nghost(m_ngstcell);

    m_cecnd = SimpleArray<real_type>(
        small_vector<ssize_t>{
            total,
            (StaticMesh::CLMFC + 1) * ndim},
        0);
    m_cecnd.set_nghost(m_ngstcell);

    m_sfcnd = SimpleArray<real_type>(
        small_vector<ssize_t>{
            m_ncell,
            StaticMesh::CLMFC * StaticMesh::FCMND,
            ndim},
        0);

    m_sfnml = SimpleArray<real_type>(
        small_vector<ssize_t>{
            m_ncell,
            StaticMesh::CLMFC * StaticMesh::FCMND,
            ndim},
        0);
}

void EulerCore::prepare_ce()
{
    if (2 == m_ndim)
    {
        prepare_ce_2d();
    }
    else if (3 == m_ndim)
    {
        prepare_ce_3d();
    }
    else
    {
        throw std::invalid_argument("EulerCore: ndim must be 2 or 3");
    }
}

void EulerCore::prepare_ce_2d()
{
    constexpr size_t ndim = 2;

    auto const & msh = *m_mesh;
    auto const ncell = m_ncell;

    for (int_type icl = 0; icl < ncell; ++icl)
    {
        int_type const clnfc = msh.clfcs(icl, 0);

        real_type volc = 0.0;
        real_type cecnd0 = 0.0;
        real_type cecnd1 = 0.0;

        real_type const crdi0 = msh.clcnd(icl, 0);
        real_type const crdi1 = msh.clcnd(icl, 1);

        for (int_type ifl = 1; ifl <= clnfc; ++ifl)
        {
            int_type const ifc = msh.clfcs(icl, ifl);
            int_type const jcl = msh.fcrcl(ifc, icl);

            real_type const crde0 = msh.clcnd(jcl, 0);
            real_type const crde1 = msh.clcnd(jcl, 1);

            // Face node coordinates.
            int_type const ind0 = msh.fcnds(ifc, 1);
            int_type const ind1 = msh.fcnds(ifc, 2);
            real_type const crd10 = msh.ndcrd(ind0, 0);
            real_type const crd11 = msh.ndcrd(ind0, 1);
            real_type const crd20 = msh.ndcrd(ind1, 0);
            real_type const crd21 = msh.ndcrd(ind1, 1);

            // Inner triangle: cell center + two face nodes.
            real_type const cndi0 = (crd10 + crd20 + crdi0) / 3.0;
            real_type const cndi1 = (crd11 + crd21 + crdi1) / 3.0;
            real_type const voli =
                std::fabs(
                    (crd10 - crdi0) * (crd21 - crdi1) -
                    (crd11 - crdi1) * (crd20 - crdi0)) /
                2.0;

            // Outer triangle: neighbor center + two face nodes.
            real_type const cnde0 = (crd10 + crd20 + crde0) / 3.0;
            real_type const cnde1 = (crd11 + crd21 + crde1) / 3.0;
            real_type const vole =
                std::fabs(
                    (crd10 - crde0) * (crd21 - crde1) -
                    (crd11 - crde1) * (crd20 - crde0)) /
                2.0;

            // BCE volume and centroid.
            real_type const volb = voli + vole;
            m_cevol(icl, ifl) = volb;

            size_t const cecnd_col = static_cast<size_t>(ifl) * ndim;
            m_cecnd(icl, cecnd_col + 0) =
                (cndi0 * voli + cnde0 * vole) / volb;
            m_cecnd(icl, cecnd_col + 1) =
                (cndi1 * voli + cnde1 * vole) / volb;

            // Accumulate CCE.
            volc += volb;
            cecnd0 += (cndi0 * voli + cnde0 * vole);
            cecnd1 += (cndi1 * voli + cnde1 * vole);

            // Sub-face metrics.
            size_t const sf_base =
                static_cast<size_t>(ifl - 1) * StaticMesh::FCMND;

            // Sub-face 0: from crde to crd1 (first face node).
            bool const outward =
                (msh.fccls(ifc, 0) == icl);
            real_type const voe_sign = outward ? 1.0 : -1.0;

            // Sub-face centroid: midpoint of crde and the node.
            m_sfcnd(icl, sf_base + 0, 0) = (crde0 + crd10) / 2.0;
            m_sfcnd(icl, sf_base + 0, 1) = (crde1 + crd11) / 2.0;
            // Sub-face normal (rotated edge, outward).
            real_type const sf0_dx = crde0 - crd10;
            real_type const sf0_dy = crde1 - crd11;
            m_sfnml(icl, sf_base + 0, 0) = -sf0_dy * voe_sign;
            m_sfnml(icl, sf_base + 0, 1) = sf0_dx * voe_sign;

            // Sub-face 1: from crd2 (second face node) to crde.
            m_sfcnd(icl, sf_base + 1, 0) = (crd20 + crde0) / 2.0;
            m_sfcnd(icl, sf_base + 1, 1) = (crd21 + crde1) / 2.0;
            real_type const sf1_dx = crd20 - crde0;
            real_type const sf1_dy = crd21 - crde1;
            m_sfnml(icl, sf_base + 1, 0) = -sf1_dy * voe_sign;
            m_sfnml(icl, sf_base + 1, 1) = sf1_dx * voe_sign;
        }

        // Store CCE (index 0).
        m_cevol(icl, 0) = volc;
        m_cecnd(icl, 0) = cecnd0 / volc;
        m_cecnd(icl, 1) = cecnd1 / volc;
    }
}

void EulerCore::prepare_ce_3d()
{
    constexpr size_t ndim = 3;

    auto const & msh = *m_mesh;
    auto const ncell = m_ncell;

    for (int_type icl = 0; icl < ncell; ++icl)
    {
        int_type const clnfc = msh.clfcs(icl, 0);

        real_type volc = 0.0;
        real_type cecnd0 = 0.0;
        real_type cecnd1 = 0.0;
        real_type cecnd2 = 0.0;

        real_type const crdi0 = msh.clcnd(icl, 0);
        real_type const crdi1 = msh.clcnd(icl, 1);
        real_type const crdi2 = msh.clcnd(icl, 2);

        for (int_type ifl = 1; ifl <= clnfc; ++ifl)
        {
            int_type const ifc = msh.clfcs(icl, ifl);
            int_type const jcl = msh.fcrcl(ifc, icl);
            int_type const fcnnd = msh.fcnds(ifc, 0);

            real_type const crde0 = msh.clcnd(jcl, 0);
            real_type const crde1 = msh.clcnd(jcl, 1);
            real_type const crde2 = msh.clcnd(jcl, 2);

            real_type const pfccnd0 = msh.fccnd(ifc, 0);
            real_type const pfccnd1 = msh.fccnd(ifc, 1);
            real_type const pfccnd2 = msh.fccnd(ifc, 2);

            // Collect face node coordinates.
            std::array<std::array<real_type, 3>, StaticMesh::FCMND + 2> crd{};
            for (int_type inf = 1; inf <= fcnnd; ++inf)
            {
                int_type const ind = msh.fcnds(ifc, inf);
                crd[inf][0] = msh.ndcrd(ind, 0);
                crd[inf][1] = msh.ndcrd(ind, 1);
                crd[inf][2] = msh.ndcrd(ind, 2);
            }
            // Wrap around for the loop.
            crd[fcnnd + 1][0] = crd[1][0];
            crd[fcnnd + 1][1] = crd[1][1];
            crd[fcnnd + 1][2] = crd[1][2];

            real_type volb = 0.0;
            real_type bcecnd0 = 0.0;
            real_type bcecnd1 = 0.0;
            real_type bcecnd2 = 0.0;

            bool const outward = (msh.fccls(ifc, 0) == icl);
            real_type const voe = outward ? 0.5 : -0.5;

            size_t const sf_base =
                static_cast<size_t>(ifl - 1) * StaticMesh::FCMND;

            for (int_type inf = 1; inf <= fcnnd; ++inf)
            {
                // Base triangle vectors from face center.
                real_type const disu0 = crd[inf][0] - pfccnd0;
                real_type const disu1 = crd[inf][1] - pfccnd1;
                real_type const disu2 = crd[inf][2] - pfccnd2;
                real_type const disv0 = crd[inf + 1][0] - pfccnd0;
                real_type const disv1 = crd[inf + 1][1] - pfccnd1;
                real_type const disv2 = crd[inf + 1][2] - pfccnd2;

                // Cross product (base triangle normal * 2).
                real_type const dist0 =
                    disu1 * disv2 - disu2 * disv1;
                real_type const dist1 =
                    disu2 * disv0 - disu0 * disv2;
                real_type const dist2 =
                    disu0 * disv1 - disu1 * disv0;

                // Inner tetrahedron.
                real_type const diswi0 = crdi0 - pfccnd0;
                real_type const diswi1 = crdi1 - pfccnd1;
                real_type const diswi2 = crdi2 - pfccnd2;
                real_type const voli =
                    std::fabs(
                        dist0 * diswi0 +
                        dist1 * diswi1 +
                        dist2 * diswi2) /
                    6.0;
                real_type const cndi0 =
                    (crd[inf][0] + crd[inf + 1][0] + pfccnd0 + crdi0) /
                    4.0;
                real_type const cndi1 =
                    (crd[inf][1] + crd[inf + 1][1] + pfccnd1 + crdi1) /
                    4.0;
                real_type const cndi2 =
                    (crd[inf][2] + crd[inf + 1][2] + pfccnd2 + crdi2) /
                    4.0;

                // Outer tetrahedron.
                real_type const diswe0 = crde0 - pfccnd0;
                real_type const diswe1 = crde1 - pfccnd1;
                real_type const diswe2 = crde2 - pfccnd2;
                real_type const vole =
                    std::fabs(
                        dist0 * diswe0 +
                        dist1 * diswe1 +
                        dist2 * diswe2) /
                    6.0;
                real_type const cnde0 =
                    (crd[inf][0] + crd[inf + 1][0] + pfccnd0 + crde0) /
                    4.0;
                real_type const cnde1 =
                    (crd[inf][1] + crd[inf + 1][1] + pfccnd1 + crde1) /
                    4.0;
                real_type const cnde2 =
                    (crd[inf][2] + crd[inf + 1][2] + pfccnd2 + crde2) /
                    4.0;

                // Accumulate BCE.
                volb += voli + vole;
                bcecnd0 += cndi0 * voli + cnde0 * vole;
                bcecnd1 += cndi1 * voli + cnde1 * vole;
                bcecnd2 += cndi2 * voli + cnde2 * vole;

                // Sub-face centroid: average of crde and the two
                // consecutive face nodes.
                size_t const sf_idx = sf_base + (inf - 1);
                m_sfcnd(icl, sf_idx, 0) =
                    (crde0 + crd[inf][0] + crd[inf + 1][0]) / 3.0;
                m_sfcnd(icl, sf_idx, 1) =
                    (crde1 + crd[inf][1] + crd[inf + 1][1]) / 3.0;
                m_sfcnd(icl, sf_idx, 2) =
                    (crde2 + crd[inf][2] + crd[inf + 1][2]) / 3.0;

                // Sub-face normal: cross(node-crde, next_node-crde) *
                // sign.
                real_type const sfu0 = crd[inf][0] - crde0;
                real_type const sfu1 = crd[inf][1] - crde1;
                real_type const sfu2 = crd[inf][2] - crde2;
                real_type const sfv0 = crd[inf + 1][0] - crde0;
                real_type const sfv1 = crd[inf + 1][1] - crde1;
                real_type const sfv2 = crd[inf + 1][2] - crde2;
                m_sfnml(icl, sf_idx, 0) =
                    (sfu1 * sfv2 - sfu2 * sfv1) * voe;
                m_sfnml(icl, sf_idx, 1) =
                    (sfu2 * sfv0 - sfu0 * sfv2) * voe;
                m_sfnml(icl, sf_idx, 2) =
                    (sfu0 * sfv1 - sfu1 * sfv0) * voe;
            }

            // Store BCE.
            m_cevol(icl, ifl) = volb;
            size_t const cecnd_col = static_cast<size_t>(ifl) * ndim;
            m_cecnd(icl, cecnd_col + 0) = bcecnd0 / volb;
            m_cecnd(icl, cecnd_col + 1) = bcecnd1 / volb;
            m_cecnd(icl, cecnd_col + 2) = bcecnd2 / volb;

            // Accumulate CCE.
            volc += volb;
            cecnd0 += bcecnd0;
            cecnd1 += bcecnd1;
            cecnd2 += bcecnd2;
        }

        // Store CCE (index 0).
        m_cevol(icl, 0) = volc;
        m_cecnd(icl, 0) = cecnd0 / volc;
        m_cecnd(icl, 1) = cecnd1 / volc;
        m_cecnd(icl, 2) = cecnd2 / volc;
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
