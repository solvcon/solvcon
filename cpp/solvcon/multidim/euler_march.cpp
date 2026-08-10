/*
 * Copyright (c) 2016, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * Order-0 solution marching for the Euler CESE solver.
 */

#include <solvcon/multidim/euler.hpp>

#include <array>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

// Euler flux Jacobian and flux function for one cell statea. jacos is indexed
// [ieq][jeq][idm] and fcn is indexed [ieq][idm].
template <size_t NDIM>
struct EulerJacobian;

template <>
struct EulerJacobian<2>
{
    static constexpr size_t NEQ = 4;
    static constexpr double TINY = 1.e-60;

    std::array<std::array<std::array<double, 2>, NEQ>, NEQ> jacos = {};
    std::array<std::array<double, 2>, NEQ> fcn = {};

    void update(double gamma, std::array<double, NEQ> const & sol)
    {
        double const ga = gamma;
        double const ga1 = ga - 1;
        double const ga3 = ga - 3;
        double const ga1h = ga1 / 2;
        double const u1 = sol[0] + TINY;
        double const u2 = sol[1];
        double const u3 = sol[2];
        double const u4 = sol[3];

        double const rho2 = u1 * u1;
        double const v1 = u2 / u1;
        double const v1o2 = v1 * v1;
        double const v2 = u3 / u1;
        double const v2o2 = v2 * v2;
        double const ke2 = (u2 * u2 + u3 * u3) / u1;
        double const g1ke2 = ga1 * ke2;
        double const vs = ke2 / u1;
        double const gretot = ga * u4;
        double const getot = gretot / u1;
        double const pr = ga1 * u4 - ga1h * ke2;

        fcn[0] = {u2, u3};
        fcn[1] = {pr + u2 * v1, u2 * v2};
        fcn[2] = {u3 * v1, pr + u3 * v2};
        fcn[3] = {(pr + u4) * v1, (pr + u4) * v2};

        jacos[0][0] = {0, 0};
        jacos[0][1] = {1, 0};
        jacos[0][2] = {0, 1};
        jacos[0][3] = {0, 0};

        jacos[1][0] = {-v1o2 + ga1h * vs, -v1 * v2};
        jacos[1][1] = {-ga3 * v1, v2};
        jacos[1][2] = {-ga1 * v2, v1};
        jacos[1][3] = {ga1, 0};

        jacos[2][0] = {-v2 * v1, -v2o2 + ga1h * vs};
        jacos[2][1] = {v2, -ga1 * v1};
        jacos[2][2] = {v1, -ga3 * v2};
        jacos[2][3] = {0, ga1};

        jacos[3][0] = {(-gretot + g1ke2) * u2 / rho2, (-gretot + g1ke2) * u3 / rho2};
        jacos[3][1] = {getot - ga1h * (vs + 2 * v1o2), -ga1 * v1 * v2};
        jacos[3][2] = {-ga1 * v2 * v1, getot - ga1h * (vs + 2 * v2o2)};
        jacos[3][3] = {ga * v1, ga * v2};
    }
}; /* end struct EulerJacobian<2> */

template <>
struct EulerJacobian<3>
{
    static constexpr size_t NEQ = 5;
    static constexpr double TINY = 1.e-60;

    std::array<std::array<std::array<double, 3>, NEQ>, NEQ> jacos = {};
    std::array<std::array<double, 3>, NEQ> fcn = {};

    void update(double gamma, std::array<double, NEQ> const & sol)
    {
        double const ga = gamma;
        double const ga1 = ga - 1;
        double const ga3 = ga - 3;
        double const ga1h = ga1 / 2;
        double const u1 = sol[0] + TINY;
        double const u2 = sol[1];
        double const u3 = sol[2];
        double const u4 = sol[3];
        double const u5 = sol[4];

        double const rho2 = u1 * u1;
        double const v1 = u2 / u1;
        double const v1o2 = v1 * v1;
        double const v2 = u3 / u1;
        double const v2o2 = v2 * v2;
        double const v3 = u4 / u1;
        double const v3o2 = v3 * v3;
        double const ke2 = (u2 * u2 + u3 * u3 + u4 * u4) / u1;
        double const g1ke2 = ga1 * ke2;
        double const vs = ke2 / u1;
        double const gretot = ga * u5;
        double const getot = gretot / u1;
        double const pr = ga1 * u5 - ga1h * ke2;

        fcn[0] = {u2, u3, u4};
        fcn[1] = {pr + u2 * v1, u2 * v2, u2 * v3};
        fcn[2] = {u3 * v1, pr + u3 * v2, u3 * v3};
        fcn[3] = {u4 * v1, u4 * v2, pr + u4 * v3};
        fcn[4] = {(pr + u5) * v1, (pr + u5) * v2, (pr + u5) * v3};

        jacos[0][0] = {0, 0, 0};
        jacos[0][1] = {1, 0, 0};
        jacos[0][2] = {0, 1, 0};
        jacos[0][3] = {0, 0, 1};
        jacos[0][4] = {0, 0, 0};

        jacos[1][0] = {-v1o2 + ga1h * vs, -v1 * v2, -v1 * v3};
        jacos[1][1] = {-ga3 * v1, v2, v3};
        jacos[1][2] = {-ga1 * v2, v1, 0};
        jacos[1][3] = {-ga1 * v3, 0, v1};
        jacos[1][4] = {ga1, 0, 0};

        jacos[2][0] = {-v2 * v1, -v2o2 + ga1h * vs, -v2 * v3};
        jacos[2][1] = {v2, -ga1 * v1, 0};
        jacos[2][2] = {v1, -ga3 * v2, v3};
        jacos[2][3] = {0, -ga1 * v3, v2};
        jacos[2][4] = {0, ga1, 0};

        jacos[3][0] = {-v3 * v1, -v3 * v2, -v3o2 + ga1h * vs};
        jacos[3][1] = {v3, 0, -ga1 * v1};
        jacos[3][2] = {0, v3, -ga1 * v2};
        jacos[3][3] = {v1, v2, -ga3 * v3};
        jacos[3][4] = {0, 0, ga1};

        jacos[4][0] = {(-gretot + g1ke2) * u2 / rho2,
                       (-gretot + g1ke2) * u3 / rho2,
                       (-gretot + g1ke2) * u4 / rho2};
        jacos[4][1] = {getot - ga1h * (vs + 2 * v1o2), -ga1 * v1 * v2, -ga1 * v1 * v3};
        jacos[4][2] = {-ga1 * v2 * v1, getot - ga1h * (vs + 2 * v2o2), -ga1 * v2 * v3};
        jacos[4][3] = {-ga1 * v3 * v1, -ga1 * v3 * v2, getot - ga1h * (vs + 2 * v3o2)};
        jacos[4][4] = {ga * v1, ga * v2, ga * v3};
    }
}; /* end struct EulerJacobian<3> */

// so0t = -(Jacobian . so1c) for both real and ghost cells.
template <size_t NDIM>
void calc_solt_impl(EulerCore & ec)
{
    constexpr size_t neq = NDIM + 2;
    SimpleArray<double> & so0c = ec.so0c();
    SimpleArray<double> & so0t = ec.so0t();
    SimpleArray<double> & so1c = ec.so1c();
    SimpleArray<double> & gamma = ec.gamma();
    EulerJacobian<NDIM> jaco;
    for (int32_t icl = -ec.ngstcell(); icl < ec.ncell(); ++icl)
    {
        std::array<double, neq> sol = {};
        for (size_t ieq = 0; ieq < neq; ++ieq)
        {
            sol[ieq] = so0c(icl, ieq);
        }
        jaco.update(gamma(icl), sol);
        for (size_t ieq = 0; ieq < neq; ++ieq)
        {
            double val = 0.0;
            for (size_t idm = 0; idm < NDIM; ++idm)
            {
                for (size_t jeq = 0; jeq < neq; ++jeq)
                {
                    val += jaco.jacos[ieq][jeq][idm] * so1c(icl, jeq, idm);
                }
            }
            so0t(icl, ieq) = -val;
        }
    }
}

// so0n = CESE space-time flux integral over the BCEs, per real cell. Neighbor
// CE centroids for ghost cells fall back to the mesh cell centroid
// (mesh.clcnd), matching GradientElement; the cecnd ghost rows carry no mirror
// image.
template <size_t NDIM>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void calc_soln_impl(EulerCore & ec)
{
    constexpr size_t neq = NDIM + 2;
    constexpr size_t fcmnd = StaticMesh::FCMND;
    auto const & msh = *ec.mesh();
    SimpleArray<double> & cevol = ec.cevol();
    SimpleArray<double> & cecnd = ec.cecnd();
    SimpleArray<double> & sfcnd = ec.sfcnd();
    SimpleArray<double> & sfnml = ec.sfnml();
    SimpleArray<double> & so0c = ec.so0c();
    SimpleArray<double> & so0n = ec.so0n();
    SimpleArray<double> & so0t = ec.so0t();
    SimpleArray<double> & so1c = ec.so1c();
    SimpleArray<double> & gamma = ec.gamma();

    double const dt = ec.time_increment();
    double const qdt = dt * 0.25;
    double const hdt = dt * 0.5;
    EulerJacobian<NDIM> jaco;

    for (int32_t icl = 0; icl < ec.ncell(); ++icl)
    {
        int32_t const clnfc = msh.clfcs(icl, 0);
        std::array<double, neq> acc = {};

        for (int32_t ifl = 1; ifl <= clnfc; ++ifl)
        {
            int32_t const ifc = msh.clfcs(icl, ifl);
            int32_t const jcl = msh.fcrcl(ifc, icl);

            // Neighbor CE centroid (cell-centroid mirror for ghost cells).
            std::array<double, NDIM> jcecnd = {};
            for (size_t d = 0; d < NDIM; ++d)
            {
                jcecnd[d] = (jcl >= 0) ? cecnd(jcl, d) : msh.clcnd(jcl, d);
            }
            // Self BCE geometry for this face.
            double const bvol = cevol(icl, ifl);
            std::array<double, NDIM> bcnd = {};
            for (size_t d = 0; d < NDIM; ++d)
            {
                bcnd[d] = cecnd(icl, static_cast<size_t>(ifl) * NDIM + d);
            }

            // Spatial flux (given time): neighbor solution reconstructed at the
            // BCE centroid, weighted by the BCE volume.
            for (size_t ieq = 0; ieq < neq; ++ieq)
            {
                double fusp = so0c(jcl, ieq);
                for (size_t d = 0; d < NDIM; ++d)
                {
                    fusp += (bcnd[d] - jcecnd[d]) * so1c(jcl, ieq, d);
                }
                acc[ieq] += fusp * bvol;
            }

            // Temporal flux (given space): Jacobian of the neighbor state,
            // with the neighbor gamma (ghost rows carry it too).
            std::array<double, neq> jsol = {};
            for (size_t ieq = 0; ieq < neq; ++ieq)
            {
                jsol[ieq] = so0c(jcl, ieq);
            }
            jaco.update(gamma(jcl), jsol);

            int32_t const fcnnd = msh.fcnds(ifc, 0);
            size_t const sf_base = static_cast<size_t>(ifl - 1) * fcmnd;
            for (int32_t inf = 0; inf < fcnnd; ++inf)
            {
                size_t const sfi = sf_base + static_cast<size_t>(inf);
                // Solution at the sub-face center.
                std::array<double, neq> usfc = {};
                for (size_t ieq = 0; ieq < neq; ++ieq)
                {
                    usfc[ieq] = qdt * so0t(jcl, ieq);
                    for (size_t d = 0; d < NDIM; ++d)
                    {
                        usfc[ieq] += (sfcnd(icl, sfi, d) - jcecnd[d]) * so1c(jcl, ieq, d);
                    }
                }
                // Flux derivative dotted with the sub-face normal.
                for (size_t ieq = 0; ieq < neq; ++ieq)
                {
                    double dot = 0.0;
                    for (size_t d2 = 0; d2 < NDIM; ++d2)
                    {
                        double dfcn = jaco.fcn[ieq][d2];
                        for (size_t jeq = 0; jeq < neq; ++jeq)
                        {
                            dfcn += jaco.jacos[ieq][jeq][d2] * usfc[jeq];
                        }
                        dot += dfcn * sfnml(icl, sfi, d2);
                    }
                    acc[ieq] -= hdt * dot;
                }
            }
        }

        double const cvol = cevol(icl, 0);
        for (size_t ieq = 0; ieq < neq; ++ieq)
        {
            so0n(icl, ieq) = acc[ieq] / cvol;
        }
    }
}

} /* end namespace detail */

void EulerCore::calc_solt()
{
    if (2 == m_ndim)
    {
        detail::calc_solt_impl<2>(*this);
    }
    else if (3 == m_ndim)
    {
        detail::calc_solt_impl<3>(*this);
    }
    else
    {
        throw std::invalid_argument("EulerCore::calc_solt: ndim must be 2 or 3");
    }
}

void EulerCore::calc_soln()
{
    if (2 == m_ndim)
    {
        detail::calc_soln_impl<2>(*this);
    }
    else if (3 == m_ndim)
    {
        detail::calc_soln_impl<3>(*this);
    }
    else
    {
        throw std::invalid_argument("EulerCore::calc_soln: ndim must be 2 or 3");
    }
}

void EulerCore::march_substep()
{
    update();
    calc_solt();
    calc_soln();
    bc_soln();
    calc_cfl();
    calc_dsoln();
    bc_dsoln();
}

void EulerCore::march(int_type steps)
{
    for (int_type istep = 0; istep < steps; ++istep)
    {
        for (int_type isub = 0; isub < SUBSTEP_RUN; ++isub)
        {
            march_substep();
        }
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
