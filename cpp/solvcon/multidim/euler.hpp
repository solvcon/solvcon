#pragma once

/*
 * Copyright (c) 2016, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The space-time CESE solver for the Euler equations.
 *
 * @ingroup group_multidim
 */

#include <solvcon/mesh/mesh.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace solvcon
{

// Boundary-condition kind for the ghost-cell trim passes.  NonReflective also
// realizes outflow; there is no separate outlet handler.
enum class EulerBC : uint8_t
{
    NonReflective = 0,
    SlipWall = 1,
    Inlet = 2,
}; /* end enum class EulerBC */

/**
 * One registered boundary condition: a handler kind over a set of boundary
 * faces (global face indices). value carries the Inlet free stream as
 * [rho, v(ndim), p, gamma] and is unused by the other kinds.
 *
 * @ingroup group_multidim
 */
struct EulerBoundary
{
    EulerBC kind = EulerBC::NonReflective;
    SimpleCollector<int32_t> faces;
    SimpleCollector<double> value;
}; /* end struct EulerBoundary */

/**
 * Multi-dimensional CESE solver core for the compressible Euler equations.
 *
 * Advances the conserved fields on a StaticMesh by one space-time step per
 * call. The conserved set is density, the ndim momentum components, and total
 * energy. Construct through EulerCore::construct so the instance is shared.
 *
 * @ingroup group_multidim
 */
class EulerCore
    : public NumberBase<int32_t, double>
    , public std::enable_shared_from_this<EulerCore>
{

private:

    class ctor_passkey
    {
    }; /* end class ctor_passkey */

public:

    using number_base = NumberBase<int32_t, double>;
    using int_type = typename number_base::int_type;
    using uint_type = typename number_base::uint_type;
    using real_type = typename number_base::real_type;

    template <typename... Args>
    static std::shared_ptr<EulerCore> construct(Args &&... args)
    {
        return std::make_shared<EulerCore>(std::forward<Args>(args)..., ctor_passkey());
    }

    EulerCore(std::shared_ptr<StaticMesh> const & mesh, real_type time_increment, ctor_passkey const &);

    EulerCore() = delete;
    EulerCore(EulerCore const &) = delete;
    EulerCore(EulerCore &&) = delete;
    EulerCore & operator=(EulerCore const &) = delete;
    EulerCore & operator=(EulerCore &&) = delete;
    ~EulerCore() = default;

    std::shared_ptr<StaticMesh> const & mesh() const { return m_mesh; }
    uint8_t ndim() const { return m_ndim; }
    int_type ncell() const { return m_ncell; }
    int_type ngstcell() const { return m_ngstcell; }
    // Number of conserved equations: density + ndim momentum + total energy.
    int_type neq() const { return m_neq; }
    real_type time_increment() const { return m_time_increment; }

    real_type sigma0() const { return m_sigma0; }
    real_type taumin() const { return m_taumin; }
    real_type tauscale() const { return m_tauscale; }
    void set_sigma0(real_type v) { m_sigma0 = v; }
    void set_taumin(real_type v) { m_taumin = v; }
    void set_tauscale(real_type v) { m_tauscale = v; }

    SimpleArray<real_type> & cevol() { return m_cevol; }
    SimpleArray<real_type> & cecnd() { return m_cecnd; }
    SimpleArray<real_type> & sfcnd() { return m_sfcnd; }
    SimpleArray<real_type> & sfnml() { return m_sfnml; }

    SimpleArray<real_type> & so0c() { return m_so0c; }
    SimpleArray<real_type> & so0n() { return m_so0n; }
    SimpleArray<real_type> & so0t() { return m_so0t; }
    SimpleArray<real_type> & so1c() { return m_so1c; }
    SimpleArray<real_type> & so1n() { return m_so1n; }
    SimpleArray<real_type> & stm() { return m_stm; }
    SimpleArray<real_type> & cflo() { return m_cflo; }
    SimpleArray<real_type> & cflc() { return m_cflc; }
    SimpleArray<real_type> & gamma() { return m_gamma; }

    void prepare_ce();

    void init_solution(real_type gamma, real_type rho, std::array<real_type, 3> const & velocity, real_type p);
    void calc_cfl();
    void update()
    {
        m_so0c.swap(m_so0n);
        m_so1c.swap(m_so1n);
    }

    // Solution marching: calc_solt, calc_soln, calc_dsoln.

    // calc_solt fills the order-0 temporal derivative so0t from the Jacobian
    // and so1c.
    void calc_solt();
    // calc_soln advances the order-0 solution so0n with the CESE flux integral
    // over the BCEs.
    void calc_soln();
    // calc_dsoln reconstructs the order-1 derivative so1n from a per-cell
    // GradientElement plus the gradient weighting
    void calc_dsoln();

    // march_substep runs one CESE substep and march runs the requested number
    // of full steps (SUBSTEP_RUN substeps per step).
    void march_substep();
    void march(int_type steps);

    // Boundary conditions: ghost-cell trimming.

    // Register a boundary condition of the given kind over the listed global
    // boundary face indices.  For Inlet, value is the free stream
    // [rho, v(ndim), p, gamma]; it is ignored (and may be empty) otherwise.
    void add_bc(EulerBC kind, std::vector<int_type> const & faces, std::vector<real_type> const & value);
    void clear_bc() { m_boundaries.clear(); }
    std::vector<EulerBoundary> const & boundaries() const { return m_boundaries; }

    // Orthonormal frame for face ifc with the outward unit normal as the first
    // row: a 2D rotation, or in 3D the normal plus a stable tangent pair.
    std::vector<std::vector<real_type>> get_normal_matrix(int_type ifc) const;

    // bc_soln is the order-0 ghost update (trim_do0); bc_dsoln is the order-1
    // ghost update (trim_do1).  Both run over every registered boundary.
    void bc_soln();
    void bc_dsoln();

    // Number of CESE substeps per full marching step.
    static constexpr int_type SUBSTEP_RUN = 2;

private:

    void initialize_arrays();
    void initialize_solution();

    void prepare_ce_2d();
    void prepare_ce_3d();

    std::shared_ptr<StaticMesh> m_mesh;
    real_type m_time_increment = 0.0;

    uint8_t m_ndim = 0;
    int_type m_ncell = 0;
    int_type m_ngstcell = 0;
    int_type m_neq = 0;

    // sigma0 caps the gradient weighting
    real_type m_sigma0 = 3.0;
    // tau (taumin + |cfl| * tauscale) sets the per-cell gradient-element
    // spread.
    real_type m_taumin = 0.0;
    real_type m_tauscale = 1.0;

    // CE geometry arrays.
    SimpleArray<real_type> m_cevol; // [ncell, CLMFC+1]
    SimpleArray<real_type> m_cecnd; // [ncell, (CLMFC+1)*ndim]
    SimpleArray<real_type> m_sfcnd; // [ncell, CLMFC*FCMND, ndim]
    SimpleArray<real_type> m_sfnml; // [ncell, CLMFC*FCMND, ndim]

    // Solution arrays.  The first axis spans ghost + body cells (set_nghost),
    // so every table carries ghost rows; neq = ndim + 2.
    SimpleArray<real_type> m_so0c; // [total, neq] order-0, current step
    SimpleArray<real_type> m_so0n; // [total, neq] order-0, new step
    SimpleArray<real_type> m_so0t; // [total, neq] order-0, temporal
    SimpleArray<real_type> m_so1c; // [total, neq, ndim] order-1, current step
    SimpleArray<real_type> m_so1n; // [total, neq, ndim] order-1, new step
    SimpleArray<real_type> m_stm; // [total, neq] solution time-marching scratch
    SimpleArray<real_type> m_cflo; // [total] original CFL number
    SimpleArray<real_type> m_cflc; // [total] clamped CFL number
    SimpleArray<real_type> m_gamma; // [total] ratio of specific heat

    // Registered boundary conditions applied by bc_soln / bc_dsoln.
    std::vector<EulerBoundary> m_boundaries;

}; /* end class EulerCore */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
