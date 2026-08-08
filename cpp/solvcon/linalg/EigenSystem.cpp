/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <memory>
#include <stdexcept>
#include <variant>

// EigenSystem.hpp requires a vendor LAPACK; only compile the surrogate's
// out-of-line bodies when one is available.  Without it this is an empty TU.
#ifdef SC_HAS_VENDOR_LAPACK

#include <solvcon/linalg/EigenSystem.hpp>

namespace solvcon
{

// Reinterpret the plex's type-erased instance as the typed SimpleArray<T> it
// actually holds; the caller must have verified data_type() == T.
template <typename T>
static SimpleArray<T> const & typed_array(SimpleArrayPlex const & plex)
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return *reinterpret_cast<SimpleArray<T> const *>(plex.instance_ptr());
}

EigenSystemPlex::EigenSystemPlex(SimpleArrayPlex const & matrix, bool do_vl, bool do_vr)
    : m_matrix(matrix)
    , m_data_type(matrix.data_type())
{
    switch (matrix.data_type())
    {
    case DataType::Float32:
        m_solver = EigenSystem<float>::construct(typed_array<float>(matrix), do_vl, do_vr);
        break;
    case DataType::Float64:
        m_solver = EigenSystem<double>::construct(typed_array<double>(matrix), do_vl, do_vr);
        break;
    case DataType::Complex64:
        m_solver = EigenSystem<Complex<float>>::construct(typed_array<Complex<float>>(matrix), do_vl, do_vr);
        break;
    case DataType::Complex128:
        m_solver = EigenSystem<Complex<double>>::construct(typed_array<Complex<double>>(matrix), do_vl, do_vr);
        break;
    default:
        throw std::invalid_argument(
            "EigenSystemPlex: SimpleArray data type must be float32, float64, complex64, or complex128");
    }
}

void EigenSystemPlex::run()
{
    std::visit([](auto const & solver)
               { solver->run(); },
               m_solver);
}

SimpleArrayPlex EigenSystemPlex::wr() const
{
    return std::visit(
        [](auto const & solver)
        { return SimpleArrayPlex(solver->wr()); },
        m_solver);
}

SimpleArrayPlex EigenSystemPlex::wi() const
{
    return std::visit(
        [](auto const & solver)
        { return SimpleArrayPlex(solver->wi()); },
        m_solver);
}

SimpleArrayPlex EigenSystemPlex::vl(bool suppress_exception) const
{
    return std::visit(
        [suppress_exception](auto const & solver)
        { return SimpleArrayPlex(solver->vl(suppress_exception)); },
        m_solver);
}

SimpleArrayPlex EigenSystemPlex::vr(bool suppress_exception) const
{
    return std::visit(
        [suppress_exception](auto const & solver)
        { return SimpleArrayPlex(solver->vr(suppress_exception)); },
        m_solver);
}

bool EigenSystemPlex::do_vl() const
{
    return std::visit([](auto const & solver)
                      { return solver->do_vl(); },
                      m_solver);
}

bool EigenSystemPlex::do_vr() const
{
    return std::visit([](auto const & solver)
                      { return solver->do_vr(); },
                      m_solver);
}

bool EigenSystemPlex::done() const
{
    return std::visit([](auto const & solver)
                      { return solver->done(); },
                      m_solver);
}

} /* end namespace solvcon */

#endif /* SC_HAS_VENDOR_LAPACK */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
