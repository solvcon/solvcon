#pragma once

/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <cstdint>

#include "march/core/core.hpp"
#include "march/mesh/mesh.hpp"

#include "march/gas/Solution.hpp"

namespace march {

namespace gas {

class Parameter
{

public:

    using int_type = int32_t;

    Parameter() = default;
    Parameter(Parameter const & ) = default;
    Parameter(Parameter       &&) = delete;
    Parameter & operator=(Parameter const & ) = default;
    Parameter & operator=(Parameter       &&) = delete;

    int_type   sigma0() const { return m_sigma0; }
    int_type & sigma0()       { return m_sigma0; }
    real_type   taumin() const { return m_taumin; }
    real_type & taumin()       { return m_taumin; }
    real_type   tauscale() const { return m_tauscale; }
    real_type & tauscale()       { return m_tauscale; }

private:

    int_type m_sigma0=3;
    real_type m_taumin=0.0;
    real_type m_tauscale=1.0;

#define DECL_MARCH_DEBUG(TYPE, NAME, DEFAULT) \
public: \
    TYPE   NAME() const { return m_##NAME; } \
    TYPE & NAME()       { return m_##NAME; } \
private: \
    TYPE m_##NAME = DEFAULT;

DECL_MARCH_DEBUG(real_type, stop_on_negative_density, 1.e-50)
DECL_MARCH_DEBUG(real_type, stop_on_negative_energy, 1.e-50)
DECL_MARCH_DEBUG(bool, stop_on_cfl_adjustment, true)
DECL_MARCH_DEBUG(bool, stop_on_cfl_overflow, true)

#undef DECL_MARCH_DEBUG_BOOL

}; /* end class Parameter */

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
