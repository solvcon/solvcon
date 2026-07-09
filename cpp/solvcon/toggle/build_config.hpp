#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Compile-time build switches for the toggle system.
 *
 * A build switch is decided once when the library is built, so it is an
 * inline constexpr the optimizer can fold away, not a runtime toggle in the
 * table. A genuine build switch (for example a future GPU backend flag)
 * belongs here rather than in the runtime store.
 *
 * @ingroup group_core
 */

namespace solvcon
{

namespace build_config
{

/// Whether the pilot uses PySide for its Python integration.
inline constexpr bool use_pyside = true;

} /* end namespace build_config */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
