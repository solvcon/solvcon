#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <QString>

namespace solvcon
{

/// The path of the file `name` in `solvcon.config.Config.config_home()`.
QString configPath(QString const & name);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
