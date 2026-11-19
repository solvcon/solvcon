#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Umbrella header that aggregates the pilot GUI component headers.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/visual/RDomainWidget.hpp>
#include <solvcon/pilot/app/RManager.hpp>
#ifndef SC_DISABLE_CPP_PYCON
#include <solvcon/pilot/console/RPythonConsoleDockWidget.hpp>
#endif // SC_DISABLE_CPP_PYCON

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
