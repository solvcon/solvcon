#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Device identity for owned buffer storage.
 * @ingroup group_core */

#include <cstdint>

namespace solvcon
{

/**
 * Identifies where owned buffer storage resides.
 * @ingroup group_core */
enum class BufferDevice : std::uint8_t
{
    Cpu,
    Metal,
}; /* end enum class BufferDevice */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
