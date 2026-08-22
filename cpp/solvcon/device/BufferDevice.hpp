#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Device identity shared by buffer storage backends.
 *
 * @ingroup group_core
 */

#include <cstdint>
#include <string_view>

namespace solvcon
{

enum class BufferDevice : std::uint8_t
{
    Cpu,
    Metal,
}; /* end enum class BufferDevice */

constexpr std::string_view buffer_device_name(BufferDevice device) noexcept
{
    switch (device)
    {
    case BufferDevice::Cpu:
        return "cpu";
    case BufferDevice::Metal:
        return "metal";
    }
    return "unknown";
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
