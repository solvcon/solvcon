#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <QCursor>

namespace solvcon
{

/**
 * The cursor of @p shape, safe to set on macOS.
 *
 * Qt draws the wait, busy, and size-all cursors on macOS from bitmaps that
 * carry an ICC profile. Qt 6.11 releases the color space it builds from that
 * profile before CoreGraphics reads it, which traps on macOS 27.2. The
 * returned cursor draws the same bitmap without the profile, which takes the
 * shared sRGB color space instead. Other shapes and platforms get the plain
 * shape.
 */
QCursor shapeCursor(Qt::CursorShape shape);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
