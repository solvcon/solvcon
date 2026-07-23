#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Pilot 2D canvas drawing tools and their registry.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/universe/ViewTransform2d.hpp>
#include <solvcon/universe/World.hpp>

#include <memory>
#include <span>
#include <string>
#include <vector>

class QPainter;

namespace solvcon
{

/**
 * A point in world coordinates, the unit a drawing gesture is built from.
 *
 * @ingroup group_domain
 */
struct DrawPoint
{
    double x;
    double y;
}; /* end struct DrawPoint */

/**
 * Abstract base class for a 2D canvas drawing tool.
 *
 * @ingroup group_domain
 */
class DrawToolBase
{

public:

    virtual ~DrawToolBase() = default;

    /// Stable tool name shared with the Python binding and the toolbox.
    virtual std::string name() const = 0;

    /// Wether this tool draws a shape or just navigates the view.
    virtual bool can_draw_shape() const = 0;

    /// Paint the rubber-band preview of the gesture `points`. Sets the
    /// shared preview pen, then defers to `paint_outline`.
    void paint_preview(QPainter & painter, ViewTransform2dFp64 const & view, std::span<DrawPoint const> points) const;

    /// Commit the shape described by the gesture `points` into `world`.
    virtual void commit(WorldFp64 & world, std::span<DrawPoint const> points) const = 0;

protected:

    /// Draw the shape's outline for `points`, with the preview pen already
    /// set on `painter` by `paint_preview`.
    virtual void paint_outline(QPainter & painter,
                               ViewTransform2dFp64 const & view,
                               std::span<DrawPoint const> points) const = 0;

}; /* end class DrawToolBase */

/**
 * Get the names of all registered tools, in Painter-toolbox order. The
 * first entry is the default tool (see `default_draw_tool_name`).
 *
 * @return The registered tool names in Painter-toolbox order.
 */
std::vector<std::string> const & draw_tool_names();

/**
 * Name of the default tool a fresh canvas starts with: the first
 * registered tool, which is the pan navigation tool.
 *
 * @return The default tool name.
 */
std::string const & default_draw_tool_name();

/**
 * Build the tool registered under `name`.
 *
 * @param name The registered name of the tool to build.
 * @return A unique pointer to the tool, never null for a valid name.
 * @throw std::invalid_argument for an unknown name.
 */
std::unique_ptr<DrawToolBase> make_draw_tool(std::string const & name);

/**
 * True if `name` is a registered tool R2DWidget accepts. Lets callers
 * validate a name without building a tool.
 *
 * @param name The tool name to check.
 * @return True if `name` is a registered tool, false otherwise.
 */
bool is_draw_tool(std::string const & name);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
