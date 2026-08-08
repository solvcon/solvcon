#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Derived geometric facts for the "diagnostics" level of World::describe_state.
 * These are facts a careful viewer could read off the rendered image but that
 * the basic state never spells out: proper crossings between drawn segments,
 * and shapes that have collapsed to a lower dimension. They sit in a distinct
 * level so the basic output stays a byte-for-byte subset for callers that do
 * not ask for them.
 *
 * @ingroup group_geometry
 */

#include <solvcon/buffer/small_vector.hpp>
#include <solvcon/serialization/SerializableItem.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace solvcon
{

/**
 * One proper crossing between two drawn segments.
 *
 * @ingroup group_geometry
 */
class WorldIntersection : public SerializableItem
{

public:

    WorldIntersection() = default;
    WorldIntersection(WorldIntersection const &) = default;
    WorldIntersection(WorldIntersection &&) = default;
    WorldIntersection & operator=(WorldIntersection const &) = default;
    WorldIntersection & operator=(WorldIntersection &&) = default;
    ~WorldIntersection() override = default;

    WorldIntersection(int32_t shape_a, int32_t shape_b, double x, double y)
        : m_shapes{shape_a, shape_b}
        , m_point{x, y}
    {
    }

    SC_DECL_SERIALIZABLE(
        register_member("shapes", m_shapes);
        register_member("point", m_point);)

private:

    std::vector<int32_t> m_shapes; ///< the two owning shape ids; -1 marks a bare segment
    std::vector<double> m_point; ///< {x, y} of the crossing

}; /* end class WorldIntersection */

/**
 * One shape (or bare primitive) that has collapsed to a lower dimension.
 *
 * @ingroup group_geometry
 */
class WorldDegeneracy : public SerializableItem
{

public:

    WorldDegeneracy() = default;
    WorldDegeneracy(WorldDegeneracy const &) = default;
    WorldDegeneracy(WorldDegeneracy &&) = default;
    WorldDegeneracy & operator=(WorldDegeneracy const &) = default;
    WorldDegeneracy & operator=(WorldDegeneracy &&) = default;
    ~WorldDegeneracy() override = default;

    WorldDegeneracy(int32_t shape, std::string type, std::string reason)
        : m_shape(shape)
        , m_type(std::move(type))
        , m_reason(std::move(reason))
    {
    }

    SC_DECL_SERIALIZABLE(
        register_member("shape", m_shape);
        register_member("type", m_type);
        register_member("reason", m_reason);)

private:

    int32_t m_shape = -1; ///< owning shape id; -1 for bare geometry
    std::string m_type; ///< primitive kind, e.g. "triangle", "circle", "segment"
    std::string m_reason; ///< why it is degenerate, e.g. "collinear"

}; /* end class WorldDegeneracy */

/**
 * JSON view of the world's derived facts: crossings and degeneracies. The
 * arrays are empty (but present) when nothing is found.
 *
 * @ingroup group_geometry
 */
class WorldDiagnostics : public SerializableItem
{

public:

    WorldDiagnostics() = default;
    WorldDiagnostics(WorldDiagnostics const &) = default;
    WorldDiagnostics(WorldDiagnostics &&) = default;
    WorldDiagnostics & operator=(WorldDiagnostics const &) = default;
    WorldDiagnostics & operator=(WorldDiagnostics &&) = default;
    ~WorldDiagnostics() override = default;

    void add_intersection(int32_t shape_a, int32_t shape_b, double x, double y)
    {
        m_intersections.emplace_back(shape_a, shape_b, x, y);
    }

    void add_degeneracy(int32_t shape, std::string type, std::string reason)
    {
        m_degeneracies.emplace_back(shape, std::move(type), std::move(reason));
    }

    SC_DECL_SERIALIZABLE(
        register_member("intersections", m_intersections);
        register_member("degeneracies", m_degeneracies);)

private:

    std::vector<WorldIntersection> m_intersections;
    std::vector<WorldDegeneracy> m_degeneracies;

}; /* end class WorldDiagnostics */

namespace detail
{

/**
 * Relative tolerance for the diagnostic predicates, matching the scale the
 * polygon boolean sweep-line uses. Diagnostics are computed in double
 * regardless of the world's coordinate type, so the double epsilon applies.
 */
inline constexpr double diag_eps = std::numeric_limits<double>::epsilon() * 100;

/// Z component of the 2D cross product of (ax, ay) and (bx, by).
inline double diag_cross(double ax, double ay, double bx, double by)
{
    return ax * by - ay * bx;
}

/**
 * Collapse a signed zero to positive zero so a serialized coordinate never
 * reads as "-0.000000".
 */
inline double diag_norm_zero(double v)
{
    return v == 0.0 ? 0.0 : v;
}

/**
 * Largest coordinate magnitude among the values, floored at 1, used to scale
 * the absolute tolerance of the degeneracy tests.
 */
inline double diag_scale(std::initializer_list<double> values)
{
    double scale = 1.0;
    for (double const v : values)
    {
        scale = std::max(scale, std::abs(v));
    }
    return scale;
}

/**
 * The single point strictly interior to both segments where they cross, or
 * nullopt when they are parallel, collinear, or only touch at an endpoint.
 */
inline std::optional<small_vector<double, 2>> segment_proper_intersection(
    double ax0, double ay0, double ax1, double ay1, double bx0, double by0, double bx1, double by1)
{
    double const dax = ax1 - ax0;
    double const day = ay1 - ay0;
    double const dbx = bx1 - bx0;
    double const dby = by1 - by0;
    double const denom = diag_cross(dax, day, dbx, dby);
    double const len_a = std::sqrt(dax * dax + day * day);
    double const len_b = std::sqrt(dbx * dbx + dby * dby);
    // Parallel, collinear, or a zero-length segment: no single proper crossing.
    if (std::abs(denom) <= diag_eps * std::max(len_a * len_b, 1.0))
    {
        return std::nullopt;
    }
    double const t = diag_cross(bx0 - ax0, by0 - ay0, dbx, dby) / denom;
    double const s = diag_cross(bx0 - ax0, by0 - ay0, dax, day) / denom;
    // Require both parameters strictly inside (0, 1) so shared endpoints and
    // T-junctions do not count as crossings.
    if (t <= diag_eps || t >= 1.0 - diag_eps || s <= diag_eps || s >= 1.0 - diag_eps)
    {
        return std::nullopt;
    }
    return small_vector<double, 2>{diag_norm_zero(ax0 + t * dax), diag_norm_zero(ay0 + t * day)};
}

/// A segment whose endpoints coincide (it renders as a point, not a line).
inline bool is_zero_length(double x0, double y0, double x1, double y1)
{
    double const len = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
    return len <= diag_eps * diag_scale({x0, y0, x1, y1});
}

/**
 * Three vertices on one line (zero-area triangle), including the case where
 * two of them coincide.
 */
inline bool is_collinear(double ax, double ay, double bx, double by, double cx, double cy)
{
    double const area2 = diag_cross(bx - ax, by - ay, cx - ax, cy - ay);
    double const len_ab = std::sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
    double const len_ac = std::sqrt((cx - ax) * (cx - ax) + (cy - ay) * (cy - ay));
    return std::abs(area2) <= diag_eps * std::max(len_ab * len_ac, 1.0);
}

/// An axis-aligned box that has collapsed in x or y (zero width or height).
inline bool is_zero_extent(double min_x, double min_y, double max_x, double max_y)
{
    double const scale = diag_scale({min_x, min_y, max_x, max_y});
    return (max_x - min_x) <= diag_eps * scale || (max_y - min_y) <= diag_eps * scale;
}

/**
 * Four cubic control points that all coincide (the curve renders as a point).
 * Both extents must collapse; a curve flat in only one axis is still a line.
 */
inline bool is_coincident_controls(
    double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3)
{
    double const min_x = std::min({x0, x1, x2, x3});
    double const max_x = std::max({x0, x1, x2, x3});
    double const min_y = std::min({y0, y1, y2, y3});
    double const max_y = std::max({y0, y1, y2, y3});
    double const scale = diag_scale({x0, y0, x1, y1, x2, y2, x3, y3});
    return (max_x - min_x) <= diag_eps * scale && (max_y - min_y) <= diag_eps * scale;
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
