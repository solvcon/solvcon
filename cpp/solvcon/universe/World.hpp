#pragma once

/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Geometry world container for points, segments, cubic Bezier curves, and
 * shapes, with an R-tree spatial index.
 *
 * @ingroup group_geometry
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/SimpleCollector.hpp>
#include <solvcon/serialization/SerializableItem.hpp>
#include <solvcon/universe/WorldDiagnostics.hpp>
#include <solvcon/universe/bernstein.hpp>
#include <solvcon/universe/bezier.hpp>
#include <solvcon/universe/rtree.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace solvcon
{

/**
 * Kind of geometry shape, stored as a one-byte tag.
 *
 * DEAD marks a deleted or unused registry slot; the remaining values name
 * the 0D, 1D, and 2D shapes the world can hold.
 *
 * @ingroup group_geometry
 */
enum class ShapeType : uint8_t
{
    DEAD = 0, ///< Deleted or unused slot.
    POINT = 1,

    // Rectilinear family (straight edges only).
    LINE = 2,
    POLYLINE = 3, ///< Open chain of straight segments.
    POLYGON = 4, ///< Closed chain of straight segments.
    TRIANGLE = 5,
    RECTANGLE = 6,
    SQUARE = 7, ///< Specialization of RECTANGLE with equal side lengths.

    // Curvilinear family (one or more curved edges).
    BEZIER = 8, ///< Single cubic Bezier curve.
    ELLIPSE = 9,
    CIRCLE = 10, ///< Specialization of ELLIPSE with equal radii.
}; /* end of enum class ShapeType */

inline std::string shape_type_name(ShapeType st)
{
    switch (st)
    {
    case ShapeType::DEAD: return "DEAD";
    case ShapeType::POINT: return "point";
    case ShapeType::LINE: return "line";
    case ShapeType::POLYLINE: return "polyline";
    case ShapeType::POLYGON: return "polygon";
    case ShapeType::TRIANGLE: return "triangle";
    case ShapeType::RECTANGLE: return "rectangle";
    case ShapeType::SQUARE: return "square";
    case ShapeType::BEZIER: return "bezier";
    case ShapeType::ELLIPSE: return "ellipse";
    case ShapeType::CIRCLE: return "circle";
    default: return "unknown";
    }
}

/**
 * Level of detail for World::describe_state. C++ callers pass the enum; the
 * Python binding accepts the equivalent lower-case string.
 *
 * @ingroup group_geometry
 */
enum class DescribeLevel : uint8_t
{
    BASIC = 0, ///< Only what the 2D image draws.
    DIAGNOSTICS = 1, ///< BASIC plus derived facts: intersections, degeneracies.
}; /* end enum class DescribeLevel */

inline DescribeLevel describe_level_from_string(std::string const & level)
{
    if (level == "basic")
    {
        return DescribeLevel::BASIC;
    }
    if (level == "diagnostics")
    {
        return DescribeLevel::DIAGNOSTICS;
    }
    throw std::invalid_argument(
        std::format("World: describe_state level '{}' not supported", level));
}

// TODO: The shared JSON tool formats numbers with std::to_string, which keeps
// only 6 decimal places. Coordinates that differ past the 6th decimal then
// serialize identically and small magnitudes round to 0, which can confuse the
// numeric oracles downstream.
/**
 * JSON-serializable view of one shape's rendered 2D geometry: its id, type,
 * bounding box, segment endpoints, and curve control points. The z component
 * is not rendered, so coordinates are 2D.
 *
 * @ingroup group_geometry
 */
class WorldShapeState : public SerializableItem
{

public:

    using coords_type = std::vector<double>;
    using segment_list_type = std::vector<std::vector<double>>;
    using curve_list_type = std::vector<std::vector<std::vector<double>>>;

    WorldShapeState() = default;
    WorldShapeState(WorldShapeState const &) = default;
    WorldShapeState(WorldShapeState &&) = default;
    WorldShapeState & operator=(WorldShapeState const &) = default;
    WorldShapeState & operator=(WorldShapeState &&) = default;
    ~WorldShapeState() override = default;

    int32_t id() const { return m_id; }
    int32_t & id() { return m_id; }

    ShapeType type() const { return m_type; }
    ShapeType & type() { return m_type; }

    coords_type const & bbox() const { return m_bbox; }
    coords_type & bbox() { return m_bbox; }

    segment_list_type const & segments() const { return m_segments; }
    segment_list_type & segments() { return m_segments; }

    curve_list_type const & curves() const { return m_curves; }
    curve_list_type & curves() { return m_curves; }

    // The shape type serializes to its lower-case name (e.g. "triangle") so
    // the rendered JSON stays human-readable; this is an output-only view.
    MM_DECL_SERIALIZABLE(
        register_member("id", m_id);
        register_member("type", shape_type_name(m_type));
        register_member("bbox", m_bbox);
        register_member("segments", m_segments);
        register_member("curves", m_curves);)

private:

    int32_t m_id = 0;
    ShapeType m_type = ShapeType::DEAD;
    coords_type m_bbox; ///< Bounding box as [min_x, min_y, max_x, max_y].
    segment_list_type m_segments; ///< Each segment as [x0, y0, x1, y1].
    curve_list_type m_curves; ///< Each curve as four [x, y] points.

}; /* end class WorldShapeState */

/**
 * JSON view of the whole world: shapes plus the bare segments, bare curves,
 * and free points that also render but belong to no shape.
 *
 * @ingroup group_geometry
 */
class WorldState : public SerializableItem
{

public:

    using shape_list_type = std::vector<WorldShapeState>;
    using segment_list_type = std::vector<std::vector<double>>;
    using curve_list_type = std::vector<std::vector<std::vector<double>>>;
    using point_list_type = std::vector<std::vector<double>>;

    WorldState() = default;
    WorldState(WorldState const &) = default;
    WorldState(WorldState &&) = default;
    WorldState & operator=(WorldState const &) = default;
    WorldState & operator=(WorldState &&) = default;
    ~WorldState() override = default;

    shape_list_type const & shapes() const { return m_shapes; }
    shape_list_type & shapes() { return m_shapes; }

    segment_list_type const & segments() const { return m_segments; }
    segment_list_type & segments() { return m_segments; }

    curve_list_type const & curves() const { return m_curves; }
    curve_list_type & curves() { return m_curves; }

    point_list_type const & points() const { return m_points; }
    point_list_type & points() { return m_points; }

    std::optional<WorldDiagnostics> const & diagnostics() const { return m_diagnostics; }
    std::optional<WorldDiagnostics> & diagnostics() { return m_diagnostics; }

    MM_DECL_SERIALIZABLE(
        register_member("shapes", m_shapes);
        register_member("segments", m_segments);
        register_member("curves", m_curves);
        register_member("points", m_points);
        register_member("diagnostics", m_diagnostics);)

private:

    shape_list_type m_shapes;
    segment_list_type m_segments; ///< Bare segments.
    curve_list_type m_curves; ///< Bare curves.
    point_list_type m_points; ///< Free points, each [x, y].
    std::optional<WorldDiagnostics> m_diagnostics; ///< "diagnostics"-level facts; empty (omitted) at the basic level.

}; /* end class WorldState */

/**
 * Lightweight record mapping a shape ID to the segment and curve ranges
 * it owns in the world's pads. A shape may own segments only
 * (triangle, line, rectangle), curves only (ellipse, circle), or both.
 *
 * @ingroup group_geometry
 */
struct ShapeRecord
{
    using bbox_array_type = small_vector<double, 4>; ///< Four corners, (x, y) pairs.

    ShapeType type;
    size_t segment_offset; ///< First index in SegmentPad.
    size_t segment_count; ///< Number of segments this shape occupies.
    size_t curve_offset; ///< First index in CurvePad.
    size_t curve_count; ///< Number of cubic Beziers this shape occupies.

    /// OBB corner x's in world coordinates, ordered TL, TR, BR, BL.
    bbox_array_type obb_x;
    /// OBB corner y's in world coordinates, ordered TL, TR, BR, BL.
    bbox_array_type obb_y;
}; /* end struct ShapeRecord */

/**
 * Entry stored in the R-tree: shape ID + bounding box.
 *
 * @ingroup group_geometry
 */
template <typename T>
struct ShapeEntry
{
    int32_t shape_id;
    BoundBox3d<T> bbox;
    bool operator==(ShapeEntry const & other) const { return shape_id == other.shape_id; }
}; /* end struct ShapeEntry */

/**
 * RTreeValueOps specialization that gives the R-tree the bounding box of a
 * ShapeEntry.
 *
 * @ingroup group_geometry
 */
template <typename T>
struct RTreeValueOps<ShapeEntry<T>, BoundBox3d<T>>
{
    static BoundBox3d<T> calc_bound_box(ShapeEntry<T> const & entry) { return entry.bbox; }
}; /* end struct RTreeValueOps */

/**
 * Manage all geometry entities.
 *
 * @ingroup group_geometry
 */
template <typename T>
class World
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<World<T>>
{

private:

    class ctor_passkey
    {
    }; /* end class ctor_passkey */

public:

    using real_type = T;
    using value_type = T;
    using size_type = typename NumberBase<int32_t, T>::size_type;
    using point_type = Point3d<T>;
    using segment_type = Segment3d<T>;
    using bezier_type = Bezier3d<T>;
    using point_pad_type = PointPad<T>;
    using segment_pad_type = SegmentPad<T>;
    using curve_pad_type = CurvePad<T>;
    using bbox_type = BoundBox3d<T>;
    using rtree_type = RTree<ShapeEntry<T>, bbox_type>;

    using coord2_type = small_vector<value_type, 2>; ///< An (x, y) pair.
    using bbox_array_type = small_vector<value_type, 4>; ///< As [min_x, min_y, max_x, max_y].
    using obb_array_type = small_vector<value_type, 8>; ///< Four corners, (x, y) pairs.

    template <typename... Args>
    static std::shared_ptr<World<T>> construct(Args &&... args)
    {
        return std::make_shared<World<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    explicit World(ctor_passkey const &)
        : m_points(point_pad_type::construct(/* ndim */ 3))
        , m_segments(segment_pad_type::construct(/* ndim */ 3))
        , m_curves(curve_pad_type::construct(/* ndim */ 3))
        , m_rtree(std::make_unique<rtree_type>())
    {
    }

    World() = delete;
    World(World const &) = delete;
    World(World &&) = delete;
    World & operator=(World const &) = delete;
    World & operator=(World &&) = delete;
    ~World() = default;

    void add_point(point_type const & vertex)
    {
        m_points->append(vertex);
    }
    void add_point(value_type x, value_type y, value_type z)
    {
        add_point(point_type(x, y, z));
    }
    size_t npoint() const { return m_points->size(); }
    point_type point(size_t i) const { return m_points->get(i); }
    point_type point_at(size_t i) const
    {
        check_size(i, m_points->size(), "point");
        return m_points->get(i);
    }
    std::shared_ptr<point_pad_type> const & points() const { return m_points; }

    void add_segment(segment_type const & segment)
    {
        m_segments->append(segment);
        m_bare_segment_indices.push_back(m_segments->size() - 1);
    }
    void add_segment(point_type const & p0, point_type const & p1)
    {
        add_segment(segment_type(p0, p1));
    }
    size_t nsegment() const { return m_segments->size(); }
    segment_type segment(size_t i) const { return m_segments->get(i); }
    segment_type segment_at(size_t i) const
    {
        check_size(i, m_segments->size(), "segment");
        return m_segments->get(i);
    }
    std::shared_ptr<segment_pad_type> const & segments() { return m_segments; }

    void add_bezier(bezier_type const & bezier)
    {
        m_curves->append(bezier);
    }
    void add_bezier(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
    {
        m_curves->append(p0, p1, p2, p3);
    }
    size_t nbezier() const { return m_curves->size(); }
    bezier_type bezier(size_t i) const { return m_curves->get(i); }
    bezier_type bezier_at(size_t i) const
    {
        check_size(i, m_curves->size(), "bezier");
        return m_curves->get_at(i);
    }
    std::shared_ptr<curve_pad_type> const & curves() { return m_curves; }

    /**
     * Add a triangle by decomposing it into 3 segments in the pad.
     * Returns the shape ID for later reference.
     */
    int32_t add_triangle(T x0, T y0, T x1, T y1, T x2, T y2);

    /**
     * Add a line segment as a shape. One segment in the pad.
     */
    int32_t add_line(T x0, T y0, T x1, T y1);

    /**
     * Add an axis-aligned rectangle by decomposing it into 4 segments.
     * (x_min, y_min) is the lower-left corner; (x_max, y_max) the upper-right.
     */
    int32_t add_rectangle(T x_min, T y_min, T x_max, T y_max);

    /**
     * Specialization of add_rectangle with equal side lengths. Tagged SQUARE.
     */
    int32_t add_square(T x_min, T y_min, T size);

    /**
     * Add an axis-aligned ellipse as 4 cubic Bezier curves, one per quadrant,
     * using the standard k = 4*(sqrt(2) - 1)/3 circle approximation.
     * Ellipses own curves, not segments.
     */
    int32_t add_ellipse(T cx, T cy, T rx, T ry);

    /**
     * Specialization of add_ellipse with equal radii. Tagged CIRCLE.
     */
    int32_t add_circle(T cx, T cy, T r);

    /**
     * Add a cubic Bezier controlled by four points.
     */
    int32_t add_bezier_shape(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3);

    /**
     * Add a cubic Bezier from a bezier_type struct.
     */
    int32_t add_bezier_shape(bezier_type const & bezier);

    /**
     * Add an open chain of straight segments through the given vertices.
     * Needs at least two vertices; consecutive vertices form one segment each.
     */
    int32_t add_polyline(std::vector<std::array<T, 2>> const & vertices);

    /**
     * Add a closed chain of straight segments through the given vertices, with
     * a final segment back to the first. Needs at least three vertices.
     */
    int32_t add_polygon(std::vector<std::array<T, 2>> const & vertices);

    /**
     * Translate all segments and curves belonging to a shape by (dx, dy).
     */
    void translate_shape(int32_t shape_id, value_type dx, value_type dy);

    /**
     * Rotate all segments and curves belonging to a shape by `angle`
     * radians (counter-clockwise in world space) about the pivot (cx, cy).
     */
    void rotate_shape(int32_t shape_id, value_type angle, value_type cx, value_type cy);

    /**
     * Remove a shape from the R-tree and registry.
     * Segments and curves remain in their pads as dead data; use clear() to reclaim.
     */
    void remove_shape(int32_t shape_id);

    ShapeType shape_type_of(int32_t shape_id) const { return find_shape_or_throw(shape_id).type; }

    /// Segment ``i`` owned by a live shape (0-based within the shape).
    segment_type shape_segment(int32_t shape_id, uint32_t i) const
    {
        ShapeRecord const & rec = find_shape_or_throw(shape_id);
        check_size(i, rec.segment_count, "shape segment");
        return m_segments->get(rec.segment_offset + i);
    }

    /// Curve ``i`` owned by a live shape (0-based within the shape).
    bezier_type shape_curve(int32_t shape_id, uint32_t i) const
    {
        ShapeRecord const & rec = find_shape_or_throw(shape_id);
        check_size(i, rec.curve_count, "shape curve");
        return m_curves->get(rec.curve_offset + i);
    }

    /**
     * Undo the most recent change. A change is any shape operation: creation,
     * deletion, move, rotate, or a future attribute edit. A no-op when nothing
     * is undoable.
     */
    void undo();

    /// Redo the most recently undone change. A no-op when nothing is redoable.
    void redo();

    /**
     * Whether a change is available to undo. False while a compound is open,
     * matching how undo() refuses to run mid-gesture.
     */
    bool can_undo() const { return !m_in_operation && !m_undo_stack.empty(); }

    /**
     * Whether an undone change is available to redo. False while a compound is
     * open, matching how redo() refuses to run mid-gesture.
     */
    bool can_redo() const { return !m_in_operation && !m_redo_stack.empty(); }

    /**
     * Open a compound operation, which groups multiple shape changes into a
     * single undo step. A no-op if a compound is already open.
     */
    void begin_operation();

    /**
     * Close a compound operation, which groups multiple shape changes into a
     * single undo step. A no-op if a compound is not open.
     */
    void end_operation();

    size_t nshape() const { return m_nshape; }

    /**
     * True if `shape_id` refers to a live (non-DEAD) shape. Unlike the
     * accessors above this never throws, so callers can probe a stale id.
     */
    bool shape_is_live(int32_t shape_id) const
    {
        return shape_id >= 0 &&
               static_cast<size_t>(shape_id) < m_shape_registry.size() &&
               m_shape_registry[shape_id].type != ShapeType::DEAD;
    }

    /**
     * Axis-aligned bounding box of a live shape, as
     * [min_x, min_y, max_x, max_y].
     */
    bbox_array_type shape_bbox(int32_t shape_id) const
    {
        bbox_type const bb = compute_shape_bbox(find_shape_or_throw(shape_id));
        return {bb.min_x(), bb.min_y(), bb.max_x(), bb.max_y()};
    }

    /**
     * World position of a live shape's rotate-handle anchor (the oriented
     * bounding box's top-left corner), as [x, y]. Carried by every translate
     * and rotate, so it stays put relative to the shape.
     */
    coord2_type shape_handle(int32_t shape_id) const
    {
        ShapeRecord const & rec = find_shape_or_throw(shape_id);
        return {static_cast<value_type>(rec.obb_x[0]), static_cast<value_type>(rec.obb_y[0])};
    }

    /**
     * Oriented bounding box of a live shape: four world corners ordered
     * top-left, top-right, bottom-right, bottom-left.
     */
    obb_array_type shape_obb(int32_t shape_id) const
    {
        ShapeRecord const & rec = find_shape_or_throw(shape_id);
        obb_array_type out(8);
        for (size_t i = 0; i < 4; ++i)
        {
            out[2 * i] = static_cast<value_type>(rec.obb_x[i]);
            out[2 * i + 1] = static_cast<value_type>(rec.obb_y[i]);
        }
        return out;
    }

    /// Pick the live shape at world point (x, y).
    int32_t pick_shape(value_type x, value_type y, value_type tol) const;

    /**
     * Query the R-tree for shapes whose bounding box overlaps the viewport.
     */
    std::vector<int32_t> query_visible(T min_x, T min_y, T max_x, T max_y) const;

    /**
     * Collect all segments except those belonging to DEAD shapes.
     * Includes bare segments (added via add_segment) and live shape segments.
     */
    std::shared_ptr<segment_pad_type> collect_live_segments() const;

    /**
     * Collect all curves except those belonging to DEAD shapes.
     * Includes bare curves (added via add_bezier) and live shape curves.
     */
    std::shared_ptr<curve_pad_type> collect_live_curves() const;

    /**
     * Remove all geometry entities (points, segments, curves, shapes)
     * from the world. Rebuilds pads from scratch to reclaim memory.
     */
    void clear();

    /**
     * Describe the world state as a JSON-serializable object.
     */
    std::string describe_state(DescribeLevel level = DescribeLevel::BASIC) const;

private:

    /// 2D endpoints of segment i, as [x0, y0, x1, y1].
    std::vector<double> segment_coords(size_t i) const
    {
        return {m_segments->x0(i), m_segments->y0(i), m_segments->x1(i), m_segments->y1(i)};
    }

    /// 2D control points of curve i, as four [x, y] pairs.
    std::vector<std::vector<double>> curve_coords(size_t i) const
    {
        return {{m_curves->x0(i), m_curves->y0(i)},
                {m_curves->x1(i), m_curves->y1(i)},
                {m_curves->x2(i), m_curves->y2(i)},
                {m_curves->x3(i), m_curves->y3(i)}};
    }

    void check_size(size_t i, size_t s, char const * msg) const
    {
        if (i >= s)
        {
            throw std::out_of_range(std::format("World: ({}) i {} >= size {}", msg, i, s));
        }
    }

    bbox_type compute_shape_bbox(ShapeRecord const & rec) const;

    /**
     * Minimum distance from world point (px, py) to a shape's drawn geometry:
     * its segments and its curves (sampled along each cubic).
     */
    T shape_point_distance(ShapeRecord const & rec, T px, T py) const;

    /// Distance from point (px, py) to the segment (ax, ay)-(bx, by).
    static T point_segment_distance(T px, T py, T ax, T ay, T bx, T by);

    /**
     * Derived facts for the "diagnostics" level: proper crossings between
     * live segments and degenerate shapes/primitives.
     */
    WorldDiagnostics compute_diagnostics() const;

    /**
     * Owner id of each segment and curve: a live shape id, -1 for bare
     * geometry, or the dead sentinel for geometry whose shape was removed.
     */
    void compute_geometry_owners(small_vector<int32_t> & seg_owner, small_vector<int32_t> & curve_owner) const;

    /**
     * Append every proper crossing between two live segments, in ascending
     * index order so the output is deterministic.
     */
    void append_intersections(WorldDiagnostics & diag, small_vector<int32_t> const & seg_owner) const;

    /// Append the degeneracy (if any) of one live shape, dispatched by type.
    void append_shape_degeneracy(WorldDiagnostics & diag, int32_t sid, ShapeRecord const & rec) const;

    /**
     * Append degeneracies of bare segments and curves (those with no owning
     * shape), leaving live-shape degeneracies to append_shape_degeneracy.
     */
    void append_bare_degeneracies(
        WorldDiagnostics & diag, small_vector<int32_t> const & seg_owner, small_vector<int32_t> const & curve_owner) const;

    /**
     * Append a rectangle's four boundary segments and return the pad offset of
     * the first. Shared by add_rectangle and add_square.
     */
    size_t build_rectangle_segments(T x_min, T y_min, T x_max, T y_max);

    /**
     * Append an ellipse's four quadrant cubic Beziers and return the pad
     * offset of the first. Shared by add_ellipse and add_circle.
     */
    size_t build_ellipse_curves(T cx, T cy, T rx, T ry);

    /**
     * Register a new shape owning [segment_offset, segment_offset +
     * segment_count) in the segment pad and [curve_offset, curve_offset +
     * curve_count) in the curve pad. Pushes into the registry and R-tree.
     * Either range may be empty.
     */
    int32_t register_shape(ShapeType type,
                           size_t segment_offset,
                           size_t segment_count,
                           size_t curve_offset = 0,
                           size_t curve_count = 0);

    /**
     * Check if shape_id is valid and not DEAD.
     *
     * @throw std::out_of_range if shape_id is out of bounds or shape is DEAD.
     */
    void check_shape_id(int32_t shape_id) const;

    ShapeRecord const & find_shape_or_throw(int32_t shape_id) const
    {
        check_shape_id(shape_id);
        return m_shape_registry[shape_id];
    }

    ShapeRecord & find_shape_or_throw(int32_t shape_id)
    {
        check_shape_id(shape_id);
        return m_shape_registry[shape_id];
    }

    std::shared_ptr<point_pad_type> m_points;

    std::shared_ptr<segment_pad_type> m_segments;
    SimpleCollector<size_t> m_bare_segment_indices; ///< Indices of segments not owned by any shape.

    std::shared_ptr<curve_pad_type> m_curves;

    // TODO: Replace std::vector with a custom SoA container and BoundBoxPad
    // auxiliary class. Consider moving the registry into the R-tree.
    std::vector<ShapeRecord> m_shape_registry;

    size_t m_nshape = 0; ///< Count of live (non-DEAD) shapes.
    std::unique_ptr<rtree_type> m_rtree; ///< Spatial index for shapes for viewport query.

    /// The kind of an undoable shape change.
    enum class ShapeOp : uint8_t
    {
        CREATE,
        REMOVE,
        TRANSLATE,
        ROTATE,
    }; /* end enum class ShapeOp */

    /// Record of a single shape change, for undo/redo.
    struct ShapeOperationRecord
    {
        ShapeOp op = ShapeOp::CREATE;
        int32_t shape_id = -1;
        ShapeType type = ShapeType::DEAD; ///< Shape type to restore on undo or redo.

        value_type dx = 0; ///< Net x of a TRANSLATE.
        value_type dy = 0; ///< Net y of a TRANSLATE.
        value_type angle = 0; ///< Net angle of a ROTATE, in radians.
        value_type cx = 0; ///< Pivot x of a ROTATE.
        value_type cy = 0; ///< Pivot y of a ROTATE.
    }; /* end struct ShapeOperationRecord */

    /// Translate a shape's geometry in place and reindex it.
    void apply_translate(int32_t shape_id, value_type dx, value_type dy);
    /// Rotate a shape's geometry about (cx, cy) and reindex it.
    void apply_rotate(int32_t shape_id, value_type angle, value_type cx, value_type cy);
    /// Drop a live shape from the spatial index and the live count.
    void kill_shape(int32_t shape_id);
    /// Restore a DEAD shape slot as @p type and reindex it.
    void revive_shape(int32_t shape_id, ShapeType type);

    /// Record one change, merging it into the open compound when one is active.
    void record_op(ShapeOperationRecord const & rec);
    /// Push a finished record as a new undo step and drop the redo history.
    void commit_record(ShapeOperationRecord const & rec);
    /// Apply a record's inverse effect, undoing the change.
    void undo_record(ShapeOperationRecord const & rec);
    /// Apply a record's forward effect, redoing the change.
    void redo_record(ShapeOperationRecord const & rec);
    /// True for a move or rotate record whose net effect is nothing.
    static bool is_noop(ShapeOperationRecord const & rec);

    // TODO: The undo/redo history uses std::vector while the painter is a
    // prototype. Revisit the container (and a memory cap on history depth)
    // when the canvas editing model stabilizes.
    std::vector<ShapeOperationRecord> m_undo_stack; ///< Committed changes; back is next to undo.
    std::vector<ShapeOperationRecord> m_redo_stack; ///< Undone changes; back is next to redo.
    bool m_in_operation = false; ///< Whether a begin/end_operation compound is open.
    bool m_has_open = false; ///< Whether the open compound holds an uncommitted record.
    ShapeOperationRecord m_open_operation; ///< Merged record of the open compound.

}; /* end class World */

template <typename T>
int32_t World<T>::register_shape(ShapeType type,
                                 size_t segment_offset,
                                 size_t segment_count,
                                 size_t curve_offset,
                                 size_t curve_count)
{
    auto shape_id = static_cast<int32_t>(m_shape_registry.size());
    m_shape_registry.push_back(ShapeRecord{type, segment_offset, segment_count, curve_offset, curve_count}); // NOLINT(modernize-use-designated-initializers)
    ++m_nshape;
    bbox_type const bb = compute_shape_bbox(m_shape_registry[shape_id]);

    // Seed the oriented bounding box to the axis-aligned bbox corners.
    // TODO: store the OBB directly in ShapeRecord at construction.
    ShapeRecord & seeded = m_shape_registry[shape_id];
    seeded.obb_x = ShapeRecord::bbox_array_type{bb.min_x(), bb.max_x(), bb.max_x(), bb.min_x()};
    seeded.obb_y = ShapeRecord::bbox_array_type{bb.max_y(), bb.max_y(), bb.min_y(), bb.min_y()};
    m_rtree->insert(ShapeEntry<T>{shape_id, bb});

    // Record the creation so undo removes the shape and redo brings it back
    // with the same type.
    record_op({.op = ShapeOp::CREATE, .shape_id = shape_id, .type = type});

    return shape_id;
}

template <typename T>
int32_t World<T>::add_triangle(T x0, T y0, T x1, T y1, T x2, T y2)
{
    size_t const offset = m_segments->size();
    m_segments->append(point_type(x0, y0, 0), point_type(x1, y1, 0));
    m_segments->append(point_type(x1, y1, 0), point_type(x2, y2, 0));
    m_segments->append(point_type(x2, y2, 0), point_type(x0, y0, 0));
    return register_shape(ShapeType::TRIANGLE, offset, 3);
}

template <typename T>
int32_t World<T>::add_line(T x0, T y0, T x1, T y1)
{
    size_t const offset = m_segments->size();
    m_segments->append(point_type(x0, y0, 0), point_type(x1, y1, 0));
    return register_shape(ShapeType::LINE, offset, 1);
}

template <typename T>
size_t World<T>::build_rectangle_segments(T x_min, T y_min, T x_max, T y_max)
{
    size_t const offset = m_segments->size();
    point_type const p00(x_min, y_min, 0);
    point_type const p10(x_max, y_min, 0);
    point_type const p11(x_max, y_max, 0);
    point_type const p01(x_min, y_max, 0);
    m_segments->append(p00, p10);
    m_segments->append(p10, p11);
    m_segments->append(p11, p01);
    m_segments->append(p01, p00);
    return offset;
}

template <typename T>
int32_t World<T>::add_rectangle(T x_min, T y_min, T x_max, T y_max)
{
    return register_shape(ShapeType::RECTANGLE, build_rectangle_segments(x_min, y_min, x_max, y_max), 4);
}

template <typename T>
int32_t World<T>::add_square(T x_min, T y_min, T size)
{
    return register_shape(ShapeType::SQUARE, build_rectangle_segments(x_min, y_min, x_min + size, y_min + size), 4);
}

template <typename T>
size_t World<T>::build_ellipse_curves(T cx, T cy, T rx, T ry)
{
    // Standard cubic-Bezier circle approximation: control-point offset along
    // the tangent is k * radius, with k = 4*(sqrt(2) - 1)/3.
    T const k = T(4) * (std::sqrt(T(2)) - T(1)) / T(3);
    T const kx = k * rx;
    T const ky = k * ry;
    size_t const offset = m_curves->size();
    // Quadrant 1: (cx+rx, cy) -> (cx, cy+ry), sweeping counter-clockwise.
    m_curves->append(
        point_type(cx + rx, cy, 0),
        point_type(cx + rx, cy + ky, 0),
        point_type(cx + kx, cy + ry, 0),
        point_type(cx, cy + ry, 0));
    // Quadrant 2: (cx, cy+ry) -> (cx-rx, cy).
    m_curves->append(
        point_type(cx, cy + ry, 0),
        point_type(cx - kx, cy + ry, 0),
        point_type(cx - rx, cy + ky, 0),
        point_type(cx - rx, cy, 0));
    // Quadrant 3: (cx-rx, cy) -> (cx, cy-ry).
    m_curves->append(
        point_type(cx - rx, cy, 0),
        point_type(cx - rx, cy - ky, 0),
        point_type(cx - kx, cy - ry, 0),
        point_type(cx, cy - ry, 0));
    // Quadrant 4: (cx, cy-ry) -> (cx+rx, cy).
    m_curves->append(
        point_type(cx, cy - ry, 0),
        point_type(cx + kx, cy - ry, 0),
        point_type(cx + rx, cy - ky, 0),
        point_type(cx + rx, cy, 0));
    return offset;
}

template <typename T>
int32_t World<T>::add_ellipse(T cx, T cy, T rx, T ry)
{
    return register_shape(ShapeType::ELLIPSE, /*seg_off*/ 0, /*seg_cnt*/ 0, build_ellipse_curves(cx, cy, rx, ry), 4);
}

template <typename T>
int32_t World<T>::add_circle(T cx, T cy, T r)
{
    return register_shape(ShapeType::CIRCLE, /*seg_off*/ 0, /*seg_cnt*/ 0, build_ellipse_curves(cx, cy, r, r), 4);
}

template <typename T>
int32_t World<T>::add_bezier_shape(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
{
    size_t const offset = m_curves->size();
    m_curves->append(p0, p1, p2, p3);
    return register_shape(ShapeType::BEZIER, /*seg_off*/ 0, /*seg_cnt*/ 0, offset, 1);
}

template <typename T>
int32_t World<T>::add_bezier_shape(bezier_type const & bezier)
{
    size_t const offset = m_curves->size();
    m_curves->append(bezier);
    return register_shape(ShapeType::BEZIER, /*seg_off*/ 0, /*seg_cnt*/ 0, offset, 1);
}

template <typename T>
int32_t World<T>::add_polyline(std::vector<std::array<T, 2>> const & vertices)
{
    if (vertices.size() < 2)
    {
        throw std::invalid_argument("World: add_polyline needs at least 2 vertices");
    }
    size_t const offset = m_segments->size();
    for (size_t i = 0; i + 1 < vertices.size(); ++i)
    {
        m_segments->append(
            point_type(vertices[i][0], vertices[i][1], 0),
            point_type(vertices[i + 1][0], vertices[i + 1][1], 0));
    }
    return register_shape(ShapeType::POLYLINE, offset, vertices.size() - 1);
}

template <typename T>
int32_t World<T>::add_polygon(std::vector<std::array<T, 2>> const & vertices)
{
    if (vertices.size() < 3)
    {
        throw std::invalid_argument("World: add_polygon needs at least 3 vertices");
    }
    size_t const offset = m_segments->size();
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        size_t const j = (i + 1) % vertices.size();
        m_segments->append(
            point_type(vertices[i][0], vertices[i][1], 0),
            point_type(vertices[j][0], vertices[j][1], 0));
    }
    return register_shape(ShapeType::POLYGON, offset, vertices.size());
}

template <typename T>
void World<T>::apply_translate(int32_t shape_id, value_type dx, value_type dy)
{
    ShapeRecord & rec = m_shape_registry[static_cast<size_t>(shape_id)];
    // Remove old entry from R-tree before modifying segments/curves.
    m_rtree->remove(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
    for (uint32_t i = 0; i < rec.segment_count; ++i)
    {
        size_t const idx = rec.segment_offset + i;
        m_segments->x0(idx) += dx;
        m_segments->y0(idx) += dy;
        m_segments->x1(idx) += dx;
        m_segments->y1(idx) += dy;
    }
    for (uint32_t i = 0; i < rec.curve_count; ++i)
    {
        size_t const idx = rec.curve_offset + i;
        m_curves->x0(idx) += dx;
        m_curves->y0(idx) += dy;
        m_curves->x1(idx) += dx;
        m_curves->y1(idx) += dy;
        m_curves->x2(idx) += dx;
        m_curves->y2(idx) += dy;
        m_curves->x3(idx) += dx;
        m_curves->y3(idx) += dy;
    }
    // The oriented bounding box rides along with the shape.
    for (uint32_t i = 0; i < 4; ++i)
    {
        rec.obb_x[i] += dx;
        rec.obb_y[i] += dy;
    }
    // Reinsert with updated bounding box.
    m_rtree->insert(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
}

template <typename T>
void World<T>::translate_shape(int32_t shape_id, value_type dx, value_type dy)
{
    find_shape_or_throw(shape_id);
    // A zero move changes nothing, so do not churn the R-tree or record an
    // empty undo step (a drag's first event often lands on the press point).
    if (dx == value_type(0) && dy == value_type(0))
    {
        return;
    }
    apply_translate(shape_id, dx, dy);
    record_op({.op = ShapeOp::TRANSLATE, .shape_id = shape_id, .dx = dx, .dy = dy});
}

template <typename T>
void World<T>::apply_rotate(int32_t shape_id, value_type angle, value_type cx, value_type cy)
{
    ShapeRecord & rec = m_shape_registry[static_cast<size_t>(shape_id)];

    // Remove old entry from R-tree before modifying segments/curves.
    m_rtree->remove(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});

    T const cos_a = std::cos(angle);
    T const sin_a = std::sin(angle);

    // Rotate one point about (cx, cy) in place; reads both coordinates
    // before writing either, so x and y rotate from the same origin.
    auto rotate = [&](T & x, T & y)
    {
        T const dx = x - cx;
        T const dy = y - cy;
        x = cx + cos_a * dx - sin_a * dy;
        y = cy + sin_a * dx + cos_a * dy;
    };

    for (uint32_t i = 0; i < rec.segment_count; ++i)
    {
        size_t const idx = rec.segment_offset + i;
        rotate(m_segments->x0(idx), m_segments->y0(idx));
        rotate(m_segments->x1(idx), m_segments->y1(idx));
    }
    for (uint32_t i = 0; i < rec.curve_count; ++i)
    {
        size_t const idx = rec.curve_offset + i;
        rotate(m_curves->x0(idx), m_curves->y0(idx));
        rotate(m_curves->x1(idx), m_curves->y1(idx));
        rotate(m_curves->x2(idx), m_curves->y2(idx));
        rotate(m_curves->x3(idx), m_curves->y3(idx));
    }

    // Rotate the oriented bounding box about the same pivot so it stays
    // fixed relative to the shape. Done in `double` to match its storage.
    auto const dcos = static_cast<double>(cos_a);
    auto const dsin = static_cast<double>(sin_a);
    for (uint32_t i = 0; i < 4; ++i)
    {
        double const ox = rec.obb_x[i] - cx;
        double const oy = rec.obb_y[i] - cy;
        rec.obb_x[i] = cx + dcos * ox - dsin * oy;
        rec.obb_y[i] = cy + dsin * ox + dcos * oy;
    }
    // Reinsert with updated bounding box.
    m_rtree->insert(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
}

template <typename T>
void World<T>::rotate_shape(int32_t shape_id, value_type angle, value_type cx, value_type cy)
{
    find_shape_or_throw(shape_id);
    // A zero rotation changes nothing; skip it like a zero translate so a
    // drag's first event does not record an empty undo step.
    if (angle == value_type(0))
    {
        return;
    }
    apply_rotate(shape_id, angle, cx, cy);
    record_op({.op = ShapeOp::ROTATE, .shape_id = shape_id, .angle = angle, .cx = cx, .cy = cy});
}

template <typename T>
T World<T>::point_segment_distance(T px, T py, T ax, T ay, T bx, T by)
{
    T const dx = bx - ax;
    T const dy = by - ay;
    T const len2 = dx * dx + dy * dy;
    if (len2 <= T(0)) // degenerate segment: distance to the point
    {
        return std::hypot(px - ax, py - ay);
    }
    T t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t = std::max(T(0), std::min(T(1), t)); // clamp to the segment
    return std::hypot(px - (ax + t * dx), py - (ay + t * dy));
}

template <typename T>
T World<T>::shape_point_distance(ShapeRecord const & rec, T px, T py) const
{
    T best = std::numeric_limits<T>::max();
    for (uint32_t i = 0; i < rec.segment_count; ++i)
    {
        size_t const idx = rec.segment_offset + i;
        best = std::min(best, point_segment_distance(px, py, m_segments->x0(idx), m_segments->y0(idx), m_segments->x1(idx), m_segments->y1(idx)));
    }
    // Flatten each cubic into short chords and measure the point against them.
    constexpr uint32_t SAMPLES = 16;
    for (uint32_t i = 0; i < rec.curve_count; ++i)
    {
        size_t const idx = rec.curve_offset + i;
        T const x0 = m_curves->x0(idx), y0 = m_curves->y0(idx);
        T const x1 = m_curves->x1(idx), y1 = m_curves->y1(idx);
        T const x2 = m_curves->x2(idx), y2 = m_curves->y2(idx);
        T const x3 = m_curves->x3(idx), y3 = m_curves->y3(idx);
        T prev_x = x0, prev_y = y0;
        for (uint32_t s = 1; s <= SAMPLES; ++s)
        {
            T const u = static_cast<T>(s) / static_cast<T>(SAMPLES);
            T const v = T(1) - u;
            T const b0 = v * v * v, b1 = T(3) * v * v * u, b2 = T(3) * v * u * u, b3 = u * u * u;
            T const cx = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3;
            T const cy = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3;
            best = std::min(best, point_segment_distance(px, py, prev_x, prev_y, cx, cy));
            prev_x = cx;
            prev_y = cy;
        }
    }
    return best;
}

template <typename T>
int32_t World<T>::pick_shape(value_type x, value_type y, value_type tol) const
{
    T const t = tol > T(0) ? tol : T(0);
    std::vector<int32_t> const candidates = query_visible(x - t, y - t, x + t, y + t);
    int32_t best = -1;
    T best_area = T(0);
    for (int32_t const id : candidates)
    {
        ShapeRecord const & rec = m_shape_registry[static_cast<size_t>(id)];
        if (rec.type == ShapeType::DEAD)
        {
            continue;
        }
        bbox_type const bb = compute_shape_bbox(rec);
        bool const inside = x >= bb.min_x() && x <= bb.max_x() && y >= bb.min_y() && y <= bb.max_y();
        if (!inside && shape_point_distance(rec, x, y) > t)
        {
            continue;
        }
        // Prefer the smallest box (inner shapes), ties to the newest shape.
        T const area = (bb.max_x() - bb.min_x()) * (bb.max_y() - bb.min_y());
        if (best == -1 || area < best_area || (area == best_area && id > best))
        {
            best = id;
            best_area = area;
        }
    }
    return best;
}

template <typename T>
void World<T>::kill_shape(int32_t shape_id)
{
    ShapeRecord & rec = m_shape_registry[static_cast<size_t>(shape_id)];
    m_rtree->remove(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
    rec.type = ShapeType::DEAD;
    --m_nshape;
}

template <typename T>
void World<T>::revive_shape(int32_t shape_id, ShapeType type)
{
    // Geometry stays in the pads while a shape is DEAD, so reviving only
    // restores the type and re-indexes the shape at its current geometry.
    ShapeRecord & rec = m_shape_registry[static_cast<size_t>(shape_id)];
    rec.type = type;
    ++m_nshape;
    m_rtree->insert(ShapeEntry<T>{shape_id, compute_shape_bbox(rec)});
}

template <typename T>
void World<T>::remove_shape(int32_t shape_id)
{
    ShapeRecord const & rec = find_shape_or_throw(shape_id);
    ShapeType const type = rec.type;
    kill_shape(shape_id);
    record_op({.op = ShapeOp::REMOVE, .shape_id = shape_id, .type = type});
}

template <typename T>
bool World<T>::is_noop(ShapeOperationRecord const & rec)
{
    return (rec.op == ShapeOp::TRANSLATE && rec.dx == value_type(0) && rec.dy == value_type(0)) ||
           (rec.op == ShapeOp::ROTATE && rec.angle == value_type(0));
}

template <typename T>
void World<T>::commit_record(ShapeOperationRecord const & rec)
{
    if (is_noop(rec))
    {
        return;
    }
    m_undo_stack.push_back(rec);
    m_redo_stack.clear();
}

template <typename T>
void World<T>::record_op(ShapeOperationRecord const & rec)
{
    if (m_in_operation)
    {
        // Merge the incremental edits of one gesture (same shape, operation,
        // and pivot) into a single net record, so a drag is one undo step.
        // Translations add; rotations about a fixed pivot add their angles.
        if (m_has_open && m_open_operation.shape_id == rec.shape_id &&
            m_open_operation.op == rec.op && m_open_operation.cx == rec.cx &&
            m_open_operation.cy == rec.cy)
        {
            m_open_operation.dx += rec.dx;
            m_open_operation.dy += rec.dy;
            m_open_operation.angle += rec.angle;
        }
        else
        {
            if (m_has_open)
            {
                commit_record(m_open_operation);
            }
            m_open_operation = rec;
            m_has_open = true;
        }
        return;
    }
    commit_record(rec);
}

template <typename T>
void World<T>::undo_record(ShapeOperationRecord const & rec)
{
    // apply_translate, apply_rotate, kill_shape, and revive_shape are the
    // internal mutation primitives: they change geometry or the registry and
    // reindex, but never touch the undo history. The public translate_shape,
    // rotate_shape, and remove_shape wrap these and record an undo step.

    switch (rec.op)
    {
    case ShapeOp::CREATE: kill_shape(rec.shape_id); break;
    case ShapeOp::REMOVE: revive_shape(rec.shape_id, rec.type); break;
    case ShapeOp::TRANSLATE: apply_translate(rec.shape_id, -rec.dx, -rec.dy); break;
    case ShapeOp::ROTATE: apply_rotate(rec.shape_id, -rec.angle, rec.cx, rec.cy); break;
    }
}

template <typename T>
void World<T>::redo_record(ShapeOperationRecord const & rec)
{
    switch (rec.op)
    {
    case ShapeOp::CREATE: revive_shape(rec.shape_id, rec.type); break;
    case ShapeOp::REMOVE: kill_shape(rec.shape_id); break;
    case ShapeOp::TRANSLATE: apply_translate(rec.shape_id, rec.dx, rec.dy); break;
    case ShapeOp::ROTATE: apply_rotate(rec.shape_id, rec.angle, rec.cx, rec.cy); break;
    }
}

template <typename T>
void World<T>::begin_operation()
{
    // A dangling compound (a gesture abandoned without end_operation) is
    // finalized so the stacks never carry an unterminated step.
    if (m_in_operation)
    {
        end_operation();
    }
    m_in_operation = true;
    m_has_open = false;
}

template <typename T>
void World<T>::end_operation()
{
    if (!m_in_operation)
    {
        return;
    }
    m_in_operation = false;
    if (m_has_open)
    {
        commit_record(m_open_operation);
        m_has_open = false;
    }
}

template <typename T>
void World<T>::undo()
{
    // Undo while a compound operation is open (e.g. an Undo shortcut fired
    // mid-drag) would unwind state from under the uncommitted gesture and
    // corrupt the history; ignore it until end_operation() closes the gesture.
    if (m_in_operation || m_undo_stack.empty())
    {
        return;
    }
    ShapeOperationRecord const rec = m_undo_stack.back();
    m_undo_stack.pop_back();
    undo_record(rec);
    m_redo_stack.push_back(rec);
}

template <typename T>
void World<T>::redo()
{
    // See undo(): redo is likewise ignored while a compound operation is open.
    if (m_in_operation || m_redo_stack.empty())
    {
        return;
    }
    ShapeOperationRecord const rec = m_redo_stack.back();
    m_redo_stack.pop_back();
    redo_record(rec);
    m_undo_stack.push_back(rec);
}

template <typename T>
std::vector<int32_t> World<T>::query_visible(T min_x, T min_y, T max_x, T max_y) const
{
    bbox_type const viewport(min_x, min_y, T(0), max_x, max_y, T(0));
    std::vector<ShapeEntry<T>> hits;
    m_rtree->search(viewport, hits);
    std::vector<int32_t> ids;
    ids.reserve(hits.size());
    for (auto const & entry : hits)
    {
        ids.push_back(entry.shape_id);
    }
    return ids;
}

template <typename T>
std::shared_ptr<typename World<T>::segment_pad_type> World<T>::collect_live_segments() const
{
    // Mark segment indices owned by DEAD shapes.
    small_vector<bool> dead(m_segments->size(), false);
    for (auto const & rec : m_shape_registry)
    {
        if (rec.type != ShapeType::DEAD)
        {
            continue;
        }
        for (uint32_t i = 0; i < rec.segment_count; ++i)
        {
            dead[rec.segment_offset + i] = true;
        }
    }
    auto result = segment_pad_type::construct(/* ndim */ 3);
    for (size_t i = 0; i < m_segments->size(); ++i)
    {
        if (!dead[i])
        {
            // includes both bare segments and segments of live shapes
            result->append(m_segments->get(i));
        }
    }
    return result;
}

template <typename T>
std::shared_ptr<typename World<T>::curve_pad_type> World<T>::collect_live_curves() const
{
    // Mark curve indices owned by DEAD shapes.
    small_vector<bool> dead(m_curves->size(), false);
    for (auto const & rec : m_shape_registry)
    {
        if (rec.type != ShapeType::DEAD)
        {
            continue;
        }
        for (uint32_t i = 0; i < rec.curve_count; ++i)
        {
            dead[rec.curve_offset + i] = true;
        }
    }
    auto result = curve_pad_type::construct(/* ndim */ 3);
    for (size_t i = 0; i < m_curves->size(); ++i)
    {
        if (!dead[i])
        {
            // includes both bare curves and curves of live shapes
            result->append(m_curves->get(i));
        }
    }
    return result;
}

template <typename T>
void World<T>::clear()
{
    m_points = point_pad_type::construct(/* ndim */ 3);
    m_segments = segment_pad_type::construct(/* ndim */ 3);
    m_curves = curve_pad_type::construct(/* ndim */ 3);
    m_bare_segment_indices.clear();
    m_shape_registry.clear();
    m_nshape = 0;
    m_rtree = std::make_unique<rtree_type>();
    m_undo_stack.clear();
    m_redo_stack.clear();
    m_in_operation = false;
    m_has_open = false;
}

template <typename T>
typename World<T>::bbox_type World<T>::compute_shape_bbox(ShapeRecord const & rec) const
{
    T mn_x = std::numeric_limits<T>::max();
    T mn_y = std::numeric_limits<T>::max();
    T mx_x = std::numeric_limits<T>::lowest();
    T mx_y = std::numeric_limits<T>::lowest();
    for (uint32_t i = 0; i < rec.segment_count; ++i)
    {
        size_t const idx = rec.segment_offset + i;
        mn_x = std::min({mn_x, m_segments->x0(idx), m_segments->x1(idx)});
        mn_y = std::min({mn_y, m_segments->y0(idx), m_segments->y1(idx)});
        mx_x = std::max({mx_x, m_segments->x0(idx), m_segments->x1(idx)});
        mx_y = std::max({mx_y, m_segments->y0(idx), m_segments->y1(idx)});
    }
    // Bound curves by the convex hull of their cubic control points. For
    // 4-quadrant cubic-Bezier ellipses this is exactly the ellipse bbox.
    for (uint32_t i = 0; i < rec.curve_count; ++i)
    {
        size_t const idx = rec.curve_offset + i;
        mn_x = std::min({mn_x, m_curves->x0(idx), m_curves->x1(idx), m_curves->x2(idx), m_curves->x3(idx)});
        mn_y = std::min({mn_y, m_curves->y0(idx), m_curves->y1(idx), m_curves->y2(idx), m_curves->y3(idx)});
        mx_x = std::max({mx_x, m_curves->x0(idx), m_curves->x1(idx), m_curves->x2(idx), m_curves->x3(idx)});
        mx_y = std::max({mx_y, m_curves->y0(idx), m_curves->y1(idx), m_curves->y2(idx), m_curves->y3(idx)});
    }
    return bbox_type(mn_x, mn_y, T(0), mx_x, mx_y, T(0));
}

template <typename T>
void World<T>::check_shape_id(int32_t shape_id) const
{
    if (shape_id < 0 ||
        static_cast<size_t>(shape_id) >= m_shape_registry.size())
    {
        throw std::out_of_range(
            std::format("World: shape_id {} not found", shape_id));
    }
    if (m_shape_registry[shape_id].type == ShapeType::DEAD)
    {
        throw std::invalid_argument(
            std::format("World: shape_id {} is DEAD", shape_id));
    }
}

template <typename T>
std::string World<T>::describe_state(DescribeLevel level) const
{
    // Find the segments and curves owned by live shapes
    small_vector<bool> segment_owned(m_segments->size(), false);
    small_vector<bool> curve_owned(m_curves->size(), false);

    WorldState state;
    for (size_t sid = 0; sid < m_shape_registry.size(); ++sid)
    {
        ShapeRecord const & rec = m_shape_registry[sid];
        for (uint32_t i = 0; i < rec.segment_count; ++i)
        {
            segment_owned[rec.segment_offset + i] = true;
        }
        for (uint32_t i = 0; i < rec.curve_count; ++i)
        {
            curve_owned[rec.curve_offset + i] = true;
        }
        if (rec.type == ShapeType::DEAD)
        {
            continue;
        }
        bbox_type const bb = compute_shape_bbox(rec);
        WorldShapeState shape;
        shape.id() = static_cast<int32_t>(sid);
        shape.type() = rec.type;
        shape.bbox() = {bb.min_x(), bb.min_y(), bb.max_x(), bb.max_y()};
        for (uint32_t i = 0; i < rec.segment_count; ++i)
        {
            shape.segments().push_back(segment_coords(rec.segment_offset + i));
        }
        for (uint32_t i = 0; i < rec.curve_count; ++i)
        {
            shape.curves().push_back(curve_coords(rec.curve_offset + i));
        }
        state.shapes().push_back(std::move(shape));
    }

    for (size_t i = 0; i < m_segments->size(); ++i)
    {
        if (!segment_owned[i])
        {
            // already exclude DEAD shape segments, but include bare segments
            state.segments().push_back(segment_coords(i));
        }
    }
    for (size_t i = 0; i < m_curves->size(); ++i)
    {
        if (!curve_owned[i])
        {
            state.curves().push_back(curve_coords(i));
        }
    }
    for (size_t i = 0; i < m_points->size(); ++i)
    {
        point_type const p = m_points->get(i);
        state.points().push_back({p.x(), p.y()});
    }

    if (level == DescribeLevel::DIAGNOSTICS)
    {
        state.diagnostics() = compute_diagnostics();
    }
    return state.to_json();
}

template <typename T>
WorldDiagnostics World<T>::compute_diagnostics() const
{
    small_vector<int32_t> seg_owner(m_segments->size(), -1);
    small_vector<int32_t> curve_owner(m_curves->size(), -1);
    compute_geometry_owners(seg_owner, curve_owner);

    WorldDiagnostics diag;
    append_intersections(diag, seg_owner);

    for (size_t sid = 0; sid < m_shape_registry.size(); ++sid)
    {
        ShapeRecord const & rec = m_shape_registry[sid];
        if (rec.type != ShapeType::DEAD)
        {
            append_shape_degeneracy(diag, static_cast<int32_t>(sid), rec);
        }
    }
    append_bare_degeneracies(diag, seg_owner, curve_owner);
    return diag;
}

template <typename T>
void World<T>::compute_geometry_owners(small_vector<int32_t> & seg_owner, small_vector<int32_t> & curve_owner) const
{
    // The dead sentinel marks geometry whose shape was removed, so callers can
    // exclude it; bare geometry keeps the -1 the caller initialized it with.
    constexpr int32_t dead_owner = std::numeric_limits<int32_t>::min();
    for (size_t sid = 0; sid < m_shape_registry.size(); ++sid)
    {
        ShapeRecord const & rec = m_shape_registry[sid];
        int32_t const owner = (rec.type == ShapeType::DEAD) ? dead_owner : static_cast<int32_t>(sid);
        for (uint32_t i = 0; i < rec.segment_count; ++i)
        {
            seg_owner[rec.segment_offset + i] = owner;
        }
        for (uint32_t i = 0; i < rec.curve_count; ++i)
        {
            curve_owner[rec.curve_offset + i] = owner;
        }
    }
}

template <typename T>
void World<T>::append_intersections(WorldDiagnostics & diag, small_vector<int32_t> const & seg_owner) const
{
    constexpr int32_t dead_owner = std::numeric_limits<int32_t>::min();
    for (size_t i = 0; i < m_segments->size(); ++i)
    {
        if (seg_owner[i] == dead_owner)
        {
            continue;
        }
        for (size_t j = i + 1; j < m_segments->size(); ++j)
        {
            if (seg_owner[j] == dead_owner)
            {
                continue;
            }
            auto const hit = detail::segment_proper_intersection(
                m_segments->x0(i), m_segments->y0(i), m_segments->x1(i), m_segments->y1(i), m_segments->x0(j), m_segments->y0(j), m_segments->x1(j), m_segments->y1(j));
            if (hit)
            {
                diag.add_intersection(seg_owner[i], seg_owner[j], (*hit)[0], (*hit)[1]);
            }
        }
    }
}

template <typename T>
void World<T>::append_bare_degeneracies(
    WorldDiagnostics & diag, small_vector<int32_t> const & seg_owner, small_vector<int32_t> const & curve_owner) const
{
    for (size_t i = 0; i < m_segments->size(); ++i)
    {
        if (seg_owner[i] == -1 && detail::is_zero_length(m_segments->x0(i), m_segments->y0(i), m_segments->x1(i), m_segments->y1(i)))
        {
            diag.add_degeneracy(-1, "segment", "zero-length");
        }
    }
    for (size_t i = 0; i < m_curves->size(); ++i)
    {
        if (curve_owner[i] == -1 && detail::is_coincident_controls(m_curves->x0(i), m_curves->y0(i), m_curves->x1(i), m_curves->y1(i), m_curves->x2(i), m_curves->y2(i), m_curves->x3(i), m_curves->y3(i)))
        {
            diag.add_degeneracy(-1, "bezier", "coincident-controls");
        }
    }
}

template <typename T>
void World<T>::append_shape_degeneracy(WorldDiagnostics & diag, int32_t sid, ShapeRecord const & rec) const
{
    switch (rec.type)
    {
    case ShapeType::LINE:
    {
        size_t const idx = rec.segment_offset;
        if (detail::is_zero_length(m_segments->x0(idx), m_segments->y0(idx), m_segments->x1(idx), m_segments->y1(idx)))
        {
            diag.add_degeneracy(sid, "line", "zero-length");
        }
        break;
    }
    case ShapeType::TRIANGLE:
    {
        // Vertices A, B, C are the first two segments' endpoints.
        size_t const s0 = rec.segment_offset;
        size_t const s1 = rec.segment_offset + 1;
        if (detail::is_collinear(m_segments->x0(s0), m_segments->y0(s0), m_segments->x1(s0), m_segments->y1(s0), m_segments->x1(s1), m_segments->y1(s1)))
        {
            diag.add_degeneracy(sid, "triangle", "collinear");
        }
        break;
    }
    case ShapeType::RECTANGLE:
    case ShapeType::SQUARE:
    {
        bbox_type const bb = compute_shape_bbox(rec);
        if (detail::is_zero_extent(bb.min_x(), bb.min_y(), bb.max_x(), bb.max_y()))
        {
            diag.add_degeneracy(sid, shape_type_name(rec.type), "zero-area");
        }
        break;
    }
    case ShapeType::ELLIPSE:
    case ShapeType::CIRCLE:
    {
        bbox_type const bb = compute_shape_bbox(rec);
        if (detail::is_zero_extent(bb.min_x(), bb.min_y(), bb.max_x(), bb.max_y()))
        {
            diag.add_degeneracy(sid, shape_type_name(rec.type), "zero-radius");
        }
        break;
    }
    case ShapeType::BEZIER:
    {
        size_t const idx = rec.curve_offset;
        if (detail::is_coincident_controls(m_curves->x0(idx), m_curves->y0(idx), m_curves->x1(idx), m_curves->y1(idx), m_curves->x2(idx), m_curves->y2(idx), m_curves->x3(idx), m_curves->y3(idx)))
        {
            diag.add_degeneracy(sid, "bezier", "coincident-controls");
        }
        break;
    }
    default:
        break;
    }
}

using WorldFp32 = World<float>;
using WorldFp64 = World<double>;

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
