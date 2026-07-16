# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The command vocabulary for driving a ``World`` from an AI agent.

Each command is one ``Command`` subclass registered in ``DRAW``; the JSON
Schema documents and validators are derived from them (see
:mod:`solvcon.agent._command`). Only the ``World``-specific fragments and
behavior live here, so a mesh or pilot family declares commands the same way.
``render_png`` returns a transport-ready image (base64 PNG), never raw bytes,
so a result is JSON-serializable across any transport.

Coordinates are world coordinates in the ``World``'s native units, math
convention with +Y pointing up.
"""

import json
import base64

from .. import _command as _cmd


# Shared JSON Schema fragments, used by reference; treat as immutable.
NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
INTEGER = {"type": "integer"}
POSITIVE_INT = {"type": "integer", "exclusiveMinimum": 0}
BOOLEAN = {"type": "boolean"}
STRING = {"type": "string"}


def _num(description):
    return {**NUMBER, "description": description}


def _pos(description):
    return {**POSITIVE, "description": description}


def _int(description):
    return {**INTEGER, "description": description}


def _point(description):
    return {"type": "array", "items": {"type": "number"},
            "minItems": 2, "maxItems": 3, "description": description}


# Mirrors ViewTransform2dFp64, math convention +Y up.
VIEW = {"type": "object",
        "properties": {"pan_x": NUMBER, "pan_y": NUMBER, "zoom": POSITIVE},
        "additionalProperties": False}

_BBOX = {"type": "array", "items": NUMBER, "minItems": 4, "maxItems": 4,
         "description": "Axis-aligned bounds [min_x, min_y, max_x, max_y]."}
_SEG_LIST = {"type": "array",
             "items": {"type": "array", "items": NUMBER,
                       "minItems": 4, "maxItems": 4},
             "description": "Line segments, each [x0, y0, x1, y1]."}
_CURVE_LIST = {"type": "array",
               "items": {"type": "array",
                         "items": {"type": "array", "items": NUMBER,
                                   "minItems": 2, "maxItems": 2},
                         "minItems": 4, "maxItems": 4},
               "description": "Cubic Beziers, each four [x, y] points."}
_POINT_LIST = {"type": "array",
               "items": {"type": "array", "items": NUMBER,
                         "minItems": 2, "maxItems": 2},
               "description": "Free points, each [x, y]."}

# Mirrors WorldShapeState in cpp/solvcon/universe/World.hpp.
SHAPE = {"type": "object",
         "description": "One shape: id, type, bounds, and 2D geometry.",
         "properties": {
             "id": _int("Stable id assigned when the shape was created."),
             "type": {**STRING,
                      "description": "Lower-case shape type name."},
             "bbox": _BBOX,
             "segments": _SEG_LIST,
             "curves": _CURVE_LIST},
         "required": ["id", "type", "bbox", "segments", "curves"],
         "additionalProperties": False}

# Mirrors WorldState in cpp/solvcon/universe/World.hpp.
STATE = {"type": "object",
         "description": "The whole visible world: shapes plus bare geometry.",
         "properties": {
             "shapes": {"type": "array", "items": SHAPE,
                        "description": "Every live shape."},
             "segments": _SEG_LIST,
             "curves": _CURVE_LIST,
             "points": _POINT_LIST},
         "required": ["shapes", "segments", "curves", "points"],
         "additionalProperties": False}

IMAGE = {"type": "object",
         "description": "A rendered raster image, transport-ready.",
         "properties": {
             "data": {"type": "string", "contentEncoding": "base64",
                      "contentMediaType": "image/png",
                      "description": "Base64-encoded PNG bytes."},
             "mime_type": {"const": "image/png",
                           "description": "Media type of the encoded image."},
             "width": _int("Image width in pixels."),
             "height": _int("Image height in pixels.")},
         "required": ["data", "mime_type", "width", "height"],
         "additionalProperties": False}


def _world_point(coords):
    import solvcon
    z = coords[2] if len(coords) > 2 else 0.0
    return solvcon.Point3dFp64(coords[0], coords[1], z)


def _require_live(world, shape_id):
    # Preflight so every by-id command fails the one clean way ``get_shape``
    # does, rather than leaking a C++-flavored IndexError/ValueError.
    if not world.shape_is_live(shape_id):
        raise _cmd.CommandError(f"no live shape with id {shape_id}")


DRAW = _cmd.CommandSet("Agent Draw command",
                       "Any single command in the Agent Draw schema.")


@DRAW.register
class AddPoint(_cmd.Command):
    op = "add_point"
    category = "create"
    summary = "Add a free point at world (x, y, z); z is dropped in 2D."
    arguments = {"x": _num("World x of the point."),
                 "y": _num("World y of the point (+Y up)."),
                 "z": {**NUMBER, "default": 0.0,
                       "description": "World z; ignored in 2D."}}
    optional = ("z",)
    returns = {"npoint": _int("Total free points after the add.")}

    def apply(self, world, args, ctx):
        world.add_point(args["x"], args["y"], args["z"])
        return {"npoint": world.npoint}


@DRAW.register
class AddSegment(_cmd.Command):
    op = "add_segment"
    category = "create"
    summary = "Add a bare line segment between two world points."
    arguments = {"p0": _point("Start point [x, y] or [x, y, z]."),
                 "p1": _point("End point [x, y] or [x, y, z].")}
    returns = {"nsegment": _int(
        "Total segments in the world after the add, "
        "including those owned by shapes.")}

    def apply(self, world, args, ctx):
        world.add_segment(_world_point(args["p0"]), _world_point(args["p1"]))
        return {"nsegment": world.nsegment}


@DRAW.register
class AddLine(_cmd.Command):
    op = "add_line"
    category = "create"
    summary = "Add a line shape from (x0, y0) to (x1, y1) in world coords."
    arguments = {"x0": _num("World x of the start point."),
                 "y0": _num("World y of the start point."),
                 "x1": _num("World x of the end point."),
                 "y1": _num("World y of the end point.")}
    returns = {"shape_id": _int("Id of the new shape.")}

    def apply(self, world, args, ctx):
        return {"shape_id": world.add_line(
            args["x0"], args["y0"], args["x1"], args["y1"])}


@DRAW.register
class AddTriangle(_cmd.Command):
    op = "add_triangle"
    category = "create"
    summary = "Add a triangle shape through its three corners (+Y up)."
    arguments = {"x0": _num("World x of corner 0."),
                 "y0": _num("World y of corner 0."),
                 "x1": _num("World x of corner 1."),
                 "y1": _num("World y of corner 1."),
                 "x2": _num("World x of corner 2."),
                 "y2": _num("World y of corner 2.")}
    returns = {"shape_id": _int("Id of the new shape.")}

    def apply(self, world, args, ctx):
        return {"shape_id": world.add_triangle(
            args["x0"], args["y0"], args["x1"], args["y1"],
            args["x2"], args["y2"])}


@DRAW.register
class AddRectangle(_cmd.Command):
    op = "add_rectangle"
    category = "create"
    summary = "Add an axis-aligned rectangle from lower-left to upper-right."
    arguments = {"x_min": _num("Lower-left corner x."),
                 "y_min": _num("Lower-left corner y."),
                 "x_max": _num("Upper-right corner x."),
                 "y_max": _num("Upper-right corner y.")}
    returns = {"shape_id": _int("Id of the new shape.")}

    def apply(self, world, args, ctx):
        return {"shape_id": world.add_rectangle(
            args["x_min"], args["y_min"], args["x_max"], args["y_max"])}


@DRAW.register
class AddSquare(_cmd.Command):
    op = "add_square"
    category = "create"
    summary = "Add an axis-aligned square from its lower-left corner."
    arguments = {"x_min": _num("Lower-left corner x."),
                 "y_min": _num("Lower-left corner y."),
                 "size": _pos("Edge length; must be positive.")}
    returns = {"shape_id": _int("Id of the new shape.")}

    def apply(self, world, args, ctx):
        return {"shape_id": world.add_square(
            args["x_min"], args["y_min"], args["size"])}


@DRAW.register
class AddEllipse(_cmd.Command):
    op = "add_ellipse"
    category = "create"
    summary = "Add an ellipse shape centered at (cx, cy)."
    arguments = {"cx": _num("Center x."), "cy": _num("Center y."),
                 "rx": _pos("Semi-axis along x; must be positive."),
                 "ry": _pos("Semi-axis along y; must be positive.")}
    returns = {"shape_id": _int("Id of the new shape.")}

    def apply(self, world, args, ctx):
        return {"shape_id": world.add_ellipse(
            args["cx"], args["cy"], args["rx"], args["ry"])}


@DRAW.register
class AddCircle(_cmd.Command):
    op = "add_circle"
    category = "create"
    summary = "Add a circle shape centered at (cx, cy)."
    arguments = {"cx": _num("Center x."), "cy": _num("Center y."),
                 "r": _pos("Radius; must be positive.")}
    returns = {"shape_id": _int("Id of the new shape.")}

    def apply(self, world, args, ctx):
        return {"shape_id": world.add_circle(
            args["cx"], args["cy"], args["r"])}


@DRAW.register
class AddBezier(_cmd.Command):
    op = "add_bezier"
    category = "create"
    summary = "Add a bare cubic Bezier from four control points."
    arguments = {"p0": _point("Start anchor [x, y] or [x, y, z]."),
                 "p1": _point("First control point."),
                 "p2": _point("Second control point."),
                 "p3": _point("End anchor.")}
    returns = {"nbezier": _int(
        "Total Beziers in the world after the add, "
        "including those owned by shapes.")}

    def apply(self, world, args, ctx):
        world.add_bezier(_world_point(args["p0"]), _world_point(args["p1"]),
                         _world_point(args["p2"]), _world_point(args["p3"]))
        return {"nbezier": world.nbezier}


@DRAW.register
class AddBezierShape(_cmd.Command):
    op = "add_bezier_shape"
    category = "create"
    summary = "Add a cubic Bezier shape from four control points."
    arguments = {"p0": _point("Start anchor [x, y] or [x, y, z]."),
                 "p1": _point("First control point."),
                 "p2": _point("Second control point."),
                 "p3": _point("End anchor.")}
    returns = {"shape_id": _int("Id of the new shape.")}

    def apply(self, world, args, ctx):
        return {"shape_id": world.add_bezier_shape(
            _world_point(args["p0"]), _world_point(args["p1"]),
            _world_point(args["p2"]), _world_point(args["p3"]))}


@DRAW.register
class GetShape(_cmd.Command):
    op = "get_shape"
    category = "read"
    summary = "Read one shape's id, type, bbox, and geometry by its id."
    arguments = {"shape_id": _int("Id of the shape to read.")}
    returns = {"shape": SHAPE}

    def apply(self, world, args, ctx):
        # The World has no read-one accessor, so filter the rendered state.
        shape_id = args["shape_id"]
        state = json.loads(world.describe_state())
        for shape in state["shapes"]:
            if shape["id"] == shape_id:
                return {"shape": shape}
        raise _cmd.CommandError(f"no live shape with id {shape_id}")


@DRAW.register
class ShapeTypeOf(_cmd.Command):
    op = "shape_type_of"
    category = "read"
    summary = "Name the type of the shape with the given id."
    arguments = {"shape_id": _int("Id of the shape to inspect.")}
    returns = {"type": {**STRING,
                        "description": "Lower-case shape type."}}

    def apply(self, world, args, ctx):
        _require_live(world, args["shape_id"])
        return {"type": world.shape_type_of(args["shape_id"])}


@DRAW.register
class NShape(_cmd.Command):
    op = "nshape"
    category = "read"
    summary = "Count the live shapes in the world."
    returns = {"nshape": _int("Number of live shapes.")}

    def apply(self, world, args, ctx):
        return {"nshape": world.nshape}


@DRAW.register
class QueryVisible(_cmd.Command):
    op = "query_visible"
    category = "read"
    summary = "List the ids of shapes overlapping a query box."
    arguments = {"min_x": _num("Query box lower-left x."),
                 "min_y": _num("Query box lower-left y."),
                 "max_x": _num("Query box upper-right x."),
                 "max_y": _num("Query box upper-right y.")}
    returns = {"shape_ids": {"type": "array", "items": INTEGER,
                             "description": "Ids of shapes overlapping box."}}

    def apply(self, world, args, ctx):
        ids = world.query_visible(
            args["min_x"], args["min_y"], args["max_x"], args["max_y"])
        return {"shape_ids": list(ids)}


@DRAW.register
class DescribeState(_cmd.Command):
    op = "describe_state"
    category = "read"
    summary = "Serialize the visible 2D geometry to a state object."
    arguments = {"level": {"type": "string", "enum": ["basic"],
                           "default": "basic",
                           "description": "Level of detail; only 'basic'."}}
    optional = ("level",)
    returns = {"state": STATE}

    def apply(self, world, args, ctx):
        return {"state": json.loads(world.describe_state(level=args["level"]))}


@DRAW.register
class RenderPng(_cmd.Command):
    op = "render_png"
    category = "read"
    summary = "Render the world to a PNG via the offscreen renderer."
    arguments = {"width": {**POSITIVE_INT,
                           "description": "Image width in pixels."},
                 "height": {**POSITIVE_INT,
                            "description": "Image height in pixels."},
                 "view": {**VIEW,
                          "default": {"pan_x": 0.0, "pan_y": 0.0, "zoom": 1.0},
                          "description": "2D view transform (+Y up)."},
                 "antialiasing": {**BOOLEAN, "default": False,
                                  "description": "Enable antialiased edges."}}
    optional = ("view", "antialiasing")
    returns = {"image": IMAGE}

    def apply(self, world, args, ctx):
        renderer = getattr(ctx, "renderer", None)
        if renderer is None:
            raise _cmd.CommandError(
                "render_png needs a renderer; none configured. The harness "
                "and MCP front-ends inject the offscreen QImage renderer.")
        png = renderer(world, args["view"], args["width"], args["height"],
                       args["antialiasing"])
        # Base64 so the result is plain JSON over any transport, not bytes.
        return {"image": {"data": base64.b64encode(png).decode("ascii"),
                          "mime_type": "image/png",
                          "width": args["width"], "height": args["height"]}}


@DRAW.register
class TranslateShape(_cmd.Command):
    op = "translate_shape"
    category = "update"
    summary = "Translate the shape with the given id by (dx, dy)."
    arguments = {"shape_id": _int("Id of the shape to move."),
                 "dx": _num("World displacement along x."),
                 "dy": _num("World displacement along y.")}

    def apply(self, world, args, ctx):
        _require_live(world, args["shape_id"])
        world.translate_shape(args["shape_id"], args["dx"], args["dy"])
        return {}


@DRAW.register
class RemoveShape(_cmd.Command):
    op = "remove_shape"
    category = "delete"
    summary = "Remove the shape with the given id."
    arguments = {"shape_id": _int("Id of the shape to remove.")}

    def apply(self, world, args, ctx):
        _require_live(world, args["shape_id"])
        world.remove_shape(args["shape_id"])
        return {}


@DRAW.register
class Clear(_cmd.Command):
    op = "clear"
    category = "delete"
    summary = "Remove every shape and bare primitive."

    def apply(self, world, args, ctx):
        world.clear()
        return {}


@DRAW.register
class Log(_cmd.Command):
    op = "log"
    category = "log"
    summary = "Record a free-text note in the command log."
    arguments = {"message": {**STRING,
                             "description": "Free-text note to record."}}

    def apply(self, world, args, ctx):
        ctx.append_log(args["message"])
        return {}


COMMANDS = DRAW.commands
COMMAND_SCHEMAS = DRAW.command_schemas
RESULT_SCHEMAS = DRAW.result_schemas
SCHEMA = DRAW.schema
validate_command = DRAW.validate_command
validate_result = DRAW.validate_result
validate_script = DRAW.validate_script
apply_defaults = DRAW.apply_defaults
tool_definitions = DRAW.tool_definitions
commands_by_category = DRAW.commands_by_category

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
