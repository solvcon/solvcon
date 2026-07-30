# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Observation and scene formatting for the Agent (composition rules of #966).

The model reads two kinds of text about the world each step.  The *scene* is a
bounded snapshot: a header naming the shape count and types, a capped listing
of shapes, and an explicit tail pointing at the read commands when more remain.
An *observation* is one line per command result, with consecutive identical
errors collapsed so a repeated failure does not flood the model.  Both stay
compact and deterministic so small models keep focus and a byte-stable prefix
serves prompt caches.

Binary or oversized result values (``render_png``'s base64) are routed through
the session :class:`~solvcon.agent._artifact.ArtifactStore` by
:func:`offload_blobs`, so what the prompt and transcript keep is a path
reference, never the bytes.
"""

import json
import base64

from ._artifact import ArtifactError

# Extension used when naming a stored artifact, keyed by media type.
_MIME_SUFFIX = {"image/png": ".png", "image/jpeg": ".jpg"}

# Shapes listed in a scene before the remainder collapses into a tail.
SCENE_SHAPE_CAP = 20

# Longest a serialized result value may be before the observation truncates it.
_DETAIL_CAP = 200


def _looks_like_blob(node):
    """A base64 blob: a mapping with string ``data`` and ``mime_type``."""
    return (isinstance(node, dict)
            and isinstance(node.get("data"), str)
            and isinstance(node.get("mime_type"), str))


def has_blob(value):
    """Whether ``value`` holds any base64 blob, so a caller can skip building
    the artifact store when there is nothing to offload."""
    if _looks_like_blob(value):
        return True
    if isinstance(value, dict):
        return any(has_blob(item) for item in value.values())
    if isinstance(value, list):
        return any(has_blob(item) for item in value)
    return False


def _offload_one(node, store):
    ref = {key: val for key, val in node.items() if key != "data"}
    try:
        raw = base64.b64decode(node["data"], validate=True)
    except (ValueError, TypeError) as exc:
        ref["error"] = "invalid base64: %s" % exc
        return ref
    try:
        ref["path"] = store.store(raw, _MIME_SUFFIX.get(node["mime_type"], ""))
    except (ArtifactError, OSError) as exc:
        # Quota, a full disk, or a vanished scratch dir are storage problems,
        # not command failures: annotate the reference so the observation shows
        # the drop, but never abort the turn or lose the bytes silently.
        ref["error"] = str(exc)
    return ref


def offload_blobs(value, store):
    """A copy of ``value`` with every base64 blob replaced by an artifact
    reference.

    Recurses into mappings and lists.  Each blob's ``data`` is decoded, written
    to ``store``, and replaced by a ``path``.  A blob that fails to decode,
    exceeds the quota, or cannot be written keeps no bytes and carries an
    ``error`` in its reference instead, degrading the result honestly rather
    than faking a path.  Storing the artifact is the harness's job, so the
    command outcome the runner reported is left untouched.
    """
    def walk(node):
        if _looks_like_blob(node):
            return _offload_one(node, store)
        if isinstance(node, dict):
            return {key: walk(val) for key, val in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(value)


def _bounded(text):
    """Collapse whitespace to a single line and cap the length, so one result
    can never spill across lines or flood the prompt."""
    flat = " ".join(text.split())
    return flat[:_DETAIL_CAP] + "..." if len(flat) > _DETAIL_CAP else flat


def _result_detail(value):
    if value is None:
        return ""
    if value == {} or value == []:
        return "empty"
    return _bounded(json.dumps(value, sort_keys=True, separators=(",", ":")))


def format_result(result):
    """One observation line for a single command result: ``op: ok`` with a
    compact value tail, or ``op: error: <reason>`` on failure."""
    op = getattr(result, "op", None) or "?"
    if not getattr(result, "ok", False):
        reason = getattr(result, "error", None) or "failed"
        return "%s: error: %s" % (op, _bounded(str(reason)))
    detail = _result_detail(getattr(result, "value", None))
    return "%s: ok (%s)" % (op, detail) if detail else "%s: ok" % op


def format_results(results):
    """The observation for a command batch: one line per result, with
    consecutive identical error lines collapsed.  An empty batch reads
    explicitly as such."""
    if not results:
        return "no commands run"
    lines = []
    previous = None
    repeat = 0
    for result in results:
        line = format_result(result)
        if not getattr(result, "ok", False) and line == previous:
            repeat += 1
            continue
        if repeat:
            lines.append("... and %d more identical errors" % repeat)
            repeat = 0
        lines.append(line)
        previous = line
    if repeat:
        lines.append("... and %d more identical errors" % repeat)
    return "\n".join(lines)


def _fmt_num(value):
    return "%g" % value if isinstance(value, (int, float)) else str(value)


def _shape_line(shape):
    if not isinstance(shape, dict):
        return str(shape)
    head = "#%s %s" % (shape.get("id", "?"), shape.get("type", "?"))
    bbox = shape.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return "%s bbox=[%s]" % (head, ", ".join(_fmt_num(v) for v in bbox))
    return head


def format_scene(state, cap=SCENE_SHAPE_CAP):
    """A bounded scene snapshot for the prompt.

    ``state`` is the parsed ``describe_state`` mapping.  The header names the
    shape count and distinct types; up to ``cap`` shapes are listed one per
    line, and any remainder collapses into an explicit tail directing the model
    to the read commands.
    """
    shapes = state.get("shapes", []) if isinstance(state, dict) else []
    types = sorted({s["type"] for s in shapes
                    if isinstance(s, dict) and "type" in s})
    header = "world with %d shapes (types: %s)" % (
        len(shapes), ", ".join(types) if types else "none")
    lines = [header]
    lines.extend("  " + _shape_line(shape) for shape in shapes[:cap])
    extra = len(shapes) - cap
    if extra > 0:
        lines.append(
            "  ... %d more shapes, use read commands (describe_state)" % extra)
    return "\n".join(lines)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
