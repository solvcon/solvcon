# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Compile ROS 2 IDL schemas into flat CDR decode instructions.

MCAP ``ros2idl`` schemas contain the root IDL and dependency sections.
``DecodePlan`` resolves their structs and compiles dotted field paths into
the instruction sequence an executor runs over one message payload.  Every
offset is relative to the start of the CDR body, which begins after the
four-byte encapsulation header that also states the byte order.

Each instruction is a tuple whose first item names it:

- ``("align", n)`` advances to the next multiple of ``n`` bytes.
- ``("skip", n)`` advances ``n`` bytes.
- ``("skip_string", )`` reads a 4-byte length and advances past the text.
- ``("skip_sequence", width)`` reads a 4-byte count and advances past that
  many primitives of ``width`` bytes.  CDR pads between the count and the
  elements, so when the count is nonzero the walk aligns to ``width``
  first.  An empty sequence carries no padding.
- ``("skip_sequence_body", n)`` reads a 4-byte count and runs the ``n``
  instructions that follow it that many times.
- ``("skip_array_body", count, n)`` runs the ``n`` instructions that follow
  it ``count`` times.
- ``("read", name, column)`` reads one primitive of type ``name`` into the
  output column ``column``.

A body may hold another body, so an executor walks nested containers by
recursion.  Selection is numeric and bool scalars only, and an enum reads as
its 32-bit ordinal; a container, a string, or a nested struct in the output
position is rejected, as is an indexed path.
"""

import dataclasses
import re


class DecodePlanError(ValueError):
    """A schema or requested field cannot form a supported decode plan."""


@dataclasses.dataclass(frozen=True)
class _Type:
    kind: str
    name: str = ""
    size: int = 0
    count: int = 0
    element: object = None


@dataclasses.dataclass(frozen=True)
class _Field:
    name: str
    type: _Type


_PRIMITIVES = {
    "boolean": ("bool", 1),
    "octet": ("uint8", 1),
    "char": ("uint8", 1),
    "float": ("float32", 4),
    "double": ("float64", 8),
    "int8": ("int8", 1),
    "uint8": ("uint8", 1),
    "int16": ("int16", 2),
    "uint16": ("uint16", 2),
    "int32": ("int32", 4),
    "uint32": ("uint32", 4),
    "int64": ("int64", 8),
    "uint64": ("uint64", 8),
}

# Annotations that move bytes on the wire.  Stripping one would compile a
# plan against a layout the message does not use, so reject it instead.
_LAYOUT_ANNOTATIONS = frozenset((
    "optional", "bit_bound", "extensibility", "mutable", "appendable",
    "external", "id", "hashid", "try_construct",
))

_SECTION_RE = re.compile(r"(?m)^={16,}\s*\nIDL:\s*([^\n]+?)\s*\n")
_STRUCT_RE = re.compile(
    r"\bstruct\s+([A-Za-z]\w*)\s*\{(.*?)\}\s*;", re.DOTALL)
_ENUM_RE = re.compile(
    r"\benum\s+([A-Za-z]\w*)\s*\{.*?\}\s*;", re.DOTALL)
_MEMBER_RE = re.compile(
    r"(.+?)\s+([a-zA-Z]\w*)(?:\s*\[\s*([^]]+)\s*\])?")
_PATH_RE = re.compile(r"[a-zA-Z]\w*(?:\.[a-zA-Z]\w*)*")


def _schema_text(schema):
    root = getattr(schema, "name", "")
    if hasattr(schema, "data"):
        encoding = getattr(schema, "encoding", None)
        if encoding != "ros2idl":
            raise DecodePlanError(
                "schema encoding must be 'ros2idl', not %r" % encoding)
        schema = schema.data
    if isinstance(schema, bytes):
        try:
            schema = schema.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecodePlanError("schema is not valid UTF-8") from exc
    if not isinstance(schema, str):
        raise TypeError("schema must be text, bytes, or an MCAP schema")
    return root.replace("/", "::"), schema


def _skip_literal(text, index):
    quote = text[index]
    index += 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == quote:
            return index + 1
        index += 1
    return index


def _skip_annotation(text, index):
    start = index + 1
    index = start
    while index < len(text) and (text[index].isalnum()
                                 or text[index] == "_"):
        index += 1
    name_end = index
    if text[start:index] in _LAYOUT_ANNOTATIONS:
        raise DecodePlanError(
            "annotation @%s changes the CDR layout" % text[start:index])
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "(":
        return name_end
    depth = 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            index = _skip_literal(text, index)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if 0 == depth:
                return index + 1
        index += 1
    return index


def _clean_idl(text):
    # One pass, so that a "//" inside an annotation string does not end
    # the annotation early.
    text = re.sub(r"(?m)^\s*#.*$", " ", text)
    out, index = [], 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            index = _skip_literal(text, index)
        elif text.startswith("//", index):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end < 0 else end + 2
        elif char == "@":
            index = _skip_annotation(text, index)
        else:
            out.append(char)
            index += 1
            continue
        out.append(" ")
    return "".join(out)


def _sections(text, root):
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        if not root:
            raise DecodePlanError("IDL text has no root type name")
        return [(root, text)]
    sections = []
    for index, match in enumerate(matches):
        end = (matches[index + 1].start()
               if index + 1 < len(matches) else len(text))
        sections.append((match.group(1).replace("/", "::"),
                         text[match.end():end]))
    return sections


def _split_arguments(text):
    arguments, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "," and depth == 0:
            arguments.append(text[start:index])
            start = index + 1
    arguments.append(text[start:])
    return [argument.strip() for argument in arguments]


def _type_of(text, namespace, enums):
    text = re.sub(r"\s*::\s*", "::", text.strip())
    text = re.sub(r"\s*([<>,])\s*", r"\1", text)
    primitive = _PRIMITIVES.get(text)
    if primitive:
        return _Type("primitive", *primitive)
    if re.fullmatch(r"string(?:<[^>]+>)?", text):
        return _Type("string")
    if text.startswith("sequence<") and text.endswith(">"):
        arguments = _split_arguments(text[9:-1])
        if len(arguments) not in (1, 2):
            raise DecodePlanError("invalid sequence type %r" % text)
        return _Type("sequence",
                     element=_type_of(arguments[0], namespace, enums))
    qualified = text if "::" in text else namespace + "::" + text
    if qualified in enums:
        return _Type("primitive", "uint32", 4)
    if not re.fullmatch(r"[A-Za-z]\w*(?:::[A-Za-z]\w*)*", qualified):
        raise DecodePlanError("unsupported IDL type %r" % text)
    return _Type("struct", qualified)


def _parse_schema(schema):
    root, text = _schema_text(schema)
    parts = _sections(text, root)
    if not root:
        root = parts[0][0]
    raw_structs, enums = {}, set()
    for section_name, source in parts:
        namespace = section_name.rsplit("::", 1)[0]
        clean = _clean_idl(source)
        enums.update(namespace + "::" + match.group(1)
                     for match in _ENUM_RE.finditer(clean))
        for match in _STRUCT_RE.finditer(clean):
            raw_structs[namespace + "::" + match.group(1)] = (
                namespace, match.group(2))

    structs = {}
    for name, (namespace, body) in raw_structs.items():
        fields = []
        for statement in body.split(";"):
            statement = " ".join(statement.split())
            if not statement or statement.startswith("const "):
                continue
            match = _MEMBER_RE.fullmatch(statement)
            if not match:
                raise DecodePlanError(
                    "unsupported member declaration %r in %s" %
                    (statement, name))
            type_text, field_name, count = match.groups()
            field_type = _type_of(type_text, namespace, enums)
            if count is not None:
                if not count.strip().isdigit():
                    raise DecodePlanError(
                        "array field %r has a non-integer bound" % field_name)
                field_type = _Type("array", count=int(count),
                                   element=field_type)
            fields.append(_Field(field_name, field_type))
        structs[name] = tuple(fields)
    if root not in structs:
        raise DecodePlanError("schema does not define root struct %r" % root)
    return root, structs


def _requested_fields(fields):
    if isinstance(fields, str):
        raise TypeError("fields must be a sequence of field paths")
    try:
        fields = tuple(fields)
    except TypeError as exc:
        raise TypeError("fields must be a sequence of field paths") from exc
    if not fields:
        raise DecodePlanError("at least one field must be requested")
    for field in fields:
        if not isinstance(field, str):
            raise TypeError("each requested field path must be a string")
        if not _PATH_RE.fullmatch(field):
            raise DecodePlanError(
                "indexed or invalid field path %r is not supported" % field)
    if len(set(fields)) != len(fields):
        raise DecodePlanError("requested field paths must be unique")
    return fields


def _align(instructions, boundary):
    if boundary > 1:
        instructions.append(("align", boundary))


class _Compiler:

    def __init__(self, structs, requested):
        self.structs = structs
        self.instructions = []
        self.types = [None] * len(requested)
        self.selected = [(tuple(path.split(".")), column, path)
                         for column, path in enumerate(requested)]
        self.active = set()

    def skip_container(self, opcode, element, count=None):
        """Emit a loop over the walk program of one container element."""
        head = len(self.instructions)
        self.instructions.append(None)
        self.skip(element)
        body = len(self.instructions) - head - 1
        self.instructions[head] = ((opcode, body) if count is None
                                   else (opcode, count, body))

    def skip(self, field_type):
        element = field_type.element
        if field_type.kind == "primitive":
            _align(self.instructions, field_type.size)
            self.instructions.append(("skip", field_type.size))
        elif field_type.kind == "string":
            _align(self.instructions, 4)
            self.instructions.append(("skip_string",))
        elif field_type.kind == "struct":
            self.compile_struct(field_type.name, (), True)
        elif field_type.kind == "array":
            if element.kind == "primitive":
                _align(self.instructions, element.size)
                self.instructions.append(("skip",
                                          element.size * field_type.count))
            else:
                self.skip_container("skip_array_body", element,
                                    field_type.count)
        else:
            _align(self.instructions, 4)
            if element.kind == "primitive":
                self.instructions.append(("skip_sequence", element.size))
            else:
                self.skip_container("skip_sequence_body", element)

    def compile_struct(self, name, selected, need_end):
        fields = self.structs.get(name)
        if fields is None:
            raise DecodePlanError("schema lacks nested struct %r" % name)
        if name in self.active:
            raise DecodePlanError("struct %r contains itself" % name)
        self.active.add(name)
        groups = {}
        for parts, column, path in selected:
            groups.setdefault(parts[0], []).append((parts[1:], column, path))
        field_names = {field.name for field in fields}
        for part, wanted in groups.items():
            if part not in field_names:
                raise DecodePlanError(
                    "schema has no field path %r" % wanted[0][2])
        last = max((index for index, field in enumerate(fields)
                    if field.name in groups), default=-1)
        limit = len(fields) if need_end else last + 1
        for index, field in enumerate(fields[:limit]):
            wanted = groups.get(field.name, ())
            if not wanted:
                self.skip(field.type)
                continue
            if (field.type.kind == "primitive"
                    and all(not tail for tail, _c, _p in wanted)):
                _align(self.instructions, field.type.size)
                for _tail, column, _path in wanted:
                    self.instructions.append(
                        ("read", field.type.name, column))
                    self.types[column] = field.type.name
            elif (field.type.kind == "struct"
                  and all(tail for tail, _c, _p in wanted)):
                self.compile_struct(field.type.name, wanted,
                                    need_end or index + 1 < limit)
            else:
                raise DecodePlanError(
                    "field path %r does not select a numeric or bool scalar" %
                    wanted[0][2])
        self.active.discard(name)

    def compile(self, root):
        self.compile_struct(root, self.selected, False)
        return tuple(self.types), tuple(self.instructions)


class DecodePlan:
    """A flat CDR walk program for selected scalar IDL field paths."""

    __slots__ = ("_fields", "_types", "_instructions")

    def __init__(self, schema, fields):
        requested = _requested_fields(fields)
        root, structs = _parse_schema(schema)
        types, instructions = _Compiler(structs, requested).compile(root)
        self._fields = requested
        self._types = types
        self._instructions = instructions

    @property
    def fields(self):
        """Requested field paths in output-column order."""
        return self._fields

    @property
    def types(self):
        """Canonical primitive type of each output column."""
        return self._types

    @property
    def instructions(self):
        """Immutable flat instruction sequence in schema walk order."""
        return self._instructions


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
