# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Schema layer: from ``ros2idl`` text to a ``DecodePlan``.

IDL is the OMG Interface Definition Language.  ROS 2 embeds IDL text
in a recording under the ``ros2idl`` schema encoding, and this module
parses that text.

``parse_schema`` turns the schema text into a registry of structs and
enums.  ``DecodePlan`` flattens the root struct into scalar leaves and
compiles the selected ones into one ``struct`` format per byte order, so a
message decodes with a single ``unpack_from``.

The plan flattens structs whose fields are scalars, enums, or other
structs.  A string, sequence, array, or union field raises ``McapError``
when a plan reaches it.
"""

import re
import struct
import collections

import solvcon as sc

from . import McapError

__all__ = ["parse_schema", "DecodePlan", "COLUMN_TYPES"]

Registry = collections.namedtuple("Registry", "structs enums")

# IDL scalar type -> column dtype.
SCALAR_TYPES = {
    "boolean": "bool",
    "octet": "uint8",
    "uint8": "uint8",
    "int8": "int8",
    "uint16": "uint16",
    "int16": "int16",
    "uint32": "uint32",
    "int32": "int32",
    "uint64": "uint64",
    "int64": "int64",
    "float": "float32",
    "double": "float64",
}
ENUM_TYPE = "uint32"

# Column dtype -> (struct format character, SimpleArray class).
COLUMN_TYPES = {
    "bool": ("?", sc.SimpleArrayBool),
    "uint8": ("B", sc.SimpleArrayUint8),
    "int8": ("b", sc.SimpleArrayInt8),
    "uint16": ("H", sc.SimpleArrayUint16),
    "int16": ("h", sc.SimpleArrayInt16),
    "uint32": ("I", sc.SimpleArrayUint32),
    "int32": ("i", sc.SimpleArrayInt32),
    "uint64": ("Q", sc.SimpleArrayUint64),
    "int64": ("q", sc.SimpleArrayInt64),
    "float32": ("f", sc.SimpleArrayFloat32),
    "float64": ("d", sc.SimpleArrayFloat64),
}

_FIELD = re.compile(r"^([\w:]+)\s+(\w+);$")
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_ANNOTATION = re.compile(r'@\w+(\s*\((?:[^()"]|"(?:[^"\\]|\\.)*")*\))?')


def parse_schema(schema):
    """
    Parse a ``ros2idl`` ``Schema`` into a ``Registry``.

    The text may be the concatenated form of a ROS 2 recording: several
    ``IDL:`` blocks with ``#include`` lines, block comments, and
    annotations, which the parser drops.  ``structs`` maps a scoped name to
    ``(field, type, scope)`` triples in declaration order, where ``scope``
    is the enclosing module path that resolves an unqualified ``type``; a
    field the prototype cannot decode keeps its whole declaration as
    ``field`` with a ``None`` type.  ``enums`` maps a scoped name to its
    member names.
    """
    if schema.encoding != "ros2idl":
        raise McapError("schema encoding {!r}".format(schema.encoding))
    text = _ANNOTATION.sub("", _COMMENT.sub("", schema.data.decode()))

    scope = []
    structs = {}
    enums = {}
    for line in text.splitlines():
        line = line.split("//")[0].strip()
        if not line or line.startswith(("=", "IDL:", "#")):
            continue
        if line.endswith("{"):
            kind, name = line.split()[:2]
            scope.append((kind, name))
            scoped = "::".join(name for _, name in scope)
            if kind == "struct":
                structs[scoped] = []
            elif kind == "enum":
                enums[scoped] = []
            continue
        if line == "};":
            scope.pop()
            continue

        kind, name = scope[-1]
        scoped = "::".join(name for _, name in scope)
        if kind == "enum":
            enums[scoped].append(line.rstrip(","))
        elif kind == "struct":
            field = _FIELD.match(line)
            modules = tuple(name for _, name in scope[:-1])
            if field:
                type_name, field_name = field.groups()
                structs[scoped].append((field_name, type_name, modules))
            else:
                structs[scoped].append((line, None, modules))
    return Registry(structs, enums)


class DecodePlan:
    """
    The selected scalar leaves of one schema.

    ``fields`` are the selected dotted paths, ``types`` their column dtypes,
    and ``enums`` the member names of each enum-typed field.  Without
    ``fields`` the plan selects every scalar leaf in declaration order.  An
    enum reads as its ``uint32`` ordinal.
    """

    def __init__(self, schema, fields=None):
        self.schema = schema
        registry = parse_schema(schema)
        leaves = _leaves(registry, schema.name.replace("/", "::"), "", ())
        types = {path: dtype for path, dtype, _ in leaves}

        self.fields = tuple(types if fields is None else fields)
        unknown = [path for path in self.fields if path not in types]
        if unknown:
            raise McapError("unknown or non-scalar fields {}".format(unknown))
        self.types = tuple(types[path] for path in self.fields)
        self.enums = {path: members for path, _, members in leaves
                      if members and path in self.fields}

        # CDR aligns each scalar to its size from the body start, so the
        # offsets are static and the whole body is one struct format.
        fmt = ""
        offset = 0
        paths = []
        for path, dtype, _ in leaves:
            code = COLUMN_TYPES[dtype][0]
            size = struct.calcsize(code)
            pad = -offset % size
            fmt += "{}x{}".format(pad, code) if pad else code
            offset += pad + size
            paths.append(path)
        self._structs = {0: struct.Struct(">" + fmt),
                         1: struct.Struct("<" + fmt)}
        self._select = tuple(paths.index(path) for path in self.fields)

    def decode(self, payload):
        """Return the selected values of one CDR payload as a tuple."""
        if len(payload) < 4 or payload[1] not in self._structs:
            raise McapError("unsupported CDR encapsulation")
        values = self._structs[payload[1]].unpack_from(payload, 4)
        return tuple(values[i] for i in self._select)


def _leaves(registry, struct_name, prefix, visiting):
    """Return ``(path, dtype, enum_members)`` per scalar leaf, in order."""
    if struct_name in visiting or struct_name not in registry.structs:
        raise McapError("bad struct {}".format(struct_name))
    leaves = []
    for field_name, type_name, modules in registry.structs[struct_name]:
        path = prefix + field_name
        if type_name in SCALAR_TYPES:
            leaves.append((path, SCALAR_TYPES[type_name], None))
            continue
        type_name = _resolve(registry, type_name, modules)
        if type_name in registry.enums:
            leaves.append((path, ENUM_TYPE, tuple(registry.enums[type_name])))
        elif type_name in registry.structs:
            leaves.extend(_leaves(registry, type_name, path + ".",
                                  visiting + (struct_name,)))
        else:
            raise McapError("unsupported field {!r} in {}".format(
                path, struct_name))
    return leaves


def _resolve(registry, type_name, modules):
    """Look ``type_name`` up from the innermost enclosing module outward."""
    for depth in range(len(modules), -1, -1) if type_name else ():
        scoped = "::".join(modules[:depth] + (type_name,))
        if scoped in registry.structs or scoped in registry.enums:
            return scoped
    return None

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
