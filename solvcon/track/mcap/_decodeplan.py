# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Schema layer: from ``ros2idl`` text to a ``DecodePlan``.

IDL is the OMG Interface Definition Language.  ROS 2 embeds IDL text in a
recording under the ``ros2idl`` schema encoding, and ``parse_schema``
turns that text into one ``Registry`` of structs, enums, unions,
typedefs, and integer constants.

``DecodePlan`` turns the root struct into a tree and walks it over the
CDR body of each message.  A scalar or enum leaf reached through nested
structs is a column.  The walk steps over a string, a sequence, or an
array, so a leaf inside one is not a column.  A union field raises
``McapError`` when a plan reaches it.
"""

import re
import struct
import collections

import solvcon as sc

from . import McapError

__all__ = ["parse_schema", "DecodePlan", "COLUMN_TYPES"]

Registry = collections.namedtuple("Registry",
                                  "structs enums unions typedefs consts")

# IDL scalar type -> column dtype.
SCALAR_TYPES = {
    "boolean": "bool",
    "octet": "uint8",
    "char": "uint8",
    "uint8": "uint8",
    "int8": "int8",
    "uint16": "uint16",
    "int16": "int16",
    "unsigned short": "uint16",
    "short": "int16",
    "uint32": "uint32",
    "int32": "int32",
    "unsigned long": "uint32",
    "long": "int32",
    "uint64": "uint64",
    "int64": "int64",
    "unsigned long long": "uint64",
    "long long": "int64",
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

_COMMENT = re.compile(r'("(?:[^"\\]|\\.)*")|/\*.*?\*/|//[^\n]*', re.S)
_ANNOTATION = re.compile(r'@\w+(\s*\((?:[^()"]|"(?:[^"\\]|\\.)*")*\))?')
_ENUM_VALUE = re.compile(r"@value\s*\(\s*(\w+)\s*\)\s*(\w+)")
_TOKEN = re.compile(r'[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*|"(?:[^"\\]|\\.)*"'
                    r'|\d[\w.]*|\S')
_SEPARATOR = re.compile(r"^=+\s*$", re.M)

# Indexed by the byte order flag of the CDR encapsulation: 0 big, 1 little.
_ORDERS = (">", "<")
_SCALARS = tuple({dtype: struct.Struct(order + code)
                  for dtype, (code, _) in COLUMN_TYPES.items()}
                 for order in _ORDERS)
_TRUNCATED = "the payload ends before the fields do"


def parse_schema(schema):
    """
    Parse a ``ros2idl`` ``Schema`` into a ``Registry``.

    Every name in the registry is scoped with ``::``.  ``structs`` maps a
    name to ``(field, type)`` pairs in declaration order, and ``enums``
    maps a name to the member names.  ``unions`` maps a name to
    ``(discriminator, cases)``.  Each case is ``(values, type)``, where
    ``values`` holds the discriminator values it matches and ``None``
    marks the default case.  ``typedefs`` maps a name to the aliased
    type.  ``consts`` maps a name to an integer value; the parser drops
    a constant that is not an integer.

    A type is a tuple headed by ``scalar``, ``string``, ``sequence``,
    ``array``, ``unsupported``, or ``named``.  A ``named`` type carries
    the raw name and the enclosing module path that resolves it.
    """
    registry = Registry({}, {}, {}, {}, {})
    try:
        text = schema.data.decode()
    except UnicodeDecodeError:
        raise McapError("schema {} is not UTF-8".format(
            schema.name)) from None
    if schema.encoding != "ros2idl":
        raise McapError("schema encoding {!r} of {}".format(
            schema.encoding, schema.name))
    _IdlParser(_strip_idl(text), registry).definitions((), None)
    return registry


def _strip_idl(text):
    """Drop annotations, comments, includes, and the bundle separators."""
    text = _COMMENT.sub(lambda match: match.group(1) or "", text)
    text = _ENUM_VALUE.sub(r"\2 = \1", text)
    lines = [line for line in _ANNOTATION.sub("", text).splitlines()
             if not line.lstrip().startswith(("#", "IDL:"))
             and not _SEPARATOR.match(line.strip())]
    return "\n".join(lines)


class _IdlParser:

    def __init__(self, text, registry):
        self.tokens = _TOKEN.findall(text)
        self.pos = 0
        self.registry = registry

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        token = self.peek()
        self.pos += 1
        return token

    def expect(self, token):
        found = self.next()
        if found != token:
            raise McapError("expected {!r} but found {!r} in the IDL".format(
                token, found))

    def skip_optional(self, token):
        if self.peek() == token:
            self.pos += 1

    def skip_statement(self):
        while self.next() not in (";", None):
            pass

    def definitions(self, modules, end):
        while True:
            token = self.next()
            if token == end:
                return
            if token is None:
                raise McapError("the IDL ends inside a block")
            if token == "module":
                name = self.next()
                self.expect("{")
                self.definitions(modules + (name,), "}")
                self.skip_optional(";")
            elif token == "struct":
                self.struct(modules)
            elif token == "enum":
                self.enum(modules)
            elif token == "union":
                self.union(modules)
            elif token == "typedef":
                type_ = self.type(modules)
                name = self.next()
                type_ = _arrayed(type_, self.dims(modules))
                self.registry.typedefs[_scoped(modules, name)] = type_
                self.expect(";")
            elif token == "const":
                self.const(modules)
            elif token != ";":
                self.skip_statement()

    def struct(self, modules):
        name = self.next()
        if self.peek() == ";":
            self.next()
            return
        if self.peek() == ":":
            raise McapError("struct {} inherits, which is unsupported".format(
                _scoped(modules, name)))
        self.expect("{")
        fields = []
        while self.peek() not in ("}", None):
            type_ = self.type(modules)
            while True:
                field = self.next()
                fields.append((field, _arrayed(type_, self.dims(modules))))
                if self.peek() != ",":
                    break
                self.next()
            self.expect(";")
        self.expect("}")
        self.skip_optional(";")
        self.registry.structs[_scoped(modules, name)] = fields

    def enum(self, modules):
        name = self.next()
        self.expect("{")
        members = []
        while self.peek() not in ("}", None):
            token = self.next()
            if token == "=":
                if self.integer(self.next(), modules) != len(members) - 1:
                    raise McapError("enum {} assigns values out of "
                                    "declaration order".format(name))
            elif token != ",":
                members.append(token)
        self.expect("}")
        self.skip_optional(";")
        self.registry.enums[_scoped(modules, name)] = tuple(members)

    def union(self, modules):
        name = self.next()
        self.expect("switch")
        self.expect("(")
        discriminator = self.type(modules)
        self.expect(")")
        self.expect("{")
        members = None
        if discriminator[0] == "named":
            scoped = _lookup(self.registry.enums, discriminator[1], modules)
            members = self.registry.enums.get(scoped)
        cases = []
        values = []
        while self.peek() not in ("}", None):
            token = self.next()
            if token == "case":
                label = ""
                while self.peek() not in (":", None):
                    label += self.next()
                self.expect(":")
                values.append(self.label_value(label, members, modules))
            elif token == "default":
                values.append(None)
                self.expect(":")
            else:
                self.pos -= 1
                type_ = self.type(modules)
                self.next()
                cases.append((tuple(values),
                              _arrayed(type_, self.dims(modules))))
                values = []
                self.expect(";")
        self.expect("}")
        self.skip_optional(";")
        self.registry.unions[_scoped(modules, name)] = (discriminator, cases)

    def label_value(self, label, members, modules):
        """Return the discriminator value a union case label names."""
        if members is not None and label.split("::")[-1] in members:
            return members.index(label.split("::")[-1])
        if label in ("TRUE", "FALSE"):
            return label == "TRUE"
        if len(label) == 3 and label[0] == label[2] == "'":
            return ord(label[1])
        value = self.integer(label, modules)
        if value is None:
            raise McapError("bad union case label {!r}".format(label))
        return value

    def integer(self, token, modules):
        """Return the integer a token holds or names, or ``None``."""
        if token is None:
            raise McapError("the IDL ends inside a block")
        try:
            return int(token, 0)
        except ValueError:
            return self.registry.consts.get(
                _lookup(self.registry.consts, token, modules))

    def const(self, modules):
        self.type(modules)
        name = self.next()
        self.expect("=")
        tokens = []
        while self.peek() not in (";", None):
            tokens.append(self.next())
        self.expect(";")
        try:
            value = int("".join(tokens), 0)
        except ValueError:
            return
        self.registry.consts[_scoped(modules, name)] = value

    def type(self, modules):
        token = self.next()
        if token == "unsigned":
            token += " " + self.next()
        if token in ("long", "unsigned long") and \
                self.peek() in ("long", "double"):
            token += " " + self.next()
        if token in SCALAR_TYPES:
            return ("scalar", SCALAR_TYPES[token])
        if token == "string":
            if self.peek() == "<":
                self.next()
                self.next()
                self.expect(">")
            return ("string",)
        if token == "sequence":
            self.expect("<")
            element = self.type(modules)
            if self.peek() == ",":
                self.next()
                self.next()
            self.expect(">")
            return ("sequence", element)
        if token in ("long double", "wchar", "wstring", "map", "bitmask",
                     "bitset"):
            return ("unsupported", token)
        if token is None or not token[0].isalpha() and token[0] != "_":
            raise McapError("expected a type but found {!r}".format(token))
        return ("named", token, modules)

    def dims(self, modules):
        """Return the flattened array size after a name, or ``None``."""
        size = None
        while self.peek() == "[":
            self.next()
            token = self.next()
            self.expect("]")
            dim = self.integer(token, modules)
            if dim is None:
                raise McapError("bad array size {!r}".format(token))
            size = dim if size is None else size * dim
        return size


def _scoped(modules, name):
    return "::".join(modules + (name,))


def _arrayed(type_, size):
    return type_ if size is None else ("array", type_, size)


def _lookup(table, name, modules):
    """Find ``name`` in ``table`` from the innermost module outward."""
    for depth in range(len(modules), -1, -1):
        scoped = _scoped(modules[:depth], name)
        if scoped in table:
            return scoped
    return None


class DecodePlan:
    """
    The selected scalar leaves of one schema.

    ``fields`` are the selected dotted paths, ``types`` their column
    dtypes, and ``enums`` the member names of each enum-typed field.
    Without ``fields`` the plan selects every scalar leaf in declaration
    order.  An enum reads as its ``uint32`` ordinal.
    """

    def __init__(self, schema, fields=None):
        self.schema = schema
        registry = parse_schema(schema)
        self._tree = _tree(registry, ("named", schema.name.replace("/", "::"),
                                      ()), ())
        leaves = _leaves(self._tree, ())
        types = {path: dtype for path, dtype, _ in leaves}

        self.fields = tuple(types if fields is None else fields)
        unknown = [path for path in self.fields if path not in types]
        if unknown:
            raise McapError("unknown or non-scalar fields {}".format(unknown))
        if len(set(self.fields)) != len(self.fields):
            raise McapError("duplicate fields in {}".format(self.fields))
        self.types = tuple(types[path] for path in self.fields)
        self.enums = {path: members for path, _, members in leaves
                      if members and path in self.fields}
        paths = [path for path, _, _ in leaves]
        self._select = tuple(paths.index(path) for path in self.fields)

    def decode(self, payload):
        """Return the selected values of one CDR payload as a tuple."""
        if len(payload) < 4 or payload[0] != 0 or payload[1] > 1:
            raise McapError("unsupported CDR encapsulation")
        body = memoryview(payload)[4:]
        values = []
        try:
            end = _walk(self._tree, body, 0, payload[1], values)
        except struct.error:
            raise McapError(_TRUNCATED) from None
        if end > len(body):
            raise McapError(_TRUNCATED)
        return tuple(values[i] for i in self._select)


def _tree(registry, type_, visiting):
    """
    Return the tree that ``_walk`` runs over ``type_``.

    A node is ``("scalar", dtype, enum members)``, ``("struct", [(field,
    node), ...])``, ``("string",)``, ``("sequence", node)``, or
    ``("array", node, count)``.
    """
    type_ = _resolve(registry, type_)
    kind = type_[0]
    if kind == "scalar":
        return ("scalar", type_[1], None)
    if kind == "enum":
        return ("scalar", ENUM_TYPE, registry.enums[type_[1]])
    if kind == "struct":
        name = type_[1]
        if name in visiting:
            raise McapError("struct {} contains itself".format(name))
        return ("struct", [(field, _tree(registry, field_type,
                                         visiting + (name,)))
                           for field, field_type in registry.structs[name]])
    if kind == "string":
        return ("string",)
    if kind == "sequence":
        return ("sequence", _tree(registry, type_[1], visiting))
    if kind == "array":
        return ("array", _tree(registry, type_[1], visiting), type_[2])
    raise McapError("unsupported type {!r}".format(type_[1]))


def _leaves(node, path):
    """Return ``(path, dtype, enum members)`` per scalar leaf, in order."""
    if node[0] == "scalar":
        return [(".".join(path), node[1], node[2])]
    if node[0] == "struct":
        return [leaf for field, child in node[1]
                for leaf in _leaves(child, path + (field,))]
    return []


def _walk(node, body, pos, order, values):
    """
    Walk ``node`` over the CDR ``body`` from ``pos`` and return the end.

    ``body`` starts after the encapsulation header, at the origin of CDR
    alignment.  ``values`` collects every scalar leaf in the order of
    ``_leaves``, or is ``None`` inside a container, whose scalars are not
    columns.
    """
    kind = node[0]
    if kind == "scalar":
        fmt = _SCALARS[order][node[1]]
        pos += -pos % fmt.size
        if values is not None:
            values.append(fmt.unpack_from(body, pos)[0])
        return pos + fmt.size
    if kind == "struct":
        for _, child in node[1]:
            pos = _walk(child, body, pos, order, values)
        return pos
    # TODO: a string, a sequence, or an array is only skipped; none of them
    # produces a column yet.
    if kind == "array":
        count = node[2]
    else:
        pos += -pos & 3
        count = _SCALARS[order]["uint32"].unpack_from(body, pos)[0]
        pos += 4
    if count > len(body) - pos:
        raise McapError(_TRUNCATED)
    if kind == "string":
        return pos + count
    if node[1][0] == "scalar":
        size = _SCALARS[order][node[1][1]].size
        return pos + (-pos % size if count else 0) + size * count
    for _ in range(count):
        pos = _walk(node[1], body, pos, order, None)
    return pos


def _resolve(registry, type_):
    """Follow names and typedefs to a concrete type."""
    seen = set()
    while type_[0] == "named":
        _, name, modules = type_
        for table, kind in ((registry.typedefs, None),
                            (registry.structs, "struct"),
                            (registry.enums, "enum"),
                            (registry.unions, "union")):
            scoped = _lookup(table, name, modules)
            if scoped is not None:
                type_ = (kind, scoped) if kind else table[scoped]
                break
        else:
            raise McapError("unknown type {!r}".format(name))
        if scoped in seen:
            raise McapError("typedef {} is cyclic".format(scoped))
        seen.add(scoped)
    return type_

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
