# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
mcap: read an MCAP recording into time-series columns.

``reader`` handles the file format and ``idl`` the ``ros2idl`` schema.
Every ``extract*`` call takes ``fields`` in three forms: ``None`` selects
every scalar leaf, a list of dotted paths selects those leaves, and a
``DecodePlan`` runs as given.
"""


class McapError(Exception):
    """A bad file, an unsupported feature, or a bad field selection."""


from .reader import Reader, Schema, Channel, Extraction  # noqa: E402
from .idl import parse_schema, DecodePlan  # noqa: E402

__all__ = [
    "McapError",
    "Reader",
    "Schema",
    "Channel",
    "Extraction",
    "parse_schema",
    "DecodePlan",
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
