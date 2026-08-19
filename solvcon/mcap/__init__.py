# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
mcap: read MCAP recordings into solvcon.

The extension carries the subsystem only when it is configured with
``BUILD_MCAP=ON``, because decompressing MCAP chunks needs lz4 and zstd.
``HAS_MCAP`` is true only in such a build, and ``Reader`` exists only then.
"""

from .. import core

HAS_MCAP = core.HAS_MCAP

__all__ = [
    "HAS_MCAP",
]

if HAS_MCAP:
    Reader = core.McapReader
    __all__.append("Reader")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
