# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Expose common and operation-specific benchmark specification types."""

from . import matmul
from . import spec


__all__ = [
    'matmul',
    'spec',
]


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
