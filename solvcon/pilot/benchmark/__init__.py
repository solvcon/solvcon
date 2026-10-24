# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Present benchmark inputs, worker progress, and results in Pilot."""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_inspector': ['BenchmarkInspector'],
})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
