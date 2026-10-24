# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Present benchmark inputs, worker progress, and results in Pilot."""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _inspector

    BenchmarkInspector = _inspector.BenchmarkInspector
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    BenchmarkInspector = None

__all__ = [
    'BenchmarkInspector',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
