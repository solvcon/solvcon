# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The MCAP viewer of the pilot: a dock that lists the open recording."""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _mcap_viewer

    McapDock = _mcap_viewer.McapDock
    McapPanel = _mcap_viewer.McapPanel
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    McapDock = None
    McapPanel = None

__all__ = [
    'McapDock',
    'McapPanel',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
