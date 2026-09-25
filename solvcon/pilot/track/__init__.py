# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The MCAP viewer of the pilot: a dock that lists the open recording and
a main window that tables one of its topics."""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _mcap_viewer

    McapDock = _mcap_viewer.McapDock
    McapPanel = _mcap_viewer.McapPanel
    McapMainWindow = _mcap_viewer.McapMainWindow
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    McapDock = None
    McapPanel = None
    McapMainWindow = None

__all__ = [
    'McapDock',
    'McapPanel',
    'McapMainWindow',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
