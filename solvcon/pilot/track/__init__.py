# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The MCAP viewer of the pilot: a dock that lists the open recording and
a main window that tables one of its topics."""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_mcap_viewer': ['McapDock', 'McapMainWindow', 'McapPanel'],
})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
