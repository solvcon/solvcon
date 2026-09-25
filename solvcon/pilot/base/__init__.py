# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The base application layer: the launch and controller wiring, the shared
GUI helpers and feature base class, the 1D application scaffold, and the
theme menu.
"""

from .. import _pilot_core as _pcore

# _gui imports the feature neighborhoods, whose modules reach back here
# for _gui_common and _base_app, so those two must bind before _gui runs.
if _pcore.enable:
    from . import _gui_common
    from . import _base_app
    from . import _theme  # noqa: F401
    from . import _gui

    PilotFeature = _gui_common.PilotFeature
    OneDimBaseApp = _base_app.OneDimBaseApp
    controller = _gui.controller
    launch = _gui.launch
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    PilotFeature = None
    OneDimBaseApp = None
    controller = None
    launch = None

__all__ = [
    'OneDimBaseApp',
    'PilotFeature',
    'controller',
    'launch',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
