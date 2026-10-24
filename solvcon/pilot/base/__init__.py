# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The base application layer: the launch and controller wiring, the shared
GUI helpers and feature base class, the 1D application scaffold, and the
theme menu.
"""

from .. import _pilot_core as _pcore

# _gui imports the feature neighborhoods, whose modules reach back here
# for the names of _gui_common and _base_app, so those must bind before _gui
# runs.
#
# _gui_common and _base_app need PySide6 but not the C++ pilot. Feature
# modules subclass PilotFeature and OneDimBaseApp at import time, and tests
# import those modules in builds without Qt, so bind the names whenever
# PySide6 imports rather than only when the pilot is enabled.
try:
    from . import _gui_common
    from . import _base_app
except ImportError:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    PilotFeature = None
    SubWindowCloseFilter = None
    apply_label_mode = None
    label_switch_and_mode = None
    OneDimBaseApp = None
    QuantityLine = None
    SolverConfig = None
else:
    PilotFeature = _gui_common.PilotFeature
    SubWindowCloseFilter = _gui_common.SubWindowCloseFilter
    apply_label_mode = _gui_common.apply_label_mode
    label_switch_and_mode = _gui_common.label_switch_and_mode
    OneDimBaseApp = _base_app.OneDimBaseApp
    QuantityLine = _base_app.QuantityLine
    SolverConfig = _base_app.SolverConfig

if _pcore.enable:
    from . import _theme  # noqa: F401
    from . import _gui
    controller = _gui.controller
    launch = _gui.launch
    register_app = _gui.register_app
else:
    controller = None
    launch = None
    register_app = None

__all__ = [
    'OneDimBaseApp',
    'PilotFeature',
    'QuantityLine',
    'SolverConfig',
    'SubWindowCloseFilter',
    'apply_label_mode',
    'controller',
    'label_switch_and_mode',
    'launch',
    'register_app',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
