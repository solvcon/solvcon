# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The base application layer: the launch and controller wiring, the shared
GUI helpers and feature base class, the 1D application scaffold, and the
theme menu.
"""

from ... import pilot

# Features import these public names while _gui assembles the controller.
if pilot.enable:
    from . import _gui_common
    from . import _base_app

    PilotFeature = _gui_common.PilotFeature
    SubWindowCloseFilter = _gui_common.SubWindowCloseFilter
    apply_label_mode = _gui_common.apply_label_mode
    label_switch_and_mode = _gui_common.label_switch_and_mode
    OneDimBaseApp = _base_app.OneDimBaseApp
    QuantityLine = _base_app.QuantityLine
    SolverConfig = _base_app.SolverConfig

    from . import _theme  # noqa: F401
    from . import _gui

    controller = _gui.controller
    launch = _gui.launch
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    PilotFeature = None
    SubWindowCloseFilter = None
    apply_label_mode = None
    label_switch_and_mode = None
    OneDimBaseApp = None
    QuantityLine = None
    SolverConfig = None
    controller = None
    launch = None

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
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
