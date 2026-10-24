# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The base application layer: the launch and controller wiring, the shared
GUI helpers and feature base class, the 1D application scaffold, and the
theme menu.
"""

from .. import _pilot_core as _pcore

# _gui imports the feature neighborhoods, whose modules subclass PilotFeature
# and OneDimBaseApp in class statements, so these names must resolve before
# _gui runs. _gui_common and _base_app need PySide6 but not the C++ pilot,
# and tests import the feature modules in builds without it, so they are
# ungated.
__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_base_app': ['OneDimBaseApp', 'QuantityLine', 'SolverConfig'],
    '_gui': ['controller', 'launch', 'register_app'],
    '_gui_common': ['PilotFeature', 'SubWindowCloseFilter',
                    'apply_label_mode', 'label_switch_and_mode'],
}, ungated=('_base_app', '_gui_common'))

# Importing _gui creates the controller singleton and imports the feature
# neighborhoods, so a pilot is ready once this package loads.
if _pcore.enable:
    from . import _gui  # noqa: F401

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
