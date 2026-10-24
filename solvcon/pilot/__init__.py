# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Drawing and visualization sub-system of solvcon.
"""

# The "pilot" sub-package houses all GUI related code and should not be
# imported to the top-level "solvcon" namespace.

from ._pilot_core import (  # noqa: F401
    enable,
    import_error,
    lazy_exports,
    require,
)

# NOTE: intentionally omit __all__ for now; binding the list lazy_exports()
# returns would make a star import resolve, and so require, every name.
__getattr__, __dir__ = lazy_exports(__name__, {
    '_pilot_core': [
        'mgr', 'RDomainWidget', 'R2DWidget', 'Overlay2dOptions',
        'RPythonConsoleDockWidget', 'RPythonTerminalDockWidget', 'RManager',
        'PlotLimits2d', 'PlotColor', 'RPlotSeries', 'RPlotModel',
        'RPlotTicker', 'plot_color_cycle', 'plot_cycle_color'],
    '_style': ['RuleCatalog'],
    'base': ['controller', 'launch', 'register_app'],
})[1:]

# The built-in applications live in the top-level "solvcon.apps" package,
# outside the pilot, and add themselves with register_app(); the pilot never
# imports them.

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
