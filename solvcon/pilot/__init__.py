# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Drawing and visualization sub-system of solvcon.
"""

# The "pilot" sub-package houses all GUI related code and should not be
# imported to the top-level "solvcon" namespace.

# Import _pilot_core first for C++ code.
from ._pilot_core import (  # noqa: F401
    enable,
    mgr,
    RDomainWidget,
    R2DWidget,
    Overlay2dOptions,
    RPythonConsoleDockWidget,
    RPythonTerminalDockWidget,
    RManager,
    PlotLimits2d,
    PlotColor,
    RPlotSeries,
    RPlotModel,
    RPlotTicker,
    plot_color_cycle,
    plot_cycle_color,
)
if enable:
    from .base._gui import (  # noqa: F401
        controller,
        launch,
        register_app,
    )
    from . import canvas  # noqa: F401
    from ._style import RuleCatalog  # noqa: F401

# The built-in applications live in the top-level "solvcon.apps" package,
# outside the pilot, and add themselves with register_app(); the pilot never
# imports them.

# NOTE: intentionally omit __all__ for now

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
