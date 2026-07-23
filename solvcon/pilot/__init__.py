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
)
if enable:
    from ._gui import (  # noqa: F401
        controller,
        launch,
    )
    from . import airfoil  # noqa: F401
    from . import canvas  # noqa: F401

# NOTE: intentionally omit __all__ for now

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
