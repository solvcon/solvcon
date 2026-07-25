# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
solvcon: the description of the package is intentionally left blank
"""


# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


from . import core
from .core import *  # noqa: F401, F403
from . import apputil  # noqa: F401
from . import spacetime  # noqa: F401
from . import onedim  # noqa: F401
from . import multidim  # noqa: F401
from . import system  # noqa: F401
from . import testing  # noqa: F401
from . import toggle  # noqa: F401
from . import config  # noqa: F401
from . import track  # noqa: F401

clinfo = core.ProcessInfo.instance.command_line


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
