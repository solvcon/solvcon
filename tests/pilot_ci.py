# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Let CI run each pilot GUI test once per operating system.

Every pilot GUI test runs twice on a pull request: once under python plus
PySide6 (``make pytest BUILD_QT=ON``) and once inside the pilot binary
(``make run_pilot_pytest``), and the widget classes are where both passes
spend their time. The pilot binary embeds the interpreter nothing else
exercises, so it keeps the full GUI pass, and the python pass exports
``SOLVCON_SKIP_PILOT_WIDGET_TESTS`` to drop the widget classes while it still
runs the headless pilot classes and the whole non-GUI suite. A developer
running ``make pytest`` leaves the variable unset and sees no change.

The guard is an environment variable rather than a pytest marker because both
Qt hosts read it the same way, the test files stay pure unittest, and they
remain usable under plain ``python -m unittest``.
"""


import os


def _flag(name):
    return (os.getenv(name) or '').upper() in ('1', 'ON', 'YES', 'TRUE')


SKIP_PILOT_WIDGETS = _flag('SOLVCON_SKIP_PILOT_WIDGET_TESTS')

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
