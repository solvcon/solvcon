# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Pytest plugin that runs the cyclic collector on nearly every allocation.

A wrapper that only the collector frees dies at an arbitrary point, so a
crash that depends on that timing shows up far more often under this plugin.
Load it with ``-p gcstress`` and this directory on ``PYTHONPATH``.
"""

import gc

gc.set_threshold(1, 1, 1)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
