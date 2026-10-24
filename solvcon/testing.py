# Copyright (c) 2023, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import os
import unittest

import numpy as np

__all__ = [
    'TestBase',
    'NO_LIVE_WINDOW',
    'requires_pilot',
    'requires_live_window',
]


class TestBase:

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-12
        return np.testing.assert_allclose(*args, **kw)


# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


def _missing_pilot():
    # Import on use, so a test module that needs no GUI never loads the pilot.
    from . import pilot
    if pilot.enable:
        return None
    return f'pilot is not available: {pilot.import_error}'


def requires_pilot(obj):
    """Skip a test case or method when the pilot GUI is not available."""
    reason = _missing_pilot()
    if reason is None:
        return obj
    return unittest.skip(reason)(obj)


def requires_live_window(obj):
    """Skip unless the pilot runs and can map a real top-level window."""
    reason = _missing_pilot()
    if reason is not None:
        return unittest.skip(reason)(obj)
    if NO_LIVE_WINDOW:
        return unittest.skip('pilot windows need a real window surface')(obj)
    return obj

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
