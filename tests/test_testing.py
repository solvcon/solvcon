# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest
from unittest import mock

from solvcon import pilot
from solvcon import testing


class _Case(unittest.TestCase):

    def test_nothing(self):
        pass


def _skip_reason(case):
    return getattr(case, '__unittest_skip_why__', None)


def _pilot_state(enabled):
    cause = None if enabled else ImportError('No module named PySide6')
    return mock.patch.multiple(pilot, enable=enabled, import_error=cause)


class RequiresPilotTC(unittest.TestCase):

    def test_keeps_case_with_pilot(self):
        with _pilot_state(True):
            case = testing.requires_pilot(type('C', (_Case,), {}))
        self.assertIsNone(_skip_reason(case))

    def test_skips_case_without_pilot(self):
        with _pilot_state(False):
            case = testing.requires_pilot(type('C', (_Case,), {}))
        self.assertTrue(case.__unittest_skip__)
        self.assertEqual('pilot is not available: No module named PySide6',
                         _skip_reason(case))


class RequiresLiveWindowTC(unittest.TestCase):

    def _decorate(self, enabled, no_window):
        with _pilot_state(enabled), \
                mock.patch.object(testing, 'NO_LIVE_WINDOW', no_window):
            return testing.requires_live_window(type('C', (_Case,), {}))

    def test_keeps_case_with_pilot_and_window(self):
        self.assertIsNone(_skip_reason(self._decorate(True, False)))

    def test_skips_case_without_window(self):
        reason = _skip_reason(self._decorate(True, True))
        self.assertEqual('pilot windows need a real window surface', reason)

    def test_names_missing_pilot_first(self):
        reason = _skip_reason(self._decorate(False, True))
        self.assertEqual('pilot is not available: No module named PySide6',
                         reason)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
