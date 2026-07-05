# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import solvcon
try:
    from solvcon import pilot
except ImportError:
    pilot = None


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PilotTC(unittest.TestCase):

    def test_import(self):
        self.assertTrue(hasattr(solvcon.pilot, "mgr"))


class PythonConsoleBackendTC(unittest.TestCase):
    """
    The persistent read-eval-print backend behind the pilot console.

    These drive ``solvcon.apputil`` and ``solvcon.system`` directly, so
    they run headlessly and do not need the Qt pilot to be built.
    """

    def setUp(self):
        from solvcon.apputil import AppEnvironment
        self.env = AppEnvironment("test_{}".format(id(self)))

    def _run(self, source):
        """Run source in the test environment, capturing stdout and stderr."""
        import io
        from contextlib import redirect_stdout, redirect_stderr
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            more = self.env.run_code(source)
        return more, out.getvalue(), err.getvalue()

    def test_pycon(self):
        # A bare expression auto-displays and binds to _, the way a
        # read-eval-print loop behaves, rather than printing nothing.
        import builtins
        more, out, err = self._run('21 + 21')
        self.assertFalse(more)
        self.assertEqual(out.strip(), '42')
        self.assertEqual(builtins._, 42)
        _, out, err = self._run('_ + 8')
        self.assertEqual(out.strip(), '50')

    def test_assignment_is_silent(self):
        more, out, err = self._run('answer = 42')
        self.assertFalse(more)
        self.assertEqual(out, '')
        _, out, err = self._run('answer')
        self.assertEqual(out.strip(), '42')

    def test_multiline_reports_more_then_runs(self):
        import io
        from contextlib import redirect_stdout
        self.assertTrue(self.env.console.push('for i in range(3):'))
        self.assertTrue(self.env.console.push('    print(i)'))
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertFalse(self.env.console.push(''))
        self.assertEqual(out.getvalue(), '0\n1\n2\n')

    def test_exception_reports_user_traceback(self):
        _, out, err = self._run('raise ValueError("boom")')
        self.assertIn('ValueError: boom', err)
        self.assertIn('File "<console>"', err)
        # The host call stack must not leak into the user-facing report.
        self.assertNotIn('apputil', err)

    def test_syntax_error_reports_offending_line(self):
        _, out, err = self._run('def bad(:')
        self.assertIn('SyntaxError', err)
        self.assertIn('bad(', err)

    def test_exit_does_not_kill_process(self):
        # exit()/quit() raises SystemExit; the console swallows it so the
        # embedded interpreter that hosts the pilot keeps running.
        more, out, err = self._run('raise SystemExit(2)')
        self.assertFalse(more)

    def test_exec_code_entry_point_drives_the_console(self):
        # The C++ widget calls solvcon.system.exec_code; it must feed the
        # same persistent interpreter and auto-display the result.
        import io
        from contextlib import redirect_stdout
        from solvcon import system
        out = io.StringIO()
        with redirect_stdout(out):
            system.exec_code('6 * 7')
        self.assertEqual(out.getvalue().strip(), '42')


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class SetupProcessTC(unittest.TestCase):

    def test_namespace_includes_pilot(self):
        import builtins
        from solvcon import system, pilot
        system.setup_process([])
        self.assertIs(builtins.solvcon, solvcon)
        self.assertIs(builtins.sc, solvcon)
        self.assertIs(builtins.pilot, pilot)

    def test_broken_pilot_import_warns(self):
        import sys
        from unittest import mock
        from solvcon import system
        # Force "from . import pilot" to raise ImportError: drop the cached
        # attribute so the import re-runs, and poison sys.modules so it
        # fails. setup_process must warn instead of crashing.
        saved = solvcon.pilot
        del solvcon.pilot
        try:
            with mock.patch.dict(sys.modules, {'solvcon.pilot': None}):
                with self.assertWarns(UserWarning):
                    system.setup_process([])
        finally:
            solvcon.pilot = saved

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
