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


class _FakeViewer:
    """A stand-in for a 3D viewer widget."""

    def __init__(self):
        self.mesh = None
        self.axis = False

    def updateMesh(self, mesh):
        self.mesh = mesh

    def showAxis(self, shown):
        self.axis = shown


class _FakeManager:
    """A duck-typed stand-in for the pilot manager."""

    def __init__(self):
        self._viewers = []

    def add3DWidget(self):
        viewer = _FakeViewer()
        self._viewers.append(viewer)
        return viewer

    def currentR3DWidget(self):
        return self._viewers[-1] if self._viewers else None

    def list3DWidgets(self):
        return list(self._viewers)


class PilotNamespaceTC(unittest.TestCase):
    """
    The curated console namespace and banner.

    Driven with a stand-in manager, so it runs headlessly and does not
    need the Qt pilot.
    """

    def setUp(self):
        from solvcon.apputil import AppEnvironment
        self.env = AppEnvironment("ns_{}".format(id(self)))
        self.mgr = _FakeManager()

    def test_curated_handles_are_seeded(self):
        from solvcon import apputil
        banner = apputil.install_pilot_namespace(self.mgr, self.env)
        g = self.env.globals
        self.assertIs(g['mgr'], self.mgr)
        self.assertIn('sc', g)
        self.assertIsNone(g['viewer'])
        self.assertIsNone(g['mesh'])
        self.assertTrue(callable(g['show_mesh']))
        self.assertIn('mgr', banner)
        self.assertIn('show_mesh(m)', banner)

    def test_show_mesh_opens_a_viewer(self):
        from solvcon import apputil
        apputil.install_pilot_namespace(self.mgr, self.env)
        sentinel = object()
        viewer = self.env.globals['show_mesh'](sentinel)
        self.assertIs(viewer.mesh, sentinel)
        self.assertTrue(viewer.axis)
        self.assertEqual(self.env.globals['viewers'](), [viewer])
        self.assertEqual(self.env.globals['meshes'](), [sentinel])

    def test_refresher_tracks_the_current_viewer(self):
        from solvcon import apputil
        apputil.install_pilot_namespace(self.mgr, self.env)
        # No viewer is open yet, so a command leaves the handles empty.
        self.env.run_code('pass')
        self.assertIsNone(self.env.globals['viewer'])
        # Opening a viewer must be reflected on the next command, not stay
        # stale at the value captured when the namespace was seeded.
        sentinel = object()
        self.mgr.add3DWidget().updateMesh(sentinel)
        self.env.run_code('pass')
        self.assertIs(self.env.globals['mesh'], sentinel)


class HousekeepingTC(unittest.TestCase):
    """The two latent bugs in the environment bookkeeping."""

    def test_get_current_appenv_returns_the_latest(self):
        from solvcon import apputil
        first = apputil.AppEnvironment("hk_first_{}".format(id(self)))
        latest = apputil.AppEnvironment("hk_latest_{}".format(id(self)))
        self.assertIsNot(first, latest)
        self.assertIs(apputil.get_current_appenv(), latest)

    def test_stop_code_removes_the_named_environment(self):
        from solvcon import apputil
        name = "hk_stop_{}".format(id(self))
        env = apputil.AppEnvironment(name)
        self.assertIn(name, apputil.environ)
        apputil.stop_code(env)
        self.assertNotIn(name, apputil.environ)


class CallTipTC(unittest.TestCase):
    """
    The call tip that the console shows when a ``(`` is typed.

    Driven through solvcon.apputil directly, so it runs headlessly.
    """

    def setUp(self):
        from solvcon.apputil import AppEnvironment
        self.env = AppEnvironment("tip_{}".format(id(self)))

    def test_builtin_signature_and_docstring(self):
        from solvcon import apputil
        tip = apputil.get_call_tip('range')
        self.assertTrue(tip.startswith('range('))
        self.assertIn('range', tip)

    def test_seeded_callable_signature_and_first_paragraph(self):
        from solvcon import apputil

        def greet(name, count=1):
            """Say hello to someone.

            This second paragraph must not appear in the tip.
            """
            return name

        self.env.seed(greet=greet)
        tip = apputil.get_call_tip('greet')
        self.assertIn('greet(name, count=1)', tip)
        self.assertIn('Say hello to someone.', tip)
        self.assertNotIn('second paragraph', tip)

    def test_non_callable_returns_empty(self):
        from solvcon import apputil
        self.env.seed(value=42)
        self.assertEqual(apputil.get_call_tip('value'), '')

    def test_unknown_name_returns_empty(self):
        from solvcon import apputil
        self.assertEqual(apputil.get_call_tip('does_not_exist'), '')

    def test_call_expression_is_not_evaluated(self):
        from solvcon import apputil

        def boom():
            raise AssertionError("the call tip must not invoke the callable")

        self.env.seed(boom=boom)
        # 'boom()' is not a bare identifier chain, so it is rejected before
        # any evaluation and boom is never called.
        self.assertEqual(apputil.get_call_tip('boom()'), '')


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
