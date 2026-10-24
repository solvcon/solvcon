# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import contextlib
import unittest
from unittest import mock

import solvcon
from solvcon import testing
try:
    from solvcon import pilot
except ImportError:
    pilot = None


@testing.requires_pilot
class PilotTC(unittest.TestCase):

    def test_import(self):
        self.assertTrue(hasattr(solvcon.pilot, "mgr"))


class PilotRequireTC(unittest.TestCase):

    def test_enable_implies_built_extension(self):
        # enable needs a Qt build, so it can only be on where HAS_PILOT is.
        if pilot.enable:
            self.assertTrue(solvcon.HAS_PILOT)
            self.assertIsNone(pilot.import_error)
        else:
            self.assertIsInstance(pilot.import_error, ImportError)

    def test_require_passes_when_enabled(self):
        from solvcon.pilot import _pilot_core
        with mock.patch.object(_pilot_core, 'enable', True):
            self.assertIsNone(pilot.require('the canvas'))

    def test_require_raises_with_cause(self):
        from solvcon.pilot import _pilot_core
        cause = ImportError('No module named PySide6')
        with mock.patch.object(_pilot_core, 'enable', False), \
                mock.patch.object(_pilot_core, 'import_error', cause):
            with self.assertRaises(ImportError) as cm:
                pilot.require('the canvas')
        self.assertEqual('the canvas needs solvcon built with Qt '
                         '(BUILD_QT=ON) and PySide6', str(cm.exception))
        self.assertIs(cm.exception.__cause__, cause)


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

    def test_push_line_drives_interactive_multiline(self):
        # push_line feeds one line at a time and reports whether the
        # statement is still open, without closing it on the caller's behalf,
        # so a terminal can show a continuation prompt.
        import io
        from contextlib import redirect_stdout
        # A complete statement closes at once.
        self.assertFalse(self.env.push_line('total = 0'))
        # An open block stays open until a blank line closes it.
        self.assertTrue(self.env.push_line('for i in range(3):'))
        self.assertTrue(self.env.push_line('    total += i'))
        self.assertFalse(self.env.push_line(''))
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertFalse(self.env.push_line('total'))
        self.assertEqual(out.getvalue().strip(), '3')

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
        self.boundaries = []

    def updateMesh(self, mesh):
        self.mesh = mesh

    def showAxis(self, shown):
        self.axis = shown

    def showBoundary(self, ibc, shown):
        if shown:
            self.boundaries.append(ibc)


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
        self.assertTrue(callable(g['load_mesh']))
        self.assertTrue(callable(g['sample_meshes']))
        self.assertTrue(callable(g['run_worker']))
        self.assertIn('mgr', banner)
        self.assertIn('show_mesh(m)', banner)
        self.assertIn('load_mesh(name)', banner)

    @testing.requires_pilot
    def test_show_mesh_opens_a_viewer(self):
        from solvcon import apputil
        apputil.install_pilot_namespace(self.mgr, self.env)
        mh = solvcon.mesh.RectDomainMesher(nx=2, ny=1).make_mesh(
            cell_type='quad')
        viewer = self.env.globals['show_mesh'](mh)
        self.assertIs(viewer.mesh, mh)
        self.assertTrue(viewer.axis)
        # The console draws like the menus, with a ribbon per group.
        self.assertEqual(viewer.boundaries, list(range(mh.nbcs)))
        self.assertEqual(self.env.globals['viewers'](), [viewer])
        self.assertEqual(self.env.globals['meshes'](), [mh])

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

    @testing.requires_pilot
    def test_sample_meshes_lists_the_known_names(self):
        from solvcon import apputil
        apputil.install_pilot_namespace(self.mgr, self.env)
        names = self.env.globals['sample_meshes']()
        self.assertIn('tetrahedron', names)
        self.assertIn('3dmix', names)

    @testing.requires_pilot
    def test_load_mesh_shows_a_named_sample(self):
        from solvcon import apputil
        apputil.install_pilot_namespace(self.mgr, self.env)
        viewer = self.env.globals['load_mesh']('tetrahedron')
        # A real StaticMesh is built and opened in a fresh viewer.
        self.assertEqual(viewer.mesh.ndim, 3)
        self.assertTrue(viewer.axis)
        self.assertEqual(self.env.globals['viewers'](), [viewer])

    @testing.requires_pilot
    def test_load_mesh_rejects_an_unknown_name(self):
        from solvcon import apputil
        apputil.install_pilot_namespace(self.mgr, self.env)
        with self.assertRaises(AttributeError):
            self.env.globals['load_mesh']('no_such_mesh')


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


class PythonCommandTC(unittest.TestCase):
    def test_uses_the_active_interpreter_and_preserves_arguments(self):
        from solvcon import system

        clinfo = mock.Mock(populated=False)
        with mock.patch.object(solvcon, 'clinfo', clinfo), \
                mock.patch.object(system.sys, 'executable', '/python path'):
            self.assertEqual(
                system.python_command('-c', 'print("hello world")'),
                ['/python path', '-c', 'print("hello world")'])

            clinfo.populated = True
            clinfo.populated_argv = ['/pilot path', '--mode=pytest', '-v']
            self.assertEqual(
                system.python_command('-m', 'solvcon.benchmark.worker'),
                ['/pilot path', '--mode=python', '-m',
                 'solvcon.benchmark.worker'])


class WorkerTC(unittest.TestCase):
    """The worker-thread helper for heavy console work."""

    def test_runs_off_the_calling_thread(self):
        import threading
        from solvcon import apputil
        caller = threading.get_ident()
        future = apputil.run_worker(threading.get_ident)
        self.assertNotEqual(future.result(timeout=5), caller)

    def test_returns_the_result(self):
        from solvcon import apputil
        future = apputil.run_worker(lambda a, b: a + b, 2, 3)
        self.assertEqual(future.result(timeout=5), 5)

    def test_propagates_exceptions_through_the_future(self):
        from solvcon import apputil

        def boom():
            raise ValueError("worker failed")

        future = apputil.run_worker(boom)
        with self.assertRaises(ValueError):
            future.result(timeout=5)


@testing.requires_pilot
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


class MissingPilotTC(unittest.TestCase):
    """
    What the process entry points say when the pilot cannot run.

    The pilot is patched off, so these run with or without a Qt build.
    """

    def _pilot_off(self, cause=None):
        from solvcon.pilot import _pilot_core
        stack = contextlib.ExitStack()
        for mod in (_pilot_core, solvcon.pilot):
            stack.enter_context(mock.patch.multiple(
                mod, enable=False, import_error=cause))
        return stack

    def test_disabled_pilot_warns_with_reason(self):
        from solvcon import system
        cause = ImportError('No module named PySide6')
        # setup_process only imports the pilot where the extension has it.
        with mock.patch.object(solvcon, 'HAS_PILOT', True), \
                self._pilot_off(cause):
            with self.assertWarnsRegex(UserWarning, 'No module named'):
                system.setup_process([])

    def test_run_pilot_names_the_missing_pilot(self):
        from solvcon import system
        with self._pilot_off():
            with self.assertRaisesRegex(
                    ImportError,
                    r'^the pilot application needs solvcon built with Qt '
                    r'\(BUILD_QT=ON\) and PySide6$'):
                system._run_pilot([])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
