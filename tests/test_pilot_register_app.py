# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the pilot registry the apps add themselves to.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
except ImportError:
    pilot = None


class _RecordingApp(object):
    def __init__(self, controller):
        self.controller = controller
        self.populated = False
        self.opened = 0

    def populate_menu(self):
        self.populated = True

    def open(self):
        self.opened += 1


@testing.requires_pilot
class RegisterAppTC(unittest.TestCase):
    @staticmethod
    def _fresh_controller():
        """An unbuilt controller, sidestepping the process-wide singleton."""
        ctl = object.__new__(_gui._Controller)
        ctl.__init__()
        return ctl

    @staticmethod
    def _unregister(name):
        """Keep the process-wide controller as the other tests expect."""
        del pilot.controller._app_factories[name]
        del pilot.controller.apps[name]

    def test_registered_before_build_starts_in_build(self):
        ctl = self._fresh_controller()
        built = []

        def factory(controller):
            built.append(_RecordingApp(controller))
            return built[-1]

        ctl.register_app("before", factory)
        self.assertEqual(built, [])
        self.assertEqual(ctl.apps, {})

        # Drive only the app step of build(), so the shared manager does not
        # get a second copy of every built-in menu entry.
        ctl._start_apps()
        self.assertEqual(len(built), 1)
        self.assertIs(ctl.apps["before"], built[0])
        self.assertIs(built[0].controller, ctl)
        self.assertTrue(built[0].populated)

    def test_registered_after_build_starts_at_once(self):
        mgr = pilot.controller.build()
        self.assertIs(pilot.controller.mgr, mgr)

        pilot.register_app("test_after_build", _RecordingApp)
        self.addCleanup(self._unregister, "test_after_build")
        app = pilot.controller.apps["test_after_build"]
        self.assertIs(app.controller, pilot.controller)
        self.assertTrue(app.populated)

    def test_a_name_registers_once(self):
        ctl = self._fresh_controller()
        ctl.register_app("twice", _RecordingApp)
        with self.assertRaisesRegex(ValueError, "already registered"):
            ctl.register_app("twice", _RecordingApp)

    def test_the_shelf_loads_an_app_by_name_and_returns_it(self):
        ctl = self._fresh_controller()
        ctl.register_app("one", _RecordingApp)
        ctl.register_app("two", _RecordingApp)
        ctl._start_apps()
        self.assertEqual(["one", "two"], dir(ctl.app_shelf))
        # Reading the loader opens nothing; load() opens the one app.
        loader = ctl.app_shelf.two
        self.assertEqual(0, ctl.apps["two"].opened)
        app = loader.load()
        self.assertIs(ctl.apps["two"], app)
        self.assertEqual((0, 1), (ctl.apps["one"].opened, app.opened))
        # A wrong name says which names there are.
        with self.assertRaisesRegex(AttributeError,
                                    "no app named 'three'.*one, two"):
            ctl.app_shelf.three
        # Each controller keeps its own apps.
        self.assertEqual([], dir(self._fresh_controller().app_shelf))

    def test_the_console_completes_and_opens_the_apps(self):
        from solvcon import apputil
        pilot.controller.build()
        # Another test may have made a newer environment the current one
        # since the build seeded the console.
        pilot.controller._seed_console_namespace()
        pilot.register_app("test_console", _RecordingApp)
        self.addCleanup(self._unregister, "test_console")
        app = pilot.controller.apps["test_console"]
        # A Tab lists an app registered after the console was seeded, and
        # opens none of what it lists.
        self.assertIn("apps.test_console",
                      apputil.get_completions("apps.test_"))
        self.assertIn("apps.test_console.load()",
                      apputil.get_completions("apps.test_console.lo"))
        self.assertEqual(0, app.opened)
        namespace = apputil.get_current_appenv().globals
        self.assertIn("test_console", dir(namespace["apps"]))
        self.assertIs(app, eval("apps.test_console.load()", namespace))
        self.assertEqual(1, app.opened)

    def test_mgr_is_none_before_build(self):
        self.assertIsNone(self._fresh_controller().mgr)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
