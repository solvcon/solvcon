# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The shock-tube app in a pilot off the screen, marched by hand instead of
by its timer.
"""

import unittest

import numpy as np

from solvcon import testing
from solvcon.apps import shocktube

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QApplication
except ImportError:
    pilot = None


@testing.requires_pilot
class AppTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # The app is built on the bare manager rather than through the
        # controller, which later tests expect to build last.
        cls.mgr = pilot.RManager.instance.setUp()
        cls.app = shocktube.ShockTubeApp(mgr=cls.mgr)
        cls.app.populate_menu()
        cls.item = cls.mgr.menu_model.action("app.shocktube")

    def setUp(self):
        # Held: a sub-window reached through a dropped wrapper of the MDI
        # area goes with it.
        self.area = self.mgr.mdiArea
        self.area.closeAllSubWindows()
        QApplication.processEvents()

    def tearDown(self):
        self.app.close()
        self.area.closeAllSubWindows()
        QApplication.processEvents()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def _open(self):
        self.item.trigger()
        QApplication.processEvents()

    def test_the_menu_opens_one_maximized_window(self):
        self.assertEqual("Shock tube", self.item.text())
        self._open()
        subwin = self.app.subwindow
        self.assertIn(subwin, self.area.subWindowList())
        self.assertIs(self.app.window, subwin.widget())
        self.assertTrue(subwin.isMaximized())
        self.assertEqual("Shock tube: Sod", subwin.windowTitle())
        # A second pick brings the open window back instead of a copy.
        subwin.showNormal()
        self._open()
        self.assertIs(subwin, self.app.subwindow)
        self.assertEqual(1, self.area.subWindowList().count(subwin))
        self.assertTrue(subwin.isMaximized())

    def test_the_window_draws_the_analytic_solution(self):
        self._open()
        self.app.st.svr.march_alpha2(steps=50)
        self.app.current_step = 50
        self.app.draw()
        plot = self.app.window.plot
        xs, values = plot.exact
        self.assertEqual((-10, 10), plot.xlim)
        # Ahead of the fan the gas is the left state, past the shock the
        # right one.
        self.assertEqual(1.0, values['density'][0])
        self.assertEqual(0.125, values['density'][-1])
        # The exact solver draws what the closed form of the legacy solver
        # gives for every quantity.
        self.app.st.build_field(t=2.5, coord=xs)
        for name in values:
            with self.subTest(name=name):
                np.testing.assert_allclose(
                    getattr(self.app.st, f"{name}_field"), values[name],
                    rtol=1.e-12, atol=1.e-12)
        self.assertEqual(['density', 'velocity', 'pressure'],
                         plot.quantities)
        # The fronts stand where the waves of Sod's problem, at the speeds
        # -1.1832, -0.0703, 0.9275, and 1.7522, have gone by t = 2.5.
        self.assertEqual(['head', 'tail', 'contact', 'shock', 'x0'],
                         [label for _, label, _ in plot.guides()])
        np.testing.assert_allclose(
            np.array([-1.1832, -0.0703, 0.9275, 1.7522],
                     dtype='float64') * 2.5,
            [x for x, _, _ in plot.guides()[:4]], atol=1.e-3)
        self.assertTrue(self.app.window.waves_text().startswith(
            "R-C-S    p* = 0.30313    u* = 0.92745"))
        self.app.window.fronts_chip.setChecked(False)
        self.assertEqual([], plot.guides())
        lo, hi = plot.yrange('density')
        self.assertLess(lo, 0.125)
        self.assertGreater(hi, 1.0)

    def test_the_xt_diagram_colors_a_row_per_step(self):
        self._open()
        window = self.app.window
        xt = window.xt
        self.assertEqual([('rarefaction', 2), ('contact', 1), ('shock', 1)],
                         [(kind, len(speeds)) for kind, speeds in xt.waves])
        self.assertAlmostEqual(2.5, xt.t_end)
        for _ in range(3):
            window.buttons['step'].click()
        # A pause redraws the same step without a row of its own.
        window.buttons['run'].click()
        window.buttons['run'].click()
        np.testing.assert_allclose([0.0, 0.05, 0.1, 0.15], xt.times)
        svr = self.app.st.svr
        np.testing.assert_array_equal(svr.density[svr.xindices], xt.rows[-1])
        # The first row holds the two states, the ends of the colors.
        self.assertEqual((0.125, 1.0), xt.vrange())
        words = xt.colors()
        self.assertEqual(xt.rows.shape, words.shape)
        self.assertTrue(np.all(words >> 24 == 0xff))
        self.assertNotEqual(words[0, 0], words[0, -1])
        self.assertFalse(xt.grab().isNull())
        window.buttons['reset'].click()
        self.assertEqual(1, len(xt.times))
        window.xt_chip.setChecked(False)
        self.assertTrue(xt.isHidden())

    def test_a_run_starts_at_t_zero_with_the_two_states(self):
        self._open()
        window = self.app.window
        self.assertEqual("Sod    t = 0", window.title())
        self.assertEqual("ready, step 0 / 50", window.state_text())
        xs, values = window.plot.points
        svr = self.app.st.svr
        np.testing.assert_array_equal(svr.coord[svr.xindices], xs)
        np.testing.assert_array_equal(np.where(xs < 0.0, 1.0, 0.125),
                                      values['density'])
        self.assertEqual(
            ("Run", True, False),
            (window.buttons['run'].text(),
             window.buttons['step'].isEnabled(),
             window.buttons['reset'].isEnabled()))

    def test_step_marches_the_solver_as_the_old_window_did(self):
        self._open()
        window = self.app.window
        for _ in range(3):
            window.buttons['step'].click()
        self.assertEqual("paused, step 3 / 50", window.state_text())
        self.assertEqual("Resume", window.buttons['run'].text())
        # The points are those of the solver marched three steps.
        tube = shocktube.ShockTube()
        tube.build_constant(gamma=1.4, pressure1=1.0, density1=1.0,
                            pressure5=0.1, density5=0.125)
        tube.build_numerical(xmin=-10, xmax=10, ncoord=201,
                             time_increment=0.05)
        tube.svr.march_alpha2(steps=3)
        np.testing.assert_array_equal(
            tube.svr.density[tube.svr.xindices],
            window.plot.points[1]['density'])
        self.assertAlmostEqual(0.15, float(window.title().split()[-1]))

    def test_the_timer_runs_to_the_last_step_and_stops(self):
        self._open()
        app, window = self.app, self.app.window
        window.buttons['run'].click()
        self.assertTrue(app.timer.isActive())
        self.assertEqual("Pause", window.buttons['run'].text())
        self.assertFalse(window.buttons['step'].isEnabled())
        for _ in range(100):
            if 'finished' == app.state:
                break
            app.timer.timeout.emit()
        self.assertFalse(app.timer.isActive())
        self.assertEqual("finished, step 50 / 50", window.state_text())
        self.assertEqual("Run again", window.buttons['run'].text())
        # Run again builds the run anew from t = 0 and marches it.
        window.buttons['run'].click()
        self.assertEqual(('running', 0), (app.state, app.current_step))
        window.buttons['reset'].click()
        self.assertFalse(app.timer.isActive())
        self.assertEqual("ready, step 0 / 50", window.state_text())

    def test_the_chips_switch_panels_but_keep_one(self):
        self._open()
        window = self.app.window
        window.chips['entropy'].setChecked(True)
        self.assertEqual(['density', 'velocity', 'pressure', 'entropy'],
                         window.plot.quantities)
        panels = window.plot.panels()
        self.assertEqual(4, len(panels))
        # Four panels take two columns, and the lower two label the x axis.
        self.assertEqual([False, False, True, True],
                         [bottom for _, _, bottom in panels])
        for name in ('density', 'velocity', 'pressure', 'entropy'):
            window.chips[name].setChecked(False)
        # The last panel cannot be switched off, or the window would go
        # blank with nothing saying why.
        self.assertEqual(['entropy'], window.plot.quantities)
        self.assertTrue(window.chips['entropy'].isChecked())
        # The panels paint in the colors of the palette.
        self.assertFalse(window.plot.grab().isNull())

    def test_the_config_panel_sets_the_problem_anew(self):
        self._open()
        dock = self.app.dock
        self.assertEqual("App config", dock.windowTitle())
        self.assertFalse(dock.isHidden())
        column = dock.widget()
        self.assertEqual((self.app.config, {}), column.read())
        self.app.window.buttons['step'].click()
        field = column.fields['max_steps']
        field.setText("20")
        field.textEdited.emit("20")
        model = self.app.solver_config
        row = [model.data(i, 0) for i in range(model.rowCount())].index(
            'max_steps')
        self.addCleanup(model.setData, row, 1, 50)
        # An edit waits for Apply, which starts over from t = 0.
        self.assertEqual("paused, step 1 / 50", self.app.window.state_text())
        column.apply_button.click()
        self.assertEqual("ready, step 0 / 20", self.app.window.state_text())
        self.assertEqual(20, self.app.config['max_steps'])

    def test_the_view_menu_toggles_the_config_panel(self):
        action = self.mgr.menu_model.action("panel.shocktube_config")
        self.assertIs(self.app.dock_action, action)
        # Without the window there is no panel to show.
        self.assertFalse(action.isEnabled())
        self._open()
        dock = self.app.dock
        self.assertTrue(action.isEnabled())
        self.assertTrue(action.isChecked())
        action.trigger()
        self.assertTrue(dock.isHidden())
        action.trigger()
        self.assertFalse(dock.isHidden())
        # A panel closed from its own bar unchecks the item.
        dock.close()
        self.assertFalse(action.isChecked())
        self.app.close()
        self.assertFalse(action.isEnabled())
        self.assertFalse(action.isChecked())

    def test_closing_the_window_stops_its_march(self):
        self._open()
        self.app.run_or_pause()
        self.assertTrue(self.app.timer.isActive())
        self.app.close()
        self.assertFalse(self.app.timer.isActive())
        self.assertIsNone(self.app.subwindow)
        self.assertIsNone(self.app.window)
        # The config panel goes with the window it edits.
        self.assertIsNone(self.app.dock)
        self._open()
        self.assertIn(self.app.subwindow, self.area.subWindowList())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
