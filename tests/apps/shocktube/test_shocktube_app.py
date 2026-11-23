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


def _type(field, text):
    """Type ``text`` into the number ``field`` as the user would."""
    field.setText(text)
    field.textEdited.emit(text)


@testing.requires_pilot
class AppTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # The app is built on the bare manager rather than through the
        # controller, which later tests expect to build last.
        cls.mgr = pilot.RManager.instance.setUp()
        cls.app_config = pilot.panel.AppConfig(mgr=cls.mgr)
        cls.app_config.populate_menu()
        cls.app = shocktube.ShockTubeApp(mgr=cls.mgr,
                                         app_config=cls.app_config)
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
        return self.app.window, self.app.column, self.app.controller

    def _step(self, count):
        for _ in range(count):
            self.app.window.strip.buttons['step'].click()

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

    def test_a_run_starts_at_t_zero_with_the_two_states(self):
        window, _, ctl = self._open()
        strip = window.strip
        self.assertEqual("Sod, 201 points", strip.problem())
        self.assertEqual(("Ready", 'muted'), strip.pill())
        xs, values = window.figure.plot.points
        np.testing.assert_array_equal(np.where(xs < 0.0, 1.0, 0.125),
                                      values['density'])
        self.assertEqual("Sod, step 0, t = 0, waves R-C-S",
                         window.figure.caption.text())
        self.assertEqual((True, False),
                         (strip.buttons['step'].isEnabled(),
                          strip.buttons['reset'].isEnabled()))

    def test_step_marches_the_solver_as_the_old_window_did(self):
        window, _, ctl = self._open()
        self._step(3)
        self.assertEqual(("Paused", 'muted'), window.strip.pill())
        self.assertEqual("Resume", window.strip.buttons['run'].text())
        tube = shocktube.ShockTube()
        tube.build_constant(gamma=1.4, pressure1=1.0, density1=1.0,
                            pressure5=0.1, density5=0.125)
        tube.build_numerical(xmin=-10, xmax=10, ncoord=201,
                             time_increment=0.05)
        tube.svr.march_alpha2(steps=3)
        np.testing.assert_array_equal(
            tube.svr.density[tube.svr.xindices],
            window.figure.plot.points[1]['density'])
        self.assertAlmostEqual(0.15, ctl.session.time)

    def test_the_figure_draws_the_analytic_solution(self):
        window, _, ctl = self._open()
        self._step(50)
        plot = window.figure.plot
        xs, values = plot.exact
        self.assertEqual((-10.0, 10.0), plot.xlim)
        self.assertEqual(1.0, values['density'][0])
        self.assertEqual(0.125, values['density'][-1])
        # The exact solver draws what the closed form of the legacy solver
        # gives for every quantity.
        tube = ctl.session.tube
        tube.build_field(t=2.5, coord=xs)
        for name in values:
            with self.subTest(name=name):
                np.testing.assert_allclose(
                    getattr(tube, f"{name}_field"), values[name],
                    rtol=1.e-12, atol=1.e-12)
        # The fronts stand where the waves of Sod's problem, at the speeds
        # -1.1832, -0.0703, 0.9275, and 1.7522, have gone by t = 2.5.
        self.assertEqual(['head', 'tail', 'contact', 'shock', 'x0'],
                         [label for _, label, _ in plot.guides()])
        np.testing.assert_allclose(
            np.array([-1.1832, -0.0703, 0.9275, 1.7522],
                     dtype='float64') * 2.5,
            [x for x, _, _ in plot.guides()[:4]], atol=1.e-3)

    def test_the_exact_combo_draws_what_the_cells_hold(self):
        window, column, ctl = self._open()
        self._step(10)
        plot = window.figure.plot
        combo = column.display._exact
        combo.setCurrentIndex(combo.findData('cells'))
        xs, values = plot.exact
        centres = plot.points[0]
        self.assertEqual(2 * len(centres), len(xs))
        # Each step spans one cell around its centre and holds its mean, so
        # the steps add up to the mass in the tube.
        faces = np.append(xs[::2], xs[-1])
        density = values['density'][::2]
        mass = ctl.session.exact.integrate(faces[0], faces[-1], 0.5)[0]
        self.assertAlmostEqual(mass, np.sum(density * np.diff(faces)),
                               places=12)
        self.assertEqual("exact cell means",
                         window.figure.legend.entries[0][1])

    def test_the_xt_diagram_colors_a_row_per_step(self):
        window, column, ctl = self._open()
        xt = window.figure.xt
        self.assertEqual([('rarefaction', 2), ('contact', 1), ('shock', 1)],
                         [(kind, len(speeds)) for kind, speeds in xt.waves])
        self.assertAlmostEqual(2.5, xt.t_end)
        self._step(3)
        # A pause redraws the same step without a row of its own.
        window.strip.buttons['run'].click()
        window.strip.buttons['run'].click()
        np.testing.assert_allclose([0.0, 0.05, 0.1, 0.15], xt.times)
        self.assertEqual((0.125, 1.0), xt.vrange())
        self.assertFalse(xt.grab().isNull())
        window.strip.buttons['reset'].click()
        self.assertEqual(1, len(xt.times))
        column.display.option_boxes['xt'].setChecked(False)
        self.assertTrue(xt.isHidden())

    def test_the_timer_runs_to_the_last_step_and_stops(self):
        window, _, ctl = self._open()
        strip = window.strip
        strip.buttons['run'].click()
        self.assertTrue(ctl._timer.isActive())
        self.assertEqual("Pause", strip.buttons['run'].text())
        self.assertFalse(strip.buttons['step'].isEnabled())
        for _ in range(100):
            if 'finished' == ctl.state:
                break
            ctl._timer.timeout.emit()
        self.assertFalse(ctl._timer.isActive())
        self.assertEqual(("Finished", 'good'), strip.pill())
        self.assertEqual("Run again", strip.buttons['run'].text())
        # Run again builds the run anew from t = 0 and marches it.
        strip.buttons['run'].click()
        self.assertEqual(('running', 0), (ctl.state, ctl.session.step))
        strip.buttons['reset'].click()
        self.assertFalse(ctl._timer.isActive())
        self.assertEqual(('ready', 0), (ctl.state, ctl.session.step))

    def test_the_panel_boxes_switch_panels_but_keep_one(self):
        window, column, _ = self._open()
        boxes = column.display.quantity_boxes
        plot = window.figure.plot
        boxes['entropy'].setChecked(True)
        self.assertEqual(['density', 'velocity', 'pressure', 'entropy'],
                         plot.quantities)
        # Four panels take two columns, and the lower two label the x axis.
        self.assertEqual([False, False, True, True],
                         [bottom for _, _, bottom in plot.panels()])
        for name in ('density', 'velocity', 'pressure', 'entropy'):
            boxes[name].setChecked(False)
        # The last panel cannot be switched off, or the window would go
        # blank with nothing saying why.
        self.assertEqual(['entropy'], plot.quantities)
        self.assertTrue(boxes['entropy'].isChecked())
        self.assertFalse(plot.grab().isNull())

    def test_the_column_sets_the_problem_anew_on_reset(self):
        window, column, ctl = self._open()
        dock = self.app_config.dock
        self.assertEqual("App config", dock.windowTitle())
        self.assertIs(column, dock.widget().currentWidget())
        self._step(1)
        _type(column.problem.fields['p_right'], "0.01")
        # An edit waits for Reset, which the strip marks.
        self.assertEqual(1, ctl.session.step)
        self.assertEqual("true",
                         window.strip.buttons['reset'].property("attention"))
        self.assertTrue(column.problem.summary().endswith("(pending)"))
        window.strip.buttons['reset'].click()
        self.assertEqual(0, ctl.session.step)
        self.assertEqual(0.01, ctl.setup.p_right)
        self.assertEqual("Shock tube: rho 1 | 0.125, p 1 | 0.01, gamma 1.4",
                         self.app.subwindow.windowTitle())

    def test_a_wrong_value_keeps_reset_from_the_run(self):
        window, column, ctl = self._open()
        _type(column.numerics.fields['time_increment'], "0.2")
        reset = window.strip.buttons['reset']
        self.assertFalse(reset.isEnabled())
        self.assertIn("Courant", column.numerics.error_text())
        self.assertIn("Numerics", window.strip._failure.full_text)

    def test_the_card_judges_the_run(self):
        _, column, _ = self._open()
        self._step(50)
        card = column.card
        self.assertEqual('good', card.cell('states', 0, 2).tone)
        self.assertEqual("0.3031", card.cell('states', 0, 1).text)
        self.assertEqual("0.55", card.row_text("CFL"))

    def test_the_view_menu_toggles_the_config_panel(self):
        action = self.app_config.action
        # Without the window there is no panel to show.
        self.assertFalse(action.isEnabled())
        self._open()
        dock = self.app_config.dock
        self.assertTrue(action.isEnabled())
        self.assertTrue(action.isChecked())
        action.trigger()
        self.assertTrue(dock.isHidden())
        action.trigger()
        self.assertFalse(dock.isHidden())
        self.app.close()
        self.assertFalse(action.isEnabled())

    def test_closing_the_window_stops_its_march(self):
        _, _, ctl = self._open()
        ctl.run_or_pause()
        self.assertTrue(ctl._timer.isActive())
        self.app.close()
        self.assertFalse(ctl._timer.isActive())
        self.assertIsNone(self.app.subwindow)
        self.assertIsNone(self.app.window)
        # The column goes with the window it edits.
        self.assertIsNone(self.app_config.dock)
        self._open()
        self.assertIn(self.app.subwindow, self.area.subWindowList())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
