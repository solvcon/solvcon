# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot.apps.obsrefl import _app
    from PySide6.QtWidgets import QApplication
except ImportError:
    pilot = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
class ObliqueShockAppTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()

    def _feature(self):
        feature = _app.ObliqueShockApp(mgr=self.mgr)
        feature.populate_menu()
        feature._action.setChecked(True)  # builds the dock and panel
        return feature

    def test_panel_starts_paused_without_session(self):
        feature = self._feature()
        # Before Start there is no run, and the status tree says so.
        self.assertIsNone(feature._session)
        root = feature._panel._tree.topLevelItem(0)
        self.assertIn("not started", root.text(0))

    def test_start_builds_session_and_viewer(self):
        feature = self._feature()
        feature._panel._mach.setValue(2.5)
        feature._on_start()
        # Stop the timer so the heavy march does not run during the test.
        feature._session['timer'].stop()
        self.assertIsNotNone(feature._session)
        self.assertEqual(feature._session['step'], 0)
        self.assertIsNotNone(self.mgr.currentR3DWidget())
        sections = [feature._panel._tree.topLevelItem(i).text(0)
                    for i in range(feature._panel._tree.topLevelItemCount())]
        self.assertTrue(any(s.startswith("step: 0") for s in sections))

    def test_start_sets_viewer_mesh_for_inspector(self):
        feature = self._feature()
        feature._on_start()
        feature._session['timer'].stop()
        # The inspector reads the active viewer's mesh; the solver viewer must
        # carry the run's mesh so the mesh panel is not empty during a run.
        self.assertIsNotNone(self.mgr.currentR3DWidget().mesh)
        QApplication.processEvents()

    def test_start_notifies_viewer_updated(self):
        feature = self._feature()
        calls = []
        feature.viewer_updated = lambda: calls.append(1)
        # Opening the viewer first and then starting reuses it, which raises no
        # sub-window activation, so start must notify the inspector itself.
        feature._panel._viewer_btn.setChecked(True)
        feature._on_start()
        feature._session['timer'].stop()
        self.assertEqual(len(calls), 1)
        QApplication.processEvents()

    def test_step_advances_one_frame(self):
        feature = self._feature()
        feature._on_start()
        feature._session['timer'].stop()
        feature._panel.set_paused(True)
        feature._panel._steps.setValue(3)
        feature._on_step()
        self.assertEqual(feature._session['step'], 3)

    def test_pause_toggle_controls_timer(self):
        feature = self._feature()
        feature._on_start()
        feature._panel._pause.setChecked(True)
        self.assertFalse(feature._session['timer'].isActive())
        feature._panel._pause.setChecked(False)
        self.assertTrue(feature._session['timer'].isActive())
        feature._session['timer'].stop()

    def test_field_change_redraws_without_marching(self):
        feature = self._feature()
        feature._on_start()
        feature._session['timer'].stop()
        feature._panel._pause.setChecked(True)
        step = feature._session['step']
        feature._panel._field.setCurrentText('pressure')
        # Picking a field recolors the current frame; it must not march.
        self.assertEqual(feature._session['step'], step)
        root = feature._panel._tree.topLevelItem(1)
        self.assertIn("pressure", root.text(0))

    def test_viewer_button_opens_and_closes_subwindow(self):
        feature = self._feature()
        feature._panel._viewer_btn.setChecked(True)
        self.assertIsNotNone(feature._viewer)
        self.assertIsNotNone(self.mgr.currentR3DWidget())
        feature._panel._viewer_btn.setChecked(False)
        self.assertIsNone(feature._viewer)
        QApplication.processEvents()

    def test_closing_viewer_stops_run_without_drawing(self):
        feature = self._feature()
        feature._on_start()
        # Closing the domain sub-window while marching must stop the timer,
        # drop the viewer, and leave later frames as no-ops rather than
        # drawing into the freed widget.
        feature._close_viewer()
        self.assertIsNone(feature._viewer)
        self.assertFalse(feature._session['timer'].isActive())
        self.assertFalse(feature._panel._viewer_btn.isChecked())
        step = feature._session['step']
        feature._advance()
        feature._draw_frame()
        self.assertEqual(feature._session['step'], step)
        QApplication.processEvents()

    def test_start_reopens_a_closed_viewer(self):
        feature = self._feature()
        feature._on_start()
        feature._close_viewer()
        feature._on_start()
        feature._session['timer'].stop()
        self.assertIsNotNone(feature._viewer)
        self.assertTrue(feature._panel._viewer_btn.isChecked())
        QApplication.processEvents()


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
class SolutionInspectorTC(unittest.TestCase):
    """The wired controller refreshes the inspector when a run sets the
    mesh."""

    def test_open_viewer_then_start_populates_inspector(self):
        ctl = _gui.controller
        ctl.build()
        ctl.tree_panel._action.setChecked(True)
        sol = ctl.obsrefl_app
        sol._action.setChecked(True)
        # Open the viewer first, then start: the reused viewer raises no
        # activation, so only the wired refresh keeps the inspector from
        # standing on "No mesh loaded".
        sol._panel._viewer_btn.setChecked(True)
        QApplication.processEvents()
        sol._on_start()
        sol._session['timer'].stop()
        QApplication.processEvents()
        root = ctl.tree_panel._mesh_tree._tree.topLevelItem(0)
        self.assertEqual(root.text(0), "StaticMesh (2D)")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
