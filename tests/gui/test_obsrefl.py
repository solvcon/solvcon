# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot.apps.obsrefl import ReflectionSession, _app
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

    @staticmethod
    def _status(feature):
        """The run readout of the panel, as ``{label: text}``."""
        run = feature._panel._run
        return {"step": run._progress.text(), "state": run._state.text()}

    def test_panel_opens_with_a_paused_preview(self):
        feature = self._feature()
        # Opening the panel opens the viewer on the configured run's
        # initial state, and the march waits for Start, Resume, or Step.
        sess = feature._control.session
        self.assertIsNotNone(sess)
        self.assertEqual(0, sess.step)
        self.assertTrue(feature._viewer.is_open)
        self.assertIsNotNone(self.mgr.currentR3DWidget())
        self.assertFalse(feature._control._timer.isActive())
        self.assertTrue(feature._panel._run._pause.isChecked())
        self.assertEqual("paused", self._status(feature)["state"])
        QApplication.processEvents()

    def test_start_builds_session_and_viewer(self):
        feature = self._feature()
        feature._panel._freestream._mach.setValue(2.5)
        feature._control.start()
        # Stop the timer so the heavy march does not run during the test.
        feature._control._timer.stop()
        self.assertIsInstance(feature._control.session, ReflectionSession)
        self.assertEqual(feature._control.session.step, 0)
        self.assertEqual(feature._control.session.shock.mach, 2.5)
        self.assertIsNotNone(self.mgr.currentR3DWidget())
        self.assertEqual(f"0 / {feature._control.session.max_steps}",
                         self._status(feature)["step"])

    @staticmethod
    def _coarsen(feature, nx=10, ny=4):
        """Cut the mesh down to what a window test can march quickly."""
        feature._panel._numerics._nx.setValue(nx)
        feature._panel._numerics._ny.setValue(ny)

    def test_the_resolution_reaches_the_mesh(self):
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._timer.stop()
        mesher = feature._control.session.shock.mesher
        self.assertEqual((10, 4), (mesher.nx, mesher.ny))
        # The viewer draws the mesh the spin boxes asked for, not the one
        # the preview was built on.
        self.assertEqual(feature._control.session.shock.mesh.ncell,
                         self.mgr.currentR3DWidget().mesh.ncell)
        QApplication.processEvents()

    def test_remesh_rebuilds_the_run_at_the_new_resolution(self):
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._on_step()
        self.assertGreater(feature._control.session.step, 0)
        self._coarsen(feature, nx=14, ny=6)
        feature._panel._numerics._remesh.click()
        # The mesh is fixed when a session is built, so a new resolution is
        # a new run: it waits on its initial state instead of marching on.
        mesher = feature._control.session.shock.mesher
        self.assertEqual((14, 6), (mesher.nx, mesher.ny))
        self.assertEqual(0, feature._control.session.step)
        self.assertFalse(feature._control._timer.isActive())
        self.assertTrue(feature._panel._run._pause.isChecked())
        self.assertEqual(feature._control.session.shock.mesh.ncell,
                         self.mgr.currentR3DWidget().mesh.ncell)
        QApplication.processEvents()

    def test_stop_ends_the_run_and_keeps_the_field(self):
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._timer.stop()
        feature._control._on_step()
        step = feature._control.session.step
        feature._panel._run._stop.click()
        sess = feature._control.session
        self.assertEqual('stopped', sess.stop_reason)
        self.assertEqual(step, sess.step)
        self.assertFalse(feature._control._timer.isActive())
        # The field the march reached is still on screen to be read.
        self.assertTrue(feature._viewer.is_open)
        self.assertEqual("stopped", self._status(feature)["state"])
        QApplication.processEvents()

    def test_reset_drops_the_run_and_the_viewer(self):
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._panel._run._reset.click()
        self.assertIsNone(feature._control.session)
        self.assertFalse(feature._viewer.is_open)
        self.assertFalse(feature._control._timer.isActive())
        self.assertFalse(feature._panel._run._viewer_btn.isChecked())
        self.assertEqual("not started", self._status(feature)["state"])
        # What was dropped is built again from the controls.
        feature._control.start()
        feature._control._timer.stop()
        self.assertIsNotNone(feature._control.session)
        self.assertTrue(feature._viewer.is_open)
        QApplication.processEvents()

    def test_reopening_the_viewer_draws_the_run_back(self):
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._timer.stop()
        feature._control._on_step()
        sess = feature._control.session
        feature._viewer.close()
        # Closing deletes the sub-window and the viewer inside it.  The run
        # is not the viewer's, so reopening has to draw it again.
        self.assertFalse(feature._viewer.is_open)
        self.assertIs(sess, feature._control.session)
        drawn = []
        real = feature._viewer.draw_field

        def spy(*args):
            drawn.append(args)
            real(*args)

        feature._viewer.draw_field = spy
        feature._panel._run._viewer_btn.click()
        self.assertTrue(feature._viewer.is_open)
        self.assertEqual(sess.shock.mesh.ncell,
                         self.mgr.currentR3DWidget().mesh.ncell)
        self.assertEqual(1, len(drawn))
        self.assertEqual(sess.step, feature._control.session.step)
        QApplication.processEvents()

    def test_a_closed_viewer_still_reports_the_run(self):
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._timer.stop()
        feature._viewer.close()
        feature._panel._run._stop.click()
        # The readout belongs to the run, not to the viewer that was drawing
        # it, so closing the viewer does not leave the panel stale.
        self.assertEqual("stopped", self._status(feature)["state"])
        QApplication.processEvents()

    def test_running_keeps_the_view_the_user_set(self):
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._timer.stop()
        viewer = self.mgr.currentR3DWidget()
        viewer.zoomCamera(3.0)
        zoom = viewer.cameraZoom
        self.assertNotAlmostEqual(1.0, zoom)
        # Neither marching a frame nor restarting the run at another
        # resolution may pull the view back to where it was framed.
        feature._control._on_step()
        self.assertAlmostEqual(zoom, viewer.cameraZoom)
        self._coarsen(feature, nx=12, ny=5)
        feature._control.start()
        feature._control._timer.stop()
        self.assertAlmostEqual(zoom, self.mgr.currentR3DWidget().cameraZoom)
        QApplication.processEvents()

    def test_start_sets_viewer_mesh_for_inspector(self):
        feature = self._feature()
        feature._control.start()
        feature._control._timer.stop()
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
        feature._panel._run._viewer_btn.setChecked(True)
        feature._control.start()
        feature._control._timer.stop()
        self.assertEqual(len(calls), 1)
        QApplication.processEvents()

    def test_step_advances_one_frame(self):
        feature = self._feature()
        feature._control.start()
        feature._control._timer.stop()
        feature._panel.set_paused(True)
        feature._panel._run._steps.setValue(3)
        feature._control._on_step()
        self.assertEqual(feature._control.session.step, 3)
        status = self._status(feature)
        self.assertEqual(f"3 / {feature._control.session.max_steps}",
                         status["step"])
        # The manual step happened under the pause, which still holds.
        self.assertEqual("paused", status["state"])

    def test_the_frame_timer_follows_the_session_to_its_end(self):
        feature = self._feature()
        feature._control.start()
        # The frame timer stops on the session's decision instead of counting
        # steps of its own.
        feature._control.session.stop()
        feature._control._advance()
        self.assertFalse(feature._control._timer.isActive())
        self.assertTrue(feature._panel._run._pause.isChecked())
        feature._control._draw_frame()
        self.assertEqual("stopped", self._status(feature)["state"])
        QApplication.processEvents()

    def test_pause_toggle_controls_timer_and_state(self):
        feature = self._feature()
        feature._control.start()
        # Pausing stops the frames, so no redraw reports it; the state cell
        # has to follow the button itself.
        feature._panel._run._pause.setChecked(True)
        self.assertFalse(feature._control._timer.isActive())
        self.assertEqual("paused", self._status(feature)["state"])
        feature._panel._run._pause.setChecked(False)
        self.assertTrue(feature._control._timer.isActive())
        self.assertEqual("running", self._status(feature)["state"])
        feature._control._timer.stop()

    def test_field_change_redraws_without_marching(self):
        feature = self._feature()
        feature._control.start()
        feature._control._timer.stop()
        feature._panel._run._pause.setChecked(True)
        step = feature._control.session.step
        feature._panel._field._selector.setCurrentText('pressure')
        # Picking a field recolors the current frame; it must not march.
        self.assertEqual(feature._control.session.step, step)
        self.assertEqual("pressure", feature._panel.field())

    def test_viewer_button_opens_and_closes_subwindow(self):
        feature = self._feature()
        # The preview already opened the viewer, and the button says so.
        self.assertTrue(feature._panel._run._viewer_btn.isChecked())
        self.assertTrue(feature._viewer.is_open)
        feature._panel._run._viewer_btn.setChecked(False)
        self.assertFalse(feature._viewer.is_open)
        feature._panel._run._viewer_btn.setChecked(True)
        self.assertTrue(feature._viewer.is_open)
        self.assertIsNotNone(self.mgr.currentR3DWidget())
        QApplication.processEvents()

    def test_closing_viewer_stops_run_without_drawing(self):
        feature = self._feature()
        feature._control.start()
        # Closing the domain sub-window while marching must stop the timer,
        # drop the viewer, and leave later frames as no-ops rather than
        # drawing into the freed widget.
        feature._viewer.close()
        self.assertFalse(feature._viewer.is_open)
        self.assertFalse(feature._control._timer.isActive())
        self.assertFalse(feature._panel._run._viewer_btn.isChecked())
        step = feature._control.session.step
        feature._control._advance()
        feature._control._draw_frame()
        self.assertEqual(feature._control.session.step, step)
        QApplication.processEvents()

    def test_start_reopens_a_closed_viewer(self):
        feature = self._feature()
        feature._control.start()
        feature._viewer.close()
        feature._control.start()
        feature._control._timer.stop()
        self.assertTrue(feature._viewer.is_open)
        self.assertTrue(feature._panel._run._viewer_btn.isChecked())
        QApplication.processEvents()

    def test_the_legend_stands_against_the_edge_the_panel_names(self):
        # The legend explains the colors, so it belongs over the field and
        # not in the control panel: it rides on the sub-window's host
        # widget, above the 3D view that fills it, against whichever edge
        # the Field box names.
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._timer.stop()
        host = feature._viewer._subwin.widget()
        host.resize(400, 300)
        QApplication.processEvents()
        for placement in ('left', 'right', 'upper', 'lower'):
            feature._panel._field._placement.setCurrentText(placement)
            bar = feature._viewer._bar
            self.assertIs(host, bar.parent())
            self.assertTrue(bar.isVisibleTo(host))
            # Left and right run the ramp up the view; the other two run it
            # across, and each sits against its own edge.
            self.assertEqual(placement in ('left', 'right'), bar.vertical)
            box = bar.geometry()
            if 'left' == placement:
                self.assertEqual(0, box.left())
            elif 'right' == placement:
                self.assertEqual(host.width(), box.right() + 1)
            elif 'upper' == placement:
                self.assertEqual(0, box.top())
            else:
                self.assertEqual(host.height(), box.bottom() + 1)
        QApplication.processEvents()

    def test_the_legend_is_pinned_to_the_field_the_viewer_colors(self):
        # The legend and the viewer have to be reading one range, or the
        # legend explains colors that are not on screen.  Each field pins
        # its own, so a field switched under the bar has to redraw it
        # rather than leave the range it was standing on.
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._timer.stop()
        sess = feature._control.session
        before = (feature._viewer._bar.lo, feature._viewer._bar.hi)
        feature._panel._field._selector.setCurrentText('pressure')
        bar = feature._viewer._bar
        self.assertNotEqual(before, (bar.lo, bar.hi))
        field = sess.field.field('pressure')
        self.assertEqual(
            sess.analysis.color_range('pressure', float(field.min()),
                                      float(field.max())),
            (bar.lo, bar.hi))
        QApplication.processEvents()

    def test_switching_the_legend_off_takes_it_away(self):
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._timer.stop()
        feature._panel._field._placement.setCurrentText('off')
        self.assertIsNone(feature._viewer._bar)
        # A run drawn with no legend must not go looking for one.
        feature._control._on_step()
        self.assertIsNone(feature._viewer._bar)
        # Naming an edge again brings it back on the standing scale.
        feature._panel._field._placement.setCurrentText('right')
        self.assertIsNotNone(feature._viewer._bar.lo)
        QApplication.processEvents()

    def test_the_reopened_viewer_carries_the_legend_back(self):
        feature = self._feature()
        self._coarsen(feature)
        feature._control.start()
        feature._control._timer.stop()
        feature._viewer.close()
        # Closing deletes the sub-window and the legend laid over it, so
        # reopening has to stand a new one up on the same scale.
        self.assertIsNone(feature._viewer._bar)
        feature._panel._run._viewer_btn.click()
        bar = feature._viewer._bar
        self.assertIsNotNone(bar)
        self.assertIs(feature._viewer._subwin.widget(), bar.parent())
        self.assertIsNotNone(bar.lo)
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
        sol._panel._run._viewer_btn.setChecked(True)
        QApplication.processEvents()
        sol._control.start()
        sol._control._timer.stop()
        QApplication.processEvents()
        root = ctl.tree_panel._mesh_tree._tree.topLevelItem(0)
        self.assertEqual(root.text(0), "StaticMesh (2D)")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
