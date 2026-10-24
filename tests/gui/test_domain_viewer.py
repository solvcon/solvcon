# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The domain viewer on its own, with no app driving it."""

import math
import unittest

import solvcon
from solvcon import testing

try:
    from solvcon import pilot
    from solvcon.pilot import visual
    from PySide6.QtWidgets import QApplication
except ImportError:
    pilot = None


@testing.requires_live_window
class DomainViewerTC(unittest.TestCase):

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.viewer = visual.DomainViewer(self.mgr)
        self.events = []
        self.viewer.closed = lambda: self.events.append('closed')
        self.viewer.mesh_updated = lambda: self.events.append('mesh')

    def tearDown(self):
        self.viewer.close()
        QApplication.processEvents()

    @staticmethod
    def _mesh():
        core = solvcon.core
        mh = core.StaticMesh(ndim=2, nnode=3, nface=0, ncell=1)
        mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (0, 1)]
        mh.cltpn.ndarray[:] = [core.StaticMesh.TRIANGLE]
        mh.clnds.ndarray[:, :4] = [(3, 0, 1, 2)]
        mh.build_interior()
        mh.build_boundary()
        mh.build_ghost()
        return mh

    def test_open_is_idempotent(self):
        self.assertFalse(self.viewer.is_open)
        self.viewer.open()
        subwin = self.viewer._subwin
        self.assertTrue(self.viewer.is_open)
        self.viewer.open()
        self.assertIs(subwin, self.viewer._subwin)

    def test_a_close_from_qt_reaches_the_owner(self):
        self.viewer.open()
        self.viewer._subwin.close()
        QApplication.processEvents()
        self.assertFalse(self.viewer.is_open)
        self.assertEqual(['closed'], self.events)

    def test_closing_a_closed_viewer_still_tells_the_owner(self):
        self.viewer.close()
        self.assertEqual(['closed'], self.events)

    def test_a_closed_viewer_draws_nothing(self):
        self.viewer.update_mesh(self._mesh())
        self.assertIsNone(self.viewer.draw_path([(0, 0), (1, 0)]))
        self.viewer.draw_field(None, None, None)
        self.assertEqual([], self.events)

    def test_update_mesh_nudges_the_owner(self):
        self.viewer.open()
        self.viewer.update_mesh(self._mesh())
        self.assertEqual(['mesh'], self.events)

    def test_two_corners_draw_a_distance(self):
        self.viewer.open()
        self.assertAlmostEqual(5.0, self.viewer.draw_path([(0, 0), (3, 4)]))

    def test_three_corners_draw_the_angle_at_the_middle(self):
        self.viewer.open()
        angle = self.viewer.draw_path([(0, 1), (0, 0), (1, 0)])
        self.assertAlmostEqual(90.0, angle, places=4)
        angle = self.viewer.draw_path(
            [(-1, 0), (0, 0), (math.cos(math.pi / 3), math.sin(math.pi / 3))])
        self.assertAlmostEqual(120.0, angle, places=4)

    def test_a_path_the_ruler_cannot_draw_is_refused(self):
        for path in ([(0, 0)], [(0, 0), (1, 0), (1, 1), (0, 1)]):
            with self.assertRaises(ValueError):
                self.viewer.draw_path(path)

    def test_the_legend_follows_the_placement(self):
        self.viewer.open()
        host = self.viewer._subwin.widget()
        self.assertIsNone(self.viewer._bar)
        for placement, vertical in (('left', True), ('upper', False)):
            self.viewer.show_bar(placement)
            self.assertIs(host, self.viewer._bar.parent())
            self.assertEqual(vertical, self.viewer._bar.vertical)
        self.viewer.show_bar('off')
        self.assertIsNone(self.viewer._bar)

    def test_a_legend_built_later_opens_on_the_last_scale(self):
        self.viewer.show_scale(-1.0, 2.0)
        self.viewer.show_bar('right')
        self.assertIsNone(self.viewer._bar)
        self.viewer.open()
        self.assertEqual((-1.0, 2.0),
                         (self.viewer._bar.lo, self.viewer._bar.hi))

    def test_the_legend_goes_with_the_sub_window(self):
        self.viewer.show_bar('lower')
        self.viewer.open()
        self.assertIsNotNone(self.viewer._bar)
        self.viewer.close()
        QApplication.processEvents()
        self.assertIsNone(self.viewer._bar)

    def test_a_reopened_viewer_draws_again(self):
        # Closing frees the sub-window and the legend over it, so reopening
        # has to build both anew, the legend on the scale it last showed.
        self.viewer.show_scale(-1.0, 2.0)
        self.viewer.show_bar('right')
        self.viewer.open()
        self.viewer.close()
        QApplication.processEvents()
        self.viewer.open()
        self.assertTrue(self.viewer.is_open)
        self.assertAlmostEqual(5.0, self.viewer.draw_path([(0, 0), (3, 4)]))
        bar = self.viewer._bar
        self.assertIs(self.viewer._subwin.widget(), bar.parent())
        self.assertEqual((-1.0, 2.0), (bar.lo, bar.hi))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
