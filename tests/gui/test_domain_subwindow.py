# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""How the domain viewer is laid out inside its MDI sub-window.

A sub-window only lays its contents out once it is on a live window, so the
geometry these check is not reachable from the widget lane: a hidden
sub-window keeps whatever size its widget was built with.
"""

import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6.QtWidgets import QApplication, QSizeGrip
except ImportError:
    pilot = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
class DomainSubWindowTC(unittest.TestCase):

    def _subwindow(self, width, height):
        """One 3D viewer sub-window of the given size, on a shown window."""
        mgr = pilot.RManager.instance.setUp()
        window = mgr.mainWindow
        window.resize(1000, 800)
        window.show()
        QApplication.processEvents()
        mgr.add3DWidget()
        # The mdiArea wrapper is thrown away right after use, taking any
        # sub-window handle reached through it, so it is held here.
        self.mdi = mgr.mdiArea
        subwin = self.mdi.activeSubWindow()
        subwin.resize(width, height)
        QApplication.processEvents()
        return subwin

    def test_the_grip_leaves_the_layout_to_the_viewer(self):
        # QMdiSubWindow adopts a size grip into the layout that holds the
        # viewer, where it would claim a row of the height.
        subwin = self._subwindow(816, 584)
        host = subwin.widget()
        layout = subwin.layout()
        self.assertEqual(1, layout.count())
        self.assertIs(host, layout.itemAt(0).widget())
        self.assertEqual(layout.contentsRect(), host.geometry())

    def test_the_grip_stays_drawn_in_the_corner(self):
        # The macOS style draws a grip only for a sub-window's own child; one
        # parented to the widget inside paints nothing.
        subwin = self._subwindow(816, 584)
        grip = subwin.findChild(QSizeGrip)
        self.assertIs(subwin, grip.parent())
        self.assertEqual(subwin.rect().bottomRight(),
                         grip.geometry().bottomRight())
        shot = grip.grab().toImage()
        drawn = {shot.pixelColor(ix, iy).rgb()
                 for ix in range(shot.width())
                 for iy in range(shot.height())}
        self.assertGreater(len(drawn), 1)

    def test_the_viewer_and_the_grip_follow_a_resize(self):
        # Qt does not move a grip that is out of the layout, so it follows
        # the corner by hand.
        subwin = self._subwindow(500, 300)
        short = subwin.widget().height()
        subwin.resize(500, 600)
        QApplication.processEvents()
        self.assertEqual(300, subwin.widget().height() - short)
        self.assertEqual(subwin.rect().bottomRight(),
                         subwin.findChild(QSizeGrip).geometry().bottomRight())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
