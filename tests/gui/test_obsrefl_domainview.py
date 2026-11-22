# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The 3D domain view of the reflection app, hosted in a plain widget
rather than in a sub-window of its own.

The inspector reads the view through the active MDI sub-window, so these
run in the window lane.  That the view draws is checked through the figure
in ``test_obsrefl_figure.py`` and through the app in ``test_app_obsrefl.py``.
"""

import unittest

import solvcon
from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
        from solvcon.apps.obsrefl import _domainview
except ImportError:
    pilot = None


def _make_mesh():
    """Two triangles and a quadrilateral in the z = 0 plane."""
    core = solvcon.core
    tri, quad = core.StaticMesh.TRIANGLE, core.StaticMesh.QUADRILATERAL
    mh = core.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
    mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
    mh.cltpn.ndarray[:] = [tri, tri, quad]
    mh.clnds.ndarray[:, :5] = [(3, 0, 3, 2, -1), (3, 0, 1, 3, -1),
                               (4, 1, 4, 5, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def _settle():
    for _ in range(20):
        QApplication.processEvents()


@testing.requires_live_window
class DomainViewTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()
        window = cls.mgr.mainWindow
        window.resize(1000, 800)
        window.show()
        _settle()

    def _hosted(self):
        """A domain view in a plain widget that is no sub-window."""
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        view = _domainview.DomainView(mgr=self.mgr)
        layout.addWidget(view)
        self.addCleanup(host.deleteLater)
        return host, view

    def test_the_inspector_finds_the_view(self):
        host, view = self._hosted()
        view.viewer.update_mesh(_make_mesh())
        # The mdiArea wrapper is thrown away right after use, taking any
        # sub-window handle reached through it, so it is held here.
        mdi = self.mdi = self.mgr.mdiArea
        subwin = mdi.addSubWindow(host)
        self.addCleanup(subwin.close)
        subwin.show()
        mdi.setActiveSubWindow(subwin)
        _settle()
        current = self.mgr.currentR3DWidget()
        self.assertIsNotNone(current)
        self.assertEqual(3, current.mesh.ncell)

    def test_escape_resets_the_camera_of_the_view(self):
        _, view = self._hosted()
        reset = self.mgr.menu_model.action("camera.reset")
        self.assertIsNotNone(reset)
        self.assertIn(reset, view.widget.actions())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
