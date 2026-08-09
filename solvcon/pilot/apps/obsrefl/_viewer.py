# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The domain viewer sub-window a reflection run draws into.

:class:`DomainViewer` wraps the one 3D sub-window of a run: it opens and
closes the window, watches for a close from any source, and forwards what
the run wants drawn (the mesh, the analytic shock overlay, the colored
field).  Every drawing call is a no-op while the window is closed, so the
owner never draws into a freed widget.  What to draw and when stays with
the owner, which hears about a close through the :attr:`closed` callback.
"""

from PySide6.QtCore import Qt, QObject, QEvent

__all__ = [  # noqa: F822
    'DomainViewer',
]


class _SubWindowCloseFilter(QObject):
    """Report a watched sub-window's close synchronously.

    A ``QMdiSubWindow`` has no close signal, so this filter is the only way
    to stop the march before Qt frees the viewer.
    """

    def __init__(self, on_close, parent):
        super().__init__(parent)
        self._on_close = on_close

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Close:
            self._on_close()
        return False


class DomainViewer(object):
    """Own the domain sub-window and draw a run into it."""

    #: Lift the analytic shock overlay off the z = 0 field plane so it is
    #: not z-fought away by the colored triangles.
    PATH_LIFT = 0.01

    def __init__(self, mgr):
        self._mgr = mgr
        # Owner-supplied callbacks: `closed` fires when the sub-window
        # closes from any source, `mesh_updated` after a run sets the mesh.
        self.closed = None
        self.mesh_updated = None
        self._viewer = None
        self._subwin = None
        self._close_filter = None
        # Held for the viewer's lifetime: a throwaway mdiArea wrapper is
        # garbage-collected right after use, invalidating any sub-window
        # handle taken through it.
        self._mdi = None

    @property
    def is_open(self):
        """True while the sub-window is open; the close filter clears it."""
        return self._viewer is not None

    def open(self):
        """Open the sub-window if it is not already open."""
        if self._viewer is not None:
            return
        self._viewer = self._mgr.add3DWidget()
        self._viewer.showAxis(True)
        # Delete the sub-window on close and watch for that close, so a
        # close from any source reaches `closed` before Qt frees the widget.
        if self._mdi is None:
            self._mdi = self._mgr.mdiArea
        self._subwin = self._mdi.activeSubWindow()
        if self._subwin is not None:
            self._subwin.setAttribute(Qt.WA_DeleteOnClose, True)
            self._close_filter = _SubWindowCloseFilter(
                self._on_subwin_closed, self._subwin)
            self._subwin.installEventFilter(self._close_filter)

    def close(self):
        """Close the sub-window; its close event fires :attr:`closed`."""
        if self._subwin is not None:
            self._subwin.close()
        else:
            self._on_subwin_closed()

    def _on_subwin_closed(self):
        # Reached from the sub-window's close event; drop the widget before
        # Qt frees it, then tell the owner so it can stop the march.
        self._viewer = None
        self._subwin = None
        self._close_filter = None
        if self.closed is not None:
            self.closed()

    def update_mesh(self, mesh):
        """Set the viewer mesh so the inspector can report the run.

        Reusing an open viewer raises no sub-window activation, so the
        :attr:`mesh_updated` callback is what nudges the inspector.
        """
        if self._viewer is None:
            return
        self._viewer.updateMesh(mesh)
        if self.mesh_updated is not None:
            self.mesh_updated()

    def draw_shock_path(self, shock):
        """Overlay the analytic shock polyline on the viewer.

        The two-arm measurement draws the incident and the reflected shock
        as one path with the reflection angle annotated, so the computed
        field can be judged against where the shocks have to stand.
        """
        if self._viewer is None:
            return
        path = [(x, y, self.PATH_LIFT) for x, y in shock.shock_path()]
        if 3 == len(path):
            self._viewer.measureAngle(path[0], path[1], path[2])
        else:
            self._viewer.measureDistance(path[0], path[1])

    def draw_field(self, verts, colors, indices):
        """Recolor the field triangles of the current frame."""
        if self._viewer is None:
            return
        self._viewer.updateColorField(verts, colors, indices)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
