# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The sub-window a 2D domain is drawn into."""

from PySide6.QtCore import Qt, QObject, QEvent

from .. import base

from . import _colorizer

__all__ = [  # noqa: F822
    'DomainViewer',
]


class _ResizeRelay(QObject):
    """Report a watched widget's resize, so an overlay can follow it."""

    def __init__(self, on_resize, parent):
        super().__init__(parent)
        self._on_resize = on_resize

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self._on_resize()
        return False


class DomainViewer(object):
    """Manage the sub-window to draw in a spatial domain.

    The spatial domain is defined by a mesh
    (:class:`~solvcon.core.StaticMesh`). The viewer colors it with field data.
    """

    #: Lift the path overlay off the z = 0 field plane so it is not
    #: z-fought away by the colored triangles.
    PATH_LIFT = 0.01

    def __init__(self, mgr):
        self._mgr = mgr
        # Owner-supplied callbacks: `closed` fires when the sub-window
        # closes from any source, `mesh_updated` after the mesh is set.
        self.closed = None
        self.mesh_updated = None
        self._viewer = None
        self._subwin = None
        self._close_filter = None
        # Held for the viewer's lifetime: a throwaway mdiArea wrapper is
        # garbage-collected right after use, invalidating any sub-window
        # handle taken through it.
        self._mdi = None
        # The legend, its edge, and the scale it was last given.  The scale
        # outlives the widget, which is rebuilt whenever the edge turns it
        # through a right angle or the sub-window is closed under it.
        self._bar = None
        self._resize_relay = None
        self._placement = 'off'
        self._scale = (None, None)

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
            self._close_filter = base.SubWindowCloseFilter(
                self._on_subwin_closed, self._subwin)
            self._subwin.installEventFilter(self._close_filter)
            host = self._host()
            if host is not None:
                self._resize_relay = _ResizeRelay(self._place_bar, host)
                host.installEventFilter(self._resize_relay)
        self._place_bar()

    def _host(self):
        """The plain widget the 3D view is hosted in, which the legend is
        laid over, or None while the sub-window is closed."""
        return None if self._subwin is None else self._subwin.widget()

    def show_bar(self, placement):
        """Stand the legend against ``placement``, or take it away.

        :param placement: ``'left'`` or ``'right'`` stands the legend
            upright against that edge of the sub-window; ``'upper'`` or
            ``'lower'`` lays it across the top or bottom edge; ``'off'``
            takes it away.
        """
        self._placement = placement
        self._place_bar()

    def show_scale(self, lo, hi):
        """Give the legend the range it is to draw, whether or not it is
        standing; a bar built later opens on the last scale seen.

        :param lo: The field value at the low end of the color ramp.
        :param hi: The field value at the high end.  The legend stays blank
            when either end is None or ``hi`` is not above ``lo``.
        """
        self._scale = (lo, hi)
        if self._bar is not None:
            self._bar.show_scale(lo, hi)

    def _place_bar(self):
        """Build, move, or drop the legend to match the wanted placement."""
        host = self._host()
        if host is None or 'off' == self._placement:
            self._drop_bar()
            return
        vertical = self._placement in ('left', 'right')
        if self._bar is None or self._bar.vertical != vertical:
            self._drop_bar()
            self._bar = _colorizer.ColorBar(vertical=vertical)
            self._bar.show_scale(*self._scale)
        self._bar.setParent(host)
        self._bar.setGeometry(*self._bar_geometry(host))
        self._bar.show()
        # The 3D view fills the host, so the legend has to be told to sit
        # over it rather than under.
        self._bar.raise_()

    def _bar_geometry(self, host):
        """Where the legend sits against its edge of ``host``."""
        width, height = host.width(), host.height()
        thick = self._bar.thickness()
        if 'left' == self._placement:
            return (0, 0, thick, height)
        if 'right' == self._placement:
            return (width - thick, 0, thick, height)
        if 'upper' == self._placement:
            return (0, 0, width, thick)
        return (0, height - thick, width, thick)

    def _drop_bar(self):
        """Take the legend off the sub-window, keeping its scale."""
        if self._bar is not None:
            self._bar.setParent(None)
            self._bar = None

    def close(self):
        """Close the sub-window; its close event fires :attr:`closed`."""
        if self._subwin is not None:
            self._subwin.close()
        else:
            self._on_subwin_closed()

    def _on_subwin_closed(self):
        # Reached from the sub-window's close event; drop the widgets before
        # Qt frees them, then tell the owner so it can stop drawing.
        self._drop_bar()
        self._viewer = None
        self._subwin = None
        self._close_filter = None
        self._resize_relay = None
        if self.closed is not None:
            self.closed()

    def update_mesh(self, mesh):
        """Set the viewer mesh so the inspector can report it.

        Reusing an open viewer raises no sub-window activation, so the
        :attr:`mesh_updated` callback is what nudges the inspector.
        """
        if self._viewer is None:
            return
        self._viewer.updateMesh(mesh)
        if self.mesh_updated is not None:
            self.mesh_updated()

    def draw_path(self, path):
        """Overlay a polyline of two or three ``(x, y)`` corners.

        The 3D view's ruler draws it: two corners as one segment, three as
        two arms meeting at the middle corner with the angle between them
        annotated.  The ruler holds one path, so a new one replaces the
        last.

        :return: The length of the segment or the angle in degrees, or None
            while the sub-window is closed.
        """
        if not 2 <= len(path) <= 3:
            raise ValueError(
                f"a path takes two or three corners, not {len(path)}")
        if self._viewer is None:
            return None
        path = [(x, y, self.PATH_LIFT) for x, y in path]
        if 3 == len(path):
            return self._viewer.measureAngle(*path)
        return self._viewer.measureDistance(*path)

    def draw_field(self, verts, colors, indices):
        """Recolor the field triangles over the mesh."""
        if self._viewer is None:
            return
        self._viewer.updateColorField(verts, colors, indices)

    def capture(self, movie):
        """Hold one frame of the whole sub-window in ``movie``.

        The host is grabbed, not the 3D view inside it, so the color bar
        laid over the view is recorded with it.  A closed window has
        nothing to record, and records nothing rather than reaching a
        freed widget.
        """
        host = self._host()
        if host is None:
            return
        movie.capture(host)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
