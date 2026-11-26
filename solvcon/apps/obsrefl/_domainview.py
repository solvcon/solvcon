# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The 3D domain view of the reflection app, hosted in a plain
widget."""

import shiboken6

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from ... import pilot
from ...pilot import visual

__all__ = [
    'DomainView',
]


def _handles():
    """Return the top-level widgets by the address of their Qt object."""
    return {shiboken6.getCppPointer(widget)[0]: widget
            for widget in QApplication.topLevelWidgets()}


def build_view():
    """Build a domain view and return ``(view, widget)``: the pybind11
    handle that draws, and the PySide handle a layout takes.

    The pilot binds the view through pybind11 and not through PySide, so
    the handle :class:`~solvcon.pilot.RDomainWidget` returns cannot go into
    a PySide layout.  The view is built without a parent, which makes it a
    top-level widget, and its PySide handle is the top-level that was not
    there before.

    :return: ``(None, None)`` where the pilot cannot build a view or the
        new view is not found among the top-level widgets.
    """
    before = _handles()
    try:
        view = pilot.RDomainWidget()
    except (AttributeError, RuntimeError):
        return None, None
    found = [widget for address, widget in _handles().items()
             if address not in before
             and widget.metaObject().className().endswith('RDomainWidget')]
    if 1 != len(found):
        return None, None
    return view, found[0]


class _HoverFilter(QObject):
    """Hand the moves of the cursor over the view to its host while no
    button is down; a drag still navigates the view and is not a hover.

    :param host: The :class:`DomainView` whose signals fire.
    """

    def __init__(self, host):
        super().__init__(host)
        self._host = host

    def eventFilter(self, watched, event):
        kind = event.type()
        if QEvent.MouseMove == kind and Qt.NoButton == event.buttons():
            pos = event.position().toPoint()
            self._host.hovered.emit(pos.x(), pos.y())
        elif QEvent.Leave == kind:
            self._host.left.emit()
        return False


class DomainView(QWidget):
    """A plain host of one 3D domain view, or of a muted note where the
    pilot cannot build one.

    The pilot builds a domain view into an MDI sub-window of its own
    (``RManager.add3DWidget``), which would make the figure a second
    window.  This widget builds the view without one and lays it into its
    own layout, so the view sits in the figure beside the color bar and
    under the profile.  The view is a ``QRhiWidget``, which composites only
    inside a plain container, never as the direct child of a sub-window;
    this widget is that container, as the plain host ``add3DWidget`` makes
    is.

    The inspector finds the view by searching the active sub-window for an
    ``RDomainWidget`` at any depth, so the view keeps reporting its mesh
    from in here.  The widget is a fold-back candidate for an ``RManager``
    call that builds a hosted view without a sub-window.

    :ivar view: The pybind11 handle of the view, or None.
    :ivar widget: The PySide handle of the view in the layout, or None.
    :param mgr: The pilot's :class:`~solvcon.pilot.RManager`, or None for
        its instance.
    """

    #: What the host reads where the pilot builds no view.
    UNAVAILABLE = "no 3D view"

    #: Fired with the pixel ``(x, y)`` of the view the cursor moved to
    #: while no button is down.
    hovered = Signal(int, int)
    #: Fired when the cursor leaves the view.
    left = Signal()

    def __init__(self, parent=None, mgr=None):
        super().__init__(parent)
        if mgr is None:
            mgr = pilot.RManager.instance
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.view, self.widget = build_view()
        #: Whether the view was built.
        self.available = self.view is not None
        #: The :class:`~solvcon.pilot.visual.DomainViewer` that draws into the
        #: view; without a view it draws nothing.
        self.viewer = visual.DomainViewer(mgr, view=self.view)
        if not self.available:
            note = QLabel(self.UNAVAILABLE)
            note.setObjectName("caption")
            note.setAlignment(Qt.AlignCenter)
            note.setEnabled(False)
            layout.addWidget(note)
            return
        layout.addWidget(self.widget)
        # The view moves the camera only while a button is down, so taking
        # the other moves leaves its navigation alone.
        self.widget.setMouseTracking(True)
        self._hover = _HoverFilter(self)
        self.widget.installEventFilter(self._hover)
        self.view.showAxis(True)
        self._bind_reset(mgr)

    def _bind_reset(self, mgr):
        """Let Escape reset the camera of this view, as ``add3DWidget``
        does for its own: the action is a widget shortcut and fires only
        from a widget it was added to.

        :param mgr: The pilot's :class:`~solvcon.pilot.RManager`, whose menu
            holds the action.
        """
        reset = mgr.menu_model.action("camera.reset")
        if reset is not None:
            self.widget.addAction(reset)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
