# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Real Agent Draw, Window, and View targets, bound to the live pilot.

The Agent command families in :mod:`solvcon.agent` are Qt-free and
duck-typed: the draw commands mutate a ``World``, the window commands drive a
window-manager surface, the view commands drive a ``ViewTransform2dFp64``.
This module supplies the pilot-backed targets those families run against so an
agent can draw on, open, list, activate, close, and save real canvas windows
and steer the active canvas's view.  All three resolve the *active* canvas per
command, so one batch that opens a canvas and then draws on it lands on the
same window; :func:`build_control_dispatcher` wires them into one dispatcher.
"""

from ... import core
from ...agent import _command as _cmd
from ...agent import draw
from ...agent.window import Executor as WindowExecutor
from ...agent.window import view as _view

__all__ = [
    'PilotWindowManager',
    'LiveDrawExecutor',
    'LiveViewExecutor',
    'build_control_dispatcher',
    'pilot_scene_context',
    'PilotTurnSeams',
]


class PilotWindowManager:
    """Adapt the pilot ``RManager`` and its ``QMdiArea`` to the Agent Window
    manager surface.

    Presents the surface the window command family calls: ``new_canvas``,
    ``list_windows``, ``activate_window``, ``close_window``, and
    ``save_image``.  Each open sub-window carries a stable integer id, which
    dies with the window it names.  ``save_image`` briefly activates the target
    window so it can be grabbed through the current canvas, then restores the
    previously active one.
    """

    #: Dynamic property holding a sub-window's id.  The id rides on the C++
    #: object because PySide hands back a fresh Python sub-window for the same
    #: window from one listing to the next: held in a table keyed on that
    #: object, every listing would rename the windows under an agent that was
    #: shown the ids before it.
    ID_PROPERTY = "solvconAgentWindowId"

    #: Dynamic property holding the next id to hand out.  It sits on the MDI
    #: area, beside the ids themselves, so a second manager over the same area
    #: keeps counting rather than restarting at 1 over windows that already
    #: carry those ids.
    COUNTER_PROPERTY = "solvconAgentNextWindowId"

    def __init__(self, mgr):
        self._mgr = mgr

    def _area(self):
        return self._mgr.mdiArea

    def _open_subwindows(self):
        return [s for s in self._area().subWindowList() if s.isVisible()]

    def _mint_id(self):
        area = self._area()
        window_id = area.property(self.COUNTER_PROPERTY) or 1
        area.setProperty(self.COUNTER_PROPERTY, window_id + 1)
        return window_id

    def _id_of(self, subwin):
        window_id = subwin.property(self.ID_PROPERTY)
        if window_id is None:
            window_id = self._mint_id()
            subwin.setProperty(self.ID_PROPERTY, window_id)
        return window_id

    def _subwindow_of(self, window_id):
        for subwin in self._open_subwindows():
            if subwin.property(self.ID_PROPERTY) == window_id:
                return subwin
        return None

    def new_canvas(self):
        # A raw canvas has no world and an un-centered view; give it both, so
        # an agent-opened window matches the blank canvas the Canvas feature
        # builds and the agent draws on it without a separate bind step.
        widget = self._mgr.add2DWidget()
        widget.updateWorld(core.WorldFp64())
        widget.resetView()
        # add2DWidget activates the window it opens, so it is the active one.
        return self._id_of(self._area().activeSubWindow())

    def list_windows(self):
        active = self._area().activeSubWindow()
        subwins = self._open_subwindows()
        return [{"id": self._id_of(s),
                 "title": s.windowTitle() or "window",
                 "active": s is active}
                for s in subwins]

    def activate_window(self, window_id):
        subwin = self._subwindow_of(window_id)
        if subwin.isMinimized():
            subwin.showNormal()
        self._area().setActiveSubWindow(subwin)

    def close_window(self, window_id):
        self._subwindow_of(window_id).close()

    def save_image(self, window_id, path):
        subwin = self._subwindow_of(window_id)
        area = self._area()
        previous = area.activeSubWindow()
        area.setActiveSubWindow(subwin)
        try:
            widget = self._mgr.currentR2DWidget()
            # Only a 2D canvas can be grabbed; report anything else as a
            # failed write rather than a bogus success.
            if widget is None:
                return False
            return widget.saveImage(path)
        finally:
            if previous is not None and previous is not subwin:
                area.setActiveSubWindow(previous)


class _ActiveCanvasExecutor:
    """Mixin for a command executor bound to the *active* 2D canvas.

    Combined with a family ``Executor``, it resolves the active canvas per
    command and fails cleanly when there is none, seeds the command's target
    from that canvas through :meth:`_seed_target`, and calls
    :meth:`_on_change` after any command that mutates the canvas (every
    non-``read`` op).  Subclasses supply only those two hooks.
    """

    def __init__(self, mgr, *args, **kw):
        super().__init__(*args, **kw)
        self._mgr = mgr

    def _apply(self, op, command):
        widget = self._mgr.currentR2DWidget()
        if widget is None:
            return _cmd.CommandResult(op, False, error="no active 2D canvas")
        try:
            self.target = self._seed_target(widget)
        except _cmd.CommandError as exc:
            return _cmd.CommandResult(op, False, error=str(exc))
        result = super()._apply(op, command)
        if result.ok and self.command_set.commands[op].category != "read":
            self._on_change(widget)
        return result

    def _seed_target(self, widget):
        """The per-command target read from ``widget``; may raise
        ``CommandError`` to fail the command cleanly."""
        raise NotImplementedError

    def _on_change(self, widget):
        """React to a mutating command, e.g. write back and repaint."""


class LiveDrawExecutor(_ActiveCanvasExecutor, draw.Executor):
    """Bind the Agent Draw commands to the active 2D canvas's world.

    The world is resolved per command, so a batch that opens a canvas then
    draws on it targets the window it just opened; the canvas is repainted
    after any command that changes it.  A canvas with no world yet fails
    cleanly.
    """

    def __init__(self, mgr, renderer=None, validate_results=False,
                 reraise=False):
        super().__init__(mgr, None, renderer,
                         validate_results=validate_results, reraise=reraise)

    def _seed_target(self, widget):
        world = widget.world
        if world is None:
            raise _cmd.CommandError("active canvas has no world")
        return world

    def _on_change(self, widget):
        widget.requestRepaint()


class LiveViewExecutor(_ActiveCanvasExecutor, _view.Executor):
    """Bind the Agent View commands to the active 2D canvas.

    ``R2DWidget.viewTransform`` hands back a detached copy, so each command
    runs against a fresh copy seeded from the active canvas; a command that
    changes the view is written back with ``setViewTransform`` and the canvas
    is repainted, while read commands touch neither.
    """

    def __init__(self, mgr, validate_results=False, reraise=False):
        super().__init__(mgr, None, validate_results=validate_results,
                         reraise=reraise)

    def _seed_target(self, widget):
        return widget.viewTransform

    def _on_change(self, widget):
        widget.setViewTransform(self.target)
        widget.requestRepaint()


def build_control_dispatcher(mgr, renderer=None):
    """Assemble the draw, window, and view command families into one
    dispatcher bound to the live pilot.  Every family resolves the active
    canvas or the MDI area itself, so the caller needs no per-turn rebinding.
    """
    return _cmd.CommandDispatcher([
        LiveDrawExecutor(mgr, renderer),
        WindowExecutor(PilotWindowManager(mgr)),
        LiveViewExecutor(mgr)])


def pilot_scene_context(dispatcher, base):
    """Append a snapshot of the open windows and the active view to ``base``.

    A window a request never touched leaves no result to learn its id from, so
    the ids stay invisible unless the scene carries them; this lists every open
    window with its id and the active view so the model can target
    ``activate_window``, ``close_window``, or ``save_image`` without guessing.
    The read commands route through ``dispatcher`` so the ids match what the
    executors assign.
    """
    lines = [base]
    windows = dispatcher.run({"op": "list_windows"}).value["windows"]
    if windows:
        lines.append("windows: " + ", ".join(
            "#%d %r%s" % (w["id"], w["title"],
                          " (active)" if w["active"] else "")
            for w in windows))
    else:
        lines.append("windows: none open")
    view = dispatcher.run({"op": "get_view"})
    if view.ok:
        transform = view.value["view"]
        lines.append("view: pan (%g, %g), zoom %g" % (
            transform["pan_x"], transform["pan_y"], transform["zoom"]))
    return "\n".join(lines)


class PilotTurnSeams:
    """The scene a pilot turn is composed against, and with it the state the
    turn is guarded by.

    :class:`~solvcon.agent.Turn` calls :meth:`scene` once per step, on the
    thread that owns the canvas, and hands what it returns to its token.  The
    step starts from the canvas that is active *now*: a batch of the turn's
    own may have opened or activated one, and the executors above retarget on
    the spot, so a session left bound to the canvas the prompt started on
    would go on showing the model that world's shapes while the commands
    landed on another.

    Rebinding here is also what lets the turn keep the core's own
    :func:`~solvcon.agent.default_token`.  That token is the active canvas's
    world plus a digest of this scene, and this scene names every open window
    with its id, marks the active one, and prints the view, so a canvas
    switched, a window closed, or a pan or zoom the user made while the model
    was thinking all move it.  The tests in ``tests/gui/test_agent_control.py``
    pin that sensitivity, because it rests on what this scene carries.
    """

    def __init__(self, mgr):
        self._mgr = mgr

    def scene(self, session):
        """Point the session at the active canvas and describe it: the world's
        summary, then the live windows and view."""
        widget = self._mgr.currentR2DWidget()
        session.bind_world(None if widget is None else widget.world)
        return pilot_scene_context(session.runner, session.scene_context())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
