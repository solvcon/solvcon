# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Agent Console: a dock that drives the 2D world from natural language.

The console sits at the bottom-right, beside the Python console, and runs the
selected AI backend on the active canvas world.  It reuses the headless
:class:`~solvcon.agent.AgentSession`, so the drawing logic stays Qt-free and
testable.  The session keeps the conversation, so each request replays the
turns before it, and one request may spend several backend steps: the panel
pumps a :class:`~solvcon.agent.Turn`, composing and applying on the Qt thread
and running only the call itself on a worker.
"""

from itertools import zip_longest

from PySide6.QtCore import Qt, QCoreApplication, QThread, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QDockWidget,
                               QLabel, QComboBox, QTextEdit, QLineEdit,
                               QPushButton)

from ...agent import (AgentBackend, AgentSession, BackendRegistry, StopReason,
                      Turn, op_of)
from ...config import Config
from . import _agent_control
from ._agent_settings import AgentBackendSettingsDialog
from ..base import _gui_common

__all__ = [  # noqa: F822
    'AgentBackendWorker',
    'AgentConsoleWidget',
    'AgentPanel',
]


class AgentBackendWorker(QThread):
    """Run one backend call off the Qt thread.

    Only the backend call runs here, the slow subprocess or HTTP round trip;
    it reads neither Qt nor the world, so it is safe off the main thread.  The
    request was frozen on the main thread before the thread started, so a
    canvas edited meanwhile cannot reach the call in flight.  The reply lands
    on :attr:`response` and the thread's own ``finished`` announces it, which
    orders the reply before the release of the worker that carried it.  The
    send goes through :meth:`~solvcon.agent.TurnRequest.send_safely`, so a
    backend that raises arrives as a transport outcome the turn can stop on
    rather than as an exception on a worker thread.
    """

    def __init__(self, backend, request, parent=None):
        super().__init__(parent)
        self._backend = backend
        self._request = request
        self.response = None

    def run(self):
        self.response = self._request.send_safely(self._backend)


class AgentConsoleWidget(QWidget):
    """The console body: a backend selector, a transcript, and a prompt box.

    Display-only.  It emits :attr:`submitted` with the typed prompt,
    :attr:`stop_requested` when the user asks to halt the running turn, and
    :attr:`settings_requested` when the user asks to configure the selected
    backend, and exposes the chosen backend; the owning feature runs the turn
    and calls back to append the reply.
    """

    submitted = Signal(str)
    stop_requested = Signal()
    settings_requested = Signal()

    def __init__(self, backends=(), parent=None):
        super().__init__(parent)
        self._backend_combo = QComboBox()
        for backend in backends:
            self._backend_combo.addItem(backend.name, backend)
        self._settings = QPushButton("Settings")

        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)

        self._status = QLabel("")
        self._status.setVisible(False)

        self._input = QLineEdit()
        self._input.setPlaceholderText(
            "Ask the agent to draw, arrange windows, or aim the view...")
        self._send = QPushButton("Send")
        self._stop = QPushButton("Stop")
        self._stop.setEnabled(False)
        self._stop.setToolTip("Stop the agent after the running step")

        selector = QHBoxLayout()
        selector.setContentsMargins(4, 2, 4, 2)
        selector.addWidget(QLabel("Backend:"))
        selector.addWidget(self._backend_combo, 1)
        selector.addWidget(self._settings)

        entry = QHBoxLayout()
        entry.setContentsMargins(4, 2, 4, 4)
        entry.addWidget(self._input, 1)
        entry.addWidget(self._send)
        entry.addWidget(self._stop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(selector)
        layout.addWidget(self._transcript, 1)
        layout.addWidget(self._status)
        layout.addLayout(entry)

        self._working_step = 0
        self._working_timer = QTimer(self)
        self._working_timer.setInterval(400)
        self._working_timer.timeout.connect(self._tick_working)

        self._input.returnPressed.connect(self._emit)
        self._send.clicked.connect(self._emit)
        self._stop.clicked.connect(self.stop_requested)
        self._settings.clicked.connect(self.settings_requested)

    def _emit(self):
        text = self._input.text().strip()
        if text:
            self.submitted.emit(text)

    def selected_backend(self):
        """The backend object behind the current selector entry, or ``None``
        when no backend is registered."""
        return self._backend_combo.currentData()

    def clear_input(self):
        self._input.clear()

    def set_busy(self, busy):
        """Lock the prompt and the settings button while a turn runs, so a
        second turn cannot overlap and an edit cannot look like it reaches the
        in-flight call, which already holds its configuration.  Stop is the one
        control that works only while busy."""
        self._input.setEnabled(not busy)
        self._send.setEnabled(not busy)
        self._settings.setEnabled(not busy)
        self._stop.setEnabled(busy)

    def start_working(self):
        """Show an animated ``working ...`` line while a turn runs, so a slow
        backend reads as busy rather than as a frozen, silent console."""
        self._working_step = 0
        self._status.setText("Agent is working .")
        self._status.setVisible(True)
        self._working_timer.start()

    def stop_working(self):
        """Hide the working line once the turn has finished."""
        self._working_timer.stop()
        self._status.clear()
        self._status.setVisible(False)

    def _tick_working(self):
        self._working_step = (self._working_step + 1) % 3
        self._status.setText(
            "Agent is working " + "." * (self._working_step + 1))

    def append_message(self, role, text):
        """Append one labelled block, e.g. ``You: ...`` or ``Agent: ...``."""
        label = {"user": "You", "agent": "Agent"}.get(role, role)
        self._transcript.append("%s: %s" % (label, text))


class AgentPanel(_gui_common.PilotFeature):
    """Agent Console dock, toggled from the View "Panels" submenu.

    It holds one :class:`~solvcon.agent.AgentSession` reused across prompts
    (the "current session"), rebinding it to the active world and the selected
    backend on each turn.  The session runs a dispatcher over the Agent Draw,
    Agent Window, and Agent View families, so a prompt can draw on the canvas,
    open or arrange windows, and aim the view through one tool surface.

    One prompt is a :class:`~solvcon.agent.Turn` this panel pumps a step at a
    time: :meth:`_pump` composes the request and starts a worker,
    :meth:`_on_step_finished` applies the reply and pumps the next step.  Only
    the backend call leaves the Qt thread, so reading the canvas and writing
    to it both stay where Qt allows them.
    """

    #: Backend calls one prompt may spend.  Enough for a plan, a repair after
    #: an error, and the empty batch that says the work is done, while a model
    #: that keeps proposing work still stops after four real calls.
    TURN_BUDGET = 4

    #: What to tell the user for a stop the transcript does not already
    #: explain by itself.  A turn the model ended (an empty batch or prose)
    #: needs no note: its own reply is the ending.
    _STOP_NOTES = {
        StopReason.NO_BACKEND: "(no backend selected)",
        StopReason.STOPPED: "(stopped)",
        StopReason.STATE: "(the canvas changed; the agent stopped)",
        StopReason.BUDGET: "(step budget reached; send again to continue)",
    }

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._panel = None
        self._session = AgentSession(
            runner=_agent_control.build_control_dispatcher(self._mgr))
        self._seams = _agent_control.PilotTurnSeams(self._mgr)
        self._config = Config.instance()
        BackendRegistry.load_settings(self._config)
        self._worker = None
        self._turn = None
        self._logged_payload = False
        # Make sure the worker thread is joined before the main thread exits.
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._join_worker)

    def _join_worker(self):
        """Halt the turn and wait for the worker before the main thread exits.

        Halting first cancels the call in flight, so quitting does not wait a
        slow CLI out, and keeps a reply delivered during shutdown from pumping
        the step after it.
        """
        self._halt_turn()
        if self._worker is not None:
            self._worker.wait()

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Agent Console", "Toggle the agent console panel",
            None, id="panel.agent_console", weight=40, checkable=True,
            checked=True)
        self._action.toggled.connect(self._on_toggled)
        # Shown by default, beside the Python console.
        self._ensure_panel()
        self._dock.show()

    def _on_toggled(self, checked):
        """Show or hide the panel."""
        if checked:
            self._ensure_panel()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        """Build the dock lazily and place it right of the Python console.

        Adding to the bottom area after the console (built at start-up in C++)
        lays the agent to the console's right, so the console keeps the
        bottom-left and the agent takes the bottom-right corner.  A split
        against the console dock would say this more explicitly, but that dock
        reaches Python as a pybind object PySide's splitDockWidget rejects, so
        insertion order is the available lever.
        """
        if self._panel is not None:
            return
        self._panel = AgentConsoleWidget(backends=BackendRegistry.available())
        self._panel.submitted.connect(self._on_submitted)
        self._panel.stop_requested.connect(self._halt_turn)
        self._panel.settings_requested.connect(self._on_settings_requested)
        self._dock = QDockWidget("Agent")
        self._dock.setWidget(self._panel)
        self._mainWindow.addDockWidget(Qt.BottomDockWidgetArea, self._dock)
        # Keep the menu check in sync when the dock is closed by its button.
        self._dock.visibilityChanged.connect(self._action.setChecked)

    def _on_settings_requested(self):
        """Configure the selected backend.  The settings live on the registry's
        backend instance, so an edit holds for every later turn, and the
        accepted values go to the configuration file so they also outlive the
        session."""
        backend = self._panel.selected_backend()
        if backend is None:
            return
        dialog = AgentBackendSettingsDialog(backend, parent=self._panel)
        if not dialog.exec():
            return
        BackendRegistry.save_settings(self._config)
        try:
            self._config.save()
        except OSError as exc:
            # The edit still holds for this session; only the file is lost, and
            # saying so beats a silent no-op the user finds out about later.
            self._panel.append_message(
                "agent", "settings not saved to %s: %s"
                % (self._config.path, exc))

    def _on_submitted(self, prompt):
        """Start one turn on the active canvas without blocking the GUI."""
        if self._turn is not None:
            return
        widget = self._mgr.currentR2DWidget()
        session = self._session
        session.backend = self._panel.selected_backend()
        # Each step rebinds through the scene seam anyway; binding here as
        # well puts the marker for a canvas switched since the last prompt
        # before that prompt in the transcript rather than after it.
        session.bind_world(None if widget is None else widget.world)
        self._panel.append_message("user", prompt)
        self._panel.clear_input()
        self._turn = Turn(session, prompt, budget=self.TURN_BUDGET,
                          scene=self._seams.scene)
        self._logged_payload = False
        self._panel.set_busy(True)
        self._panel.start_working()
        self._pump()

    def _pump(self):
        """Send the turn's next step, or finish the turn when it has none.

        Composing the request reads the canvas and the windows, so it stays
        here on the Qt thread; only the call crosses to the worker.
        """
        request = self._turn.next_request()
        if request is None:
            self._finish_turn()
            return
        if not self._logged_payload:
            # Only the first step is logged: every later one replays the same
            # conversation with the steps between appended, so logging each
            # would reprint the turn from the top once per step.
            self._logged_payload = True
            self._write_history_payload(self._session.backend, request)
        self._worker = AgentBackendWorker(
            self._session.backend, request, parent=self._panel)
        self._worker.finished.connect(self._on_step_finished)
        self._worker.start()

    def _on_step_finished(self):
        """Apply one step's reply and pump the next.

        The worker is released before the turn is fed, so the step this may go
        on to start is never the one being torn down.  The turn itself decides
        whether the reply is applied at all: a canvas that changed under the
        call, or a Stop the user hit meanwhile, drops it here.
        """
        worker, self._worker = self._worker, None
        response = worker.response
        worker.deleteLater()
        turn = self._turn.feed(response)
        if turn is not None:
            self._panel.append_message("agent", self._format_turn(turn))
        self._pump()

    def _halt_turn(self):
        """Halt the turn between steps and cancel the call in flight.

        The turn is ended first, so a reply landing after the cancel is dropped
        rather than applied; a backend with no cancel of its own simply runs
        its last call out, and its reply meets an ended turn.  Both the Stop
        button and application shutdown come through here.
        """
        if self._turn is None:
            return
        self._turn.stop()
        cancel = getattr(self._session.backend, "cancel", None)
        if cancel is not None:
            cancel()
        if self._worker is None:
            self._finish_turn()

    def _finish_turn(self):
        """Release the turn and re-enable the prompt for the next one."""
        turn, self._turn = self._turn, None
        note = self._STOP_NOTES.get(turn.stop_reason)
        if note:
            self._panel.append_message("agent", note)
        self._panel.stop_working()
        self._panel.set_busy(False)

    def _write_history_payload(self, backend, request):
        """Log the conversation about to be replayed to the backend.

        It goes to the Python console rather than the panel, whose transcript
        already carries these turns.  The section is asked of ``backend``, not
        of the base class, so a backend that composes its own payload logs the
        section it will really send; a duck-typed backend that offers none
        falls back to the standard layout.
        """
        compose = getattr(backend, "history_section",
                          AgentBackend.history_section)
        formatted = compose(request.prompt, request.scene_context,
                            request.tool_surface, request.history)
        if not formatted:
            return
        self._pycon.writeToHistory(
            "History (before send):\n%s\n\n" % formatted)

    @staticmethod
    def _turn_log_messages(turn):
        """The messages of the ``log`` commands that ran.

        A log whose command failed is dropped: its message describes work the
        turn went on to plan, and printing it as the reply would announce what
        never happened.
        """
        messages = []
        for command, result in zip_longest(turn.commands, turn.results):
            if op_of(command) != "log" or not getattr(result, "ok", False):
                continue
            message = command.get("message")
            if message is not None:
                messages.append(str(message))
        return messages

    @staticmethod
    def _turn_error_lines(turn):
        """One indented line per command that failed or never ran.  A command
        that worked stays silent: the canvas already shows it."""
        lines = []
        for command, result in zip_longest(turn.commands, turn.results):
            if result is None:
                lines.append("  - %s: not run" % op_of(command))
            elif not getattr(result, "ok", False):
                lines.append("  - %s: %s"
                             % (op_of(command),
                                getattr(result, "error", None) or "failed"))
        return lines

    @staticmethod
    def _format_turn(turn):
        """The user-facing reply: the ``log`` messages that ran (else backend
        prose), followed by a line per command that did not succeed, so a turn
        whose commands all failed cannot read as a turn that worked."""
        logs = AgentPanel._turn_log_messages(turn)
        lines = logs if logs else [turn.text or "(no reply)"]
        return "\n".join(lines + AgentPanel._turn_error_lines(turn))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
