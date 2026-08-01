# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Agent Console: a dock that drives the 2D world from natural language.

The console sits at the bottom-right, beside the Python console, and runs the
selected AI backend on the active canvas world for one request at a time.  It
reuses the headless :class:`~solvcon.agent.AgentSession`, so the drawing logic
stays Qt-free and testable.  The session keeps the conversation, so each
request replays the turns before it; driving several backend steps within one
request is a later addition.
"""

from itertools import zip_longest

from PySide6.QtCore import Qt, QCoreApplication, QThread, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QDockWidget,
                               QLabel, QComboBox, QTextEdit, QLineEdit,
                               QPushButton)

from ...agent import AgentBackend, AgentSession, BackendRegistry, op_of
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
    turns it replays are snapshotted on the main thread at construction, so a
    turn recorded meanwhile cannot land in the request being composed.  The
    reply returns through :attr:`succeeded` (a ``BackendResponse``) or
    :attr:`failed` (an error string); the owning panel applies the commands and
    repaints on the main thread, where the connected slots run.
    """

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, backend, prompt, scene_context, tool_surface,
                 history=(), parent=None):
        super().__init__(parent)
        self._backend = backend
        self._prompt = prompt
        self._scene_context = scene_context
        self._tool_surface = tool_surface
        self._history = list(history)

    def run(self):
        try:
            response = self._backend.send(
                self._prompt, self._scene_context, self._tool_surface,
                self._history)
        except Exception as exc:
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))
        else:
            self.succeeded.emit(response)


class AgentConsoleWidget(QWidget):
    """The console body: a backend selector, a transcript, and a prompt box.

    Display-only.  It emits :attr:`submitted` with the typed prompt and
    :attr:`settings_requested` when the user asks to configure the selected
    backend, and exposes the chosen backend; the owning feature runs the turn
    and calls back to append the reply.
    """

    submitted = Signal(str)
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

        selector = QHBoxLayout()
        selector.setContentsMargins(4, 2, 4, 2)
        selector.addWidget(QLabel("Backend:"))
        selector.addWidget(self._backend_combo, 1)
        selector.addWidget(self._settings)

        entry = QHBoxLayout()
        entry.setContentsMargins(4, 2, 4, 4)
        entry.addWidget(self._input, 1)
        entry.addWidget(self._send)

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
        in-flight call, which already holds its configuration."""
        self._input.setEnabled(not busy)
        self._send.setEnabled(not busy)
        self._settings.setEnabled(not busy)

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
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._panel = None
        self._session = AgentSession(
            runner=_agent_control.build_control_dispatcher(self._mgr))
        self._config = Config.instance()
        BackendRegistry.load_settings(self._config)
        self._worker = None
        self._active_widget = None
        # Make sure the worker thread is joined before the main thread exits.
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._join_worker)

    def _join_worker(self):
        """Wait for the worker to finish before the main thread exits."""
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
        if self._worker is not None:
            return
        widget = self._mgr.currentR2DWidget()
        session = self._session
        session.backend = self._panel.selected_backend()
        # The draw, window, and view executors resolve the active canvas on
        # their own; bind_world only keeps scene_context pointed at it.
        session.bind_world(None if widget is None else widget.world)
        self._panel.append_message("user", prompt)
        self._panel.clear_input()
        session.record_prompt(prompt)
        if session.backend is None:
            self._panel.append_message("agent", self._format_turn(None))
            return
        self._active_widget = widget
        self._panel.set_busy(True)
        self._panel.start_working()
        history = session.history()
        scene = _agent_control.pilot_scene_context(
            session.runner, session.scene_context())
        tool_surface = session.tool_surface()
        self._write_history_payload(session.backend, prompt, scene,
                                    tool_surface, history)
        self._worker = AgentBackendWorker(
            session.backend, prompt, scene, tool_surface,
            history, parent=self._panel)
        self._worker.succeeded.connect(self._on_backend_succeeded)
        self._worker.failed.connect(self._on_backend_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_backend_succeeded(self, response):
        """Apply the backend's commands to the world and repaint the canvas.
        """
        # TODO(#966): potential race condition here. capture a world revision
        # at submit and skip by-id commands when the world advanced.
        turn = self._session.complete_turn(response)
        self._panel.append_message("agent", self._format_turn(turn))
        widget = self._active_widget
        if widget is not None:
            try:
                widget.requestRepaint()
            except RuntimeError:
                # TODO: the widget may have been deleted while the backend was
                # running, so the repaint request fails. Need a better way to
                # track the widget's lifetime and avoid this.
                pass

    def _on_backend_failed(self, error):
        """Record a backend that raised as a failed agent turn."""
        turn = self._session.fail_turn(error)
        self._panel.append_message("agent", self._format_turn(turn))

    def _on_worker_finished(self):
        """Release the worker and re-enable the prompt for the next turn."""
        self._worker.deleteLater()
        self._worker = None
        self._active_widget = None
        self._panel.stop_working()
        self._panel.set_busy(False)

    def _write_history_payload(self, backend, prompt, scene, tool_surface,
                               history):
        """Log the conversation about to be replayed to the backend.

        It goes to the Python console rather than the panel, whose transcript
        already carries these turns.  The section is asked of ``backend``, not
        of the base class, so a backend that composes its own payload logs the
        section it will really send; a duck-typed backend that offers none
        falls back to the standard layout.
        """
        compose = getattr(backend, "history_section",
                          AgentBackend.history_section)
        formatted = compose(prompt, scene, tool_surface, history)
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
        if turn is None:
            return "(no backend selected)"
        logs = AgentPanel._turn_log_messages(turn)
        lines = logs if logs else [turn.text or "(no reply)"]
        return "\n".join(lines + AgentPanel._turn_error_lines(turn))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
