# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Agent Console: a dock that drives the 2D world from natural language.

The console sits at the bottom-right, beside the Python console, and runs the
selected AI backend on the active canvas world for one request at a time.  It
reuses the headless :class:`~solvcon.agent.AgentSession`, so the drawing logic
stays Qt-free and testable.  Multi-turn chat history is a later addition: each
prompt is an independent single turn.
"""

from itertools import zip_longest

from PySide6.QtCore import Qt, QCoreApplication, QThread, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QDockWidget,
                               QLabel, QComboBox, QTextEdit, QLineEdit,
                               QPushButton)

from ...agent import AgentSession, available_backends
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
    reply returns through :attr:`succeeded` (a ``BackendResponse``) or
    :attr:`failed` (an error string); the owning panel applies the commands and
    repaints on the main thread, where the connected slots run.
    """

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, backend, prompt, scene_context, tool_surface,
                 parent=None):
        super().__init__(parent)
        self._backend = backend
        self._prompt = prompt
        self._scene_context = scene_context
        self._tool_surface = tool_surface

    def run(self):
        try:
            response = self._backend.send(
                self._prompt, self._scene_context, self._tool_surface)
        except Exception as exc:
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))
        else:
            self.succeeded.emit(response)


class AgentConsoleWidget(QWidget):
    """The console body: a backend selector, a transcript, and a prompt box.

    Display-only.  It emits :attr:`submitted` with the typed prompt and exposes
    the chosen backend; the owning feature runs the turn and calls back to
    append the reply.
    """

    submitted = Signal(str)

    def __init__(self, backends=(), parent=None):
        super().__init__(parent)
        self._backend_combo = QComboBox()
        for backend in backends:
            self._backend_combo.addItem(backend.name, backend)

        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)

        self._status = QLabel("")
        self._status.setVisible(False)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask the agent to draw...")
        self._send = QPushButton("Send")

        selector = QHBoxLayout()
        selector.setContentsMargins(4, 2, 4, 2)
        selector.addWidget(QLabel("Backend:"))
        selector.addWidget(self._backend_combo, 1)

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
        """Lock the prompt while a turn runs so a second one cannot overlap."""
        self._input.setEnabled(not busy)
        self._send.setEnabled(not busy)

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
    backend on each turn.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._panel = None
        self._session = AgentSession()
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
        self._panel = AgentConsoleWidget(backends=available_backends())
        self._panel.submitted.connect(self._on_submitted)
        self._dock = QDockWidget("Agent")
        self._dock.setWidget(self._panel)
        self._mainWindow.addDockWidget(Qt.BottomDockWidgetArea, self._dock)
        # Keep the menu check in sync when the dock is closed by its button.
        self._dock.visibilityChanged.connect(self._action.setChecked)

    def _on_submitted(self, prompt):
        """Start one turn on the active canvas without blocking the GUI."""
        if self._worker is not None:
            return
        widget = self._mgr.currentR2DWidget()
        session = self._session
        session.backend = self._panel.selected_backend()
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
        self._worker = AgentBackendWorker(
            session.backend, prompt, session.scene_context(),
            session.tool_surface(), parent=self._panel)
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

    @staticmethod
    def _format_turn(turn):
        """One block of reply text followed by an indented line per command,
        marked ``ok`` or with its error."""
        if turn is None:
            return "(no backend selected)"
        lines = [turn.text or "(no reply)"]
        for command, result in zip_longest(turn.commands, turn.results):
            op = command.get("op", "?") if isinstance(command, dict) else "?"
            if result is None:
                lines.append("  - %s" % op)
            elif getattr(result, "ok", False):
                lines.append("  - %s: ok" % op)
            else:
                lines.append(
                    "  - %s: %s" % (op, getattr(result, "error", "failed")))
        return "\n".join(lines)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
