# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Command panel: a dock that runs Python.

Every submission becomes a block in a scrolling transcript.  A Python block
holds the code and the output it printed.
"""

import ast
import codeop
import contextlib
import io
import sys
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFontDatabase, QKeySequence, QTextCursor
from PySide6.QtWidgets import (QDockWidget, QFrame, QHBoxLayout, QLabel,
                               QPlainTextEdit, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

from ... import apputil
from .. import base
from .. import _style

__all__ = [  # noqa: F822
    'CommandBlock',
    'CommandComposer',
    'CommandPanel',
    'CommandPanelWidget',
]

PYTHON = "python"

_DOT = "\u00b7"

#: Hues with no palette role, mixed toward the text color so they keep
#: their contrast under a light theme and a dark one.
_HUES = {"python": "#3d7cc9", "ok": "#2e7d32", "error": "#c62828",
         "warning": "#b8860b"}
_HUE_MIX = 0.25


class _PanelRules(_style.RuleCatalog):
    """The panel's style rules, one per role, all mixed from the palette."""

    @staticmethod
    def _hue(shades, name):
        return _style.Shades.blend(QColor(_HUES[name]), shades.text,
                                   _HUE_MIX).name()

    @classmethod
    def blocks(cls, shades):
        return (
            "QFrame#block { border: 1px solid %s; border-radius: 6px; }"
            "QFrame#block[kind=\"python\"] { border-left: 3px solid %s; }"
            "QLabel#badge { border-radius: 3px; padding: 0px 5px;"
            " color: %s; font-weight: bold; }"
            "QLabel#badge[kind=\"python\"] { background: %s; }"
            % (shades.border.name(), cls._hue(shades, "python"),
               shades.on_accent.name(), cls._hue(shades, "python")))

    @classmethod
    def outcomes(cls, shades):
        return (
            "QLabel#status[state=\"ok\"] { color: %s; }"
            "QLabel#status[state=\"error\"] { color: %s; }"
            "QLabel#error { color: %s; }"
            "QLabel#warning { color: %s; }"
            % (cls._hue(shades, "ok"), cls._hue(shades, "error"),
               cls._hue(shades, "error"), cls._hue(shades, "warning")))


def _mono():
    return QFontDatabase.systemFont(QFontDatabase.FixedFont)


def _keys(name):
    return QKeySequence("Ctrl+" + name).toString(
        QKeySequence.SequenceFormat.NativeText)


class _TranscriptLabel(QLabel):
    """A wrapping label that clips a run it cannot wrap, so one line of
    ``print('x' * 5000)`` does not widen the transcript past the dock."""

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint


def _transcript_label(text=""):
    # Plain text: a label left to guess renders "<b>x</b>" bold.
    label = _TranscriptLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setWordWrap(True)
    return label


def _repolish(widget):
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def python_is_complete(source):
    """Whether Enter should run ``source`` rather than open a new line.

    An open string, bracket, or block header always needs more.  A last
    statement holding a block stays open until a blank last line, as at the
    interactive prompt.  Source that does not compile counts as complete, so
    running it shows the error.
    """
    try:
        if codeop.compile_command(source, "<input>", "exec") is None:
            return False
    except (SyntaxError, OverflowError, ValueError):
        return True
    lines = source.split("\n")
    if len(lines) > 1 and not lines[-1].strip():
        return True
    body = ast.parse(source).body
    return not (body and (hasattr(body[-1], "body")
                          or hasattr(body[-1], "cases")))


class CellRuns(list):
    """The ``(is_error, text)`` runs a cell wrote, in order.  :attr:`raised`
    says whether it ended in an exception; stderr alone may be a warning."""

    raised = False


class _OrderedCapture(io.TextIOBase):
    """Append to runs shared by stdout and stderr, keeping their order."""

    encoding = "utf-8"

    def __init__(self, runs, is_error):
        super().__init__()
        self._runs = runs
        self._is_error = is_error

    def writable(self):
        return True

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("write() argument must be str, not %s"
                            % type(text).__name__)
        if text:
            if self._runs and self._runs[-1][0] == self._is_error:
                self._runs[-1][1].append(text)
            else:
                self._runs.append((self._is_error, [text]))
        return len(text)


def run_cell(source, appenv=None):
    """Run ``source`` as one cell in the pilot's namespace and return its
    output as :class:`CellRuns`.

    The input compiles as a module, so it runs the way it does from a file,
    and a trailing expression is shown through ``sys.displayhook``.
    """
    appenv = appenv or apputil.get_current_appenv()
    console = appenv.console
    runs = []
    raised = False
    with apputil.stdin_at_eof(), \
            contextlib.redirect_stdout(_OrderedCapture(runs, False)), \
            contextlib.redirect_stderr(_OrderedCapture(runs, True)):
        try:
            for refresh in appenv.namespace_refreshers:
                refresh(appenv.globals)
        except Exception:
            console.showtraceback()
            source = None
            raised = True
        tree = None
        try:
            if source is not None:
                tree = ast.parse(source, "<command>", "exec")
        except (SyntaxError, ValueError, OverflowError):
            console.showsyntaxerror("<command>")
            raised = True
        if tree is not None:
            raised = _exec_tree(tree, appenv.globals, console)
    result = CellRuns((is_error, "".join(parts)) for is_error, parts in runs)
    result.raised = raised
    return result


def _exec_tree(tree, namespace, console):
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    try:
        exec(compile(tree, "<command>", "exec"), namespace)
        if last is not None:
            sys.displayhook(eval(compile(last, "<command>", "eval"),
                                 namespace))
    except SystemExit:
        console.write("SystemExit: use the window controls to quit.\n")
    except BaseException:
        console.showtraceback()
    else:
        return False
    return True


class CommandComposer(QPlainTextEdit):
    """The multi-line input under the transcript.

    Enter runs the input when it is complete and otherwise opens an indented
    continuation line; Shift+Enter always opens a line and Ctrl+Enter always
    runs.
    """

    submitted = Signal(str)
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(_mono())
        self._fit()
        self.textChanged.connect(self._fit)

    def _fit(self):
        """Grow with the input, wrapped lines included, up to eight lines."""
        lines = max(1, min(8, int(self.document().size().height())))
        metrics = self.fontMetrics()
        margins = self.contentsMargins()
        height = (lines * metrics.lineSpacing() + 2 * self.document()
                  .documentMargin() + margins.top() + margins.bottom()
                  + 2 * self.frameWidth() + 4)
        self.setFixedHeight(int(height))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _newline(self):
        cursor = self.textCursor()
        line = cursor.block().text()
        indent = line[:len(line) - len(line.lstrip())]
        if line.rstrip().endswith(":"):
            indent += "    "
        cursor.insertText("\n" + indent)

    def keyPressEvent(self, event):
        key = event.key()
        # macOS flags the arrow keys as keypad keys.
        mods = event.modifiers() & ~Qt.KeypadModifier
        text = self.toPlainText()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if mods & Qt.ShiftModifier:
                self._newline()
            elif (mods & (Qt.ControlModifier | Qt.MetaModifier)
                  or python_is_complete(text)):
                if text.strip():
                    self.submitted.emit(text)
            else:
                self._newline()
            return
        if key == Qt.Key_L and mods & Qt.ControlModifier:
            self.clear_requested.emit()
            return
        super().keyPressEvent(event)

    def set_text(self, text):
        self.setPlainText(text)
        self.moveCursor(QTextCursor.End)


class CommandBlock(QFrame):
    """One submission in the transcript, with its outcome."""

    def __init__(self, kind, text, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("block")
        self.setProperty("kind", kind)
        self._started = time.perf_counter()

        badge = QLabel({PYTHON: "py"}.get(kind, kind))
        badge.setObjectName("badge")
        badge.setProperty("kind", kind)
        badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._status = QLabel("")
        self._status.setObjectName("status")
        self._status.setAlignment(Qt.AlignRight | Qt.AlignTop)

        header = QHBoxLayout()
        header.addWidget(badge, 0, Qt.AlignTop)
        self._input = _transcript_label(self._input_text(kind, text))
        self._input.setFont(_mono())
        header.addWidget(self._input, 1)
        header.addWidget(self._status, 0, Qt.AlignTop)

        self._body = QVBoxLayout()
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addLayout(self._body)

    @staticmethod
    def _input_text(kind, text):
        if kind != PYTHON:
            return text
        lines = text.rstrip("\n").split("\n")
        return "\n".join((">>> " if i == 0 else "... ") + line
                         for i, line in enumerate(lines))

    def elapsed(self):
        return time.perf_counter() - self._started

    def set_status(self, state, text=None):
        if text is None:
            text = "%s %.2f s" % (state, self.elapsed())
        self._status.setText(text)
        self._status.setProperty("state", state)
        _repolish(self._status)

    def add_output(self, text, error=False, warning=False):
        text = text.rstrip("\n")
        if not text:
            return
        label = _transcript_label(text)
        label.setFont(_mono())
        if error:
            label.setObjectName("error")
        elif warning:
            label.setObjectName("warning")
        self._body.addWidget(label)


class CommandPanelWidget(_style.PaletteStyle, QWidget):
    """The dock body: the block transcript over the composer.  It shows
    blocks and emits :attr:`composer`'s signals; :class:`CommandPanel` runs
    them."""

    #: 192 pixels of transcript over a 40 pixel composer.
    PREFERRED_HEIGHT = 192 + 40

    def __init__(self, parent=None):
        super().__init__(parent)

        self._blocks = QVBoxLayout()
        self._blocks.setContentsMargins(6, 6, 6, 6)
        self._blocks.setSpacing(6)
        self._blocks.addStretch(1)
        self._block_list = []
        holder = QWidget()
        holder.setLayout(self._blocks)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setWidget(holder)
        # Follow new output only while the reader is at the bottom.
        self._follow = True
        bar = self._scroll.verticalScrollBar()
        bar.valueChanged.connect(self._track_bottom)
        bar.rangeChanged.connect(self._keep_bottom)

        self.composer = CommandComposer()
        self.composer.setPlaceholderText("Python")
        self.run = QPushButton("Run")
        self._hint = QLabel((" %s " % _DOT).join(
            ["Enter runs", "Shift+Enter line", _keys("L") + " clears"]))
        self._hint.setEnabled(False)
        # The hint clips rather than widen the dock.
        self._hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        entry = QHBoxLayout()
        entry.setContentsMargins(6, 0, 6, 0)
        entry.addWidget(self.composer, 1)
        entry.addWidget(self.run, 0, Qt.AlignTop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._scroll, 1)
        layout.addLayout(entry)
        layout.addWidget(self._hint)

        self.run.clicked.connect(self._run_clicked)
        self._apply_style()

    def sizeHint(self):
        # An empty transcript has no height of its own to ask for.
        hint = super().sizeHint()
        hint.setHeight(max(hint.height(), self.PREFERRED_HEIGHT))
        return hint

    def _apply_style(self):
        self.setStyleSheet(_PanelRules.sheet(self, "blocks", "outcomes"))

    def _run_clicked(self):
        text = self.composer.toPlainText()
        if text.strip():
            self.composer.submitted.emit(text)

    def _track_bottom(self, value):
        self._follow = value >= self._scroll.verticalScrollBar().maximum()

    def _keep_bottom(self, lo, hi):
        if self._follow:
            self._scroll.verticalScrollBar().setValue(hi)

    def add_block(self, kind, text):
        self._follow = True
        block = CommandBlock(kind, text)
        self._blocks.insertWidget(self._blocks.count() - 1, block)
        self._block_list.append(block)
        return block

    def blocks(self):
        # Not from the layout: a wrapper of QLayout.itemAt() outlives its
        # item, and a later widget at that address can read as a QWidgetItem.
        return list(self._block_list)

    def clear(self):
        for block in self._block_list:
            block.setParent(None)
            block.deleteLater()
        self._block_list = []


class CommandPanel(base.PilotFeature):
    """The Command dock, toggled from the View "Panels" submenu.  Python
    runs in the pilot's shared interpreter, with the output captured into
    its block."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._panel = None

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Command", "Toggle the Python command panel",
            None, id="panel.command", weight=30, checkable=True)
        self._action.toggled.connect(self._on_toggled)

    @property
    def panel(self):
        return self._panel

    def _on_toggled(self, checked):
        if checked:
            self._ensure_panel()
            self._dock.show()
            # A dock in a tab group comes back behind the tab in front.
            self._dock.raise_()
            self._panel.composer.setFocus()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        if self._panel is not None:
            return
        self._panel = CommandPanelWidget()
        self._panel.composer.submitted.connect(self.submit)
        self._panel.composer.clear_requested.connect(self.clear)
        self._dock = QDockWidget("Command")
        self._dock.setWidget(self._panel)
        self._mainWindow.addDockWidget(Qt.BottomDockWidgetArea, self._dock)
        self._dock.visibilityChanged.connect(self._sync_action)

    def _sync_action(self, _visible):
        # The signal is also False for a dock tabbed behind another.
        self._action.setChecked(not self._dock.isHidden())

    def submit(self, text):
        self._panel.composer.clear()
        self.run_python(text)

    def run_python(self, source):
        """Run ``source`` in the shared interpreter and capture its output.
        It runs on the GUI thread, so a long command freezes the window."""
        source = source.rstrip()
        block = self._panel.add_block(PYTHON, source)
        runs = run_cell(source)
        for is_error, text in runs:
            block.add_output(text, error=is_error and runs.raised,
                             warning=is_error and not runs.raised)
        block.set_status("error" if runs.raised else "ok")
        return block

    def clear(self):
        self._panel.clear()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
