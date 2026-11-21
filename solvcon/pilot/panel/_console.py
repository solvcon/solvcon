# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Console panel: a command-line dock for Python and the agent.

The console panel provides console blocks for executing code and reporting the
results.
"""

import ast
import codeop
import contextlib
import io
import re
import sys
import time

from PySide6.QtCore import (Qt, QCoreApplication, QStringListModel, QTimer,
                            Signal)
from PySide6.QtGui import QColor, QFontDatabase, QKeySequence, QTextCursor
from PySide6.QtWidgets import (QApplication, QComboBox, QCompleter,
                               QDockWidget, QFrame, QHBoxLayout, QLabel,
                               QPlainTextEdit, QPushButton, QScrollArea,
                               QSizePolicy, QToolButton, QVBoxLayout,
                               QWidget)

from ... import apputil
from ...agent import BackendRegistry, StopReason, op_of
from .. import agent
from .. import base
from .. import _style

__all__ = [  # noqa: F822
    'ConsoleBlock',
    'ConsoleComposer',
    'ConsolePanel',
    'ConsolePanelWidget',
]

PYTHON = "python"
AGENT = "agent"
NOTE = "note"
PROMPTS = {PYTHON: ">>>", AGENT: "agent>"}

#: Output past these many lines or characters folds behind a toggle.
FOLD_LINES = 20
FOLD_CHARS = 2000

#: Output past this many characters keeps its head and tail only; a label
#: holding a megabyte of text stalls the window on every layout.
KEEP_CHARS = 100000

#: The transcript keeps this many blocks and each mode's history this many
#: entries; the oldest go first.
KEEP_BLOCKS = 500
KEEP_HISTORY = 1000

_CHECK = "\u2713"
_CROSS = "\u2717"
_DOT = "\u00b7"

#: Hues with no palette role, mixed toward the text color so they keep
#: their contrast under a light theme and a dark one.
_HUES = {"python": "#3d7cc9", "agent": "#d9822b", "ok": "#2e7d32",
         "error": "#c62828", "warning": "#b8860b"}
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
            "QFrame#block[kind=\"agent\"] { border-left: 3px solid %s; }"
            "QFrame#block[kind=\"python\"] { border-left: 3px solid %s; }"
            "QLabel#badge { border-radius: 3px; padding: 0px 5px;"
            " color: %s; font-weight: bold; }"
            "QLabel#badge[kind=\"agent\"] { background: %s; }"
            "QLabel#badge[kind=\"python\"] { background: %s; }"
            "QFrame#block[kind=\"note\"] { border-left: 3px solid %s; }"
            "QLabel#badge[kind=\"note\"] { background: %s; }"
            % (shades.border.name(), cls._hue(shades, "agent"),
               cls._hue(shades, "python"), shades.on_accent.name(),
               cls._hue(shades, "agent"), cls._hue(shades, "python"),
               shades.muted.name(), shades.muted.name()))

    @classmethod
    def outcomes(cls, shades):
        return (
            "QLabel#status[state=\"running\"] { color: %s; }"
            "QLabel#status[state=\"stopping\"] { color: %s; }"
            "QLabel#status[state=\"ok\"] { color: %s; }"
            "QLabel#status[state=\"error\"] { color: %s; }"
            "QLabel#status[state=\"stopped\"] { color: %s; }"
            "QLabel#error { color: %s; }"
            "QLabel#warning { color: %s; }"
            % (cls._hue(shades, "warning"), shades.muted.name(),
               cls._hue(shades, "ok"), cls._hue(shades, "error"),
               shades.muted.name(), cls._hue(shades, "error"),
               cls._hue(shades, "warning")))

    @classmethod
    def controls(cls, shades):
        return (
            "QToolButton#chip { font-weight: bold; padding: 2px 6px; }"
            "QToolButton#chip[mode=\"agent\"] { color: %s; }"
            "QToolButton#chip[mode=\"python\"] { color: %s; }"
            "QToolButton#action { padding: 1px 6px; }"
            % (cls._hue(shades, "agent"), cls._hue(shades, "python")))


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
            appenv.refresh_namespace()
        except Exception:
            _show_traceback(console)
            source = None
            raised = True
        tree = None
        try:
            if source is not None:
                tree = ast.parse(source, "<console>", "exec")
        except (SyntaxError, ValueError, OverflowError):
            console.showsyntaxerror("<console>")
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
        exec(compile(tree, "<console>", "exec"), namespace)
        if last is not None:
            sys.displayhook(eval(compile(last, "<console>", "eval"),
                                 namespace))
    except SystemExit:
        console.write("SystemExit: use the window controls to quit.\n")
    except BaseException:
        _show_traceback(console)
    else:
        return False
    return True


def _show_traceback(console):
    # A replaced sys.excepthook, such as Ubuntu's apport hook, takes the
    # report from console.write(), and Python 3.12 also hands it the host's
    # frames.
    hook, sys.excepthook = sys.excepthook, sys.__excepthook__
    try:
        console.showtraceback()
    finally:
        sys.excepthook = hook


def python_completions(prefix):
    """Sorted names that finish ``prefix``, without its dotted root, from the
    interpreter's namespace as the next command would see it."""
    appenv = apputil.get_current_appenv()
    # A failed refresh is reported when a cell runs; a key press stays quiet.
    with contextlib.suppress(Exception):
        appenv.refresh_namespace()
    # Looking up an attribute can run anything, such as a lazy import.
    try:
        names = apputil.get_completions(prefix)
    except Exception:
        return []
    root = prefix[:prefix.rfind(".") + 1]
    return sorted({name[len(root):] for name in names})


class ConsoleComposer(QPlainTextEdit):
    """The multi-line input under the transcript.

    Enter runs the input when it is complete (always, in agent mode) and
    otherwise opens an indented continuation line; Shift+Enter always opens a
    line and Ctrl+Enter always runs.
    """

    submitted = Signal(str)
    mode_changed = Signal(str)
    navigate = Signal(int)
    stop_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = PYTHON
        self.setFont(_mono())
        self._completer = QCompleter(QStringListModel(self), self)
        self._completer.setWidget(self)
        self._completer.popup().setFont(_mono())
        self._completer.activated[str].connect(self._insert_completion)
        self._fit()
        self.textChanged.connect(self._fit)

    def set_mode(self, mode):
        """Switch to :data:`PYTHON` or :data:`AGENT` input."""
        if mode == self.mode:
            return
        self.mode = mode
        self._completer.popup().hide()
        self.mode_changed.emit(mode)

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
        if self.mode == PYTHON and line.rstrip().endswith(":"):
            indent += "    "
        cursor.insertText("\n" + indent)

    def keyPressEvent(self, event):
        key = event.key()
        # macOS flags the arrow keys as keypad keys.
        mods = event.modifiers() & ~Qt.KeypadModifier
        text = self.toPlainText()
        if self._completer.popup().isVisible():
            if key == Qt.Key_Escape:
                self._completer.popup().hide()
                return
            is_enter = key in (Qt.Key_Return, Qt.Key_Enter)
            # Ignored, these go to the completer: Enter and Tab take the
            # choice, and Backtab closes the list.
            if (is_enter and not mods) or key in (Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return
            if is_enter:
                self._completer.popup().hide()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if mods & Qt.ShiftModifier:
                self._newline()
            elif (mods & (Qt.ControlModifier | Qt.MetaModifier)
                  or self.mode == AGENT or python_is_complete(text)):
                if text.strip():
                    self.submitted.emit(text)
            else:
                self._newline()
            return
        if key == Qt.Key_Escape:
            # Stop belongs to agent mode, like the Stop button.
            if self.mode == AGENT:
                self.stop_requested.emit()
            return
        if key == Qt.Key_L and mods & Qt.ControlModifier:
            self.clear_requested.emit()
            return
        if key == Qt.Key_I and mods & Qt.ControlModifier:
            self.set_mode(AGENT if self.mode == PYTHON else PYTHON)
            return
        # Holding Backspace to clear a prompt must not fall through into
        # Python by auto-repeat.
        if (not text and key == Qt.Key_Backspace and self.mode == AGENT
                and not event.isAutoRepeat()):
            self.set_mode(PYTHON)
            return
        if key == Qt.Key_Tab:
            # A tab character means nothing to the agent.
            if self.mode == PYTHON:
                if self._text_before_caret().strip():
                    self._show_completions(insert_single=True)
                else:
                    self.textCursor().insertText("    ")
            return
        # The cursor moves through the lines a long input wraps to, and only
        # past the first or the last does the key walk the history.
        if (key == Qt.Key_Up and not mods
                and not self.textCursor().movePosition(QTextCursor.Up)):
            self.navigate.emit(-1)
            return
        if (key == Qt.Key_Down and not mods
                and not self.textCursor().movePosition(QTextCursor.Down)):
            self.navigate.emit(1)
            return
        position = self.textCursor().position()
        super().keyPressEvent(event)
        # A bare modifier also arrives here, and must keep the chosen row.
        if (self._completer.popup().isVisible()
                and (self.toPlainText() != text
                     or self.textCursor().position() != position)):
            self._show_completions(insert_single=False)

    def _text_before_caret(self):
        cursor = self.textCursor()
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        return cursor.selectedText()

    def _show_completions(self, insert_single):
        """Offer the names that finish the word behind the caret.  A leading
        dot follows an expression like ``f().``, which is not evaluated."""
        popup = self._completer.popup()
        prefix = re.search(r"[\w.]*$", self._text_before_caret()).group()
        names = ([] if not prefix or prefix[0] == "." or ".." in prefix
                 else python_completions(prefix))
        word = prefix.rpartition(".")[2]
        # A list of just the finished word would take the Enter that runs the
        # line; a name may end in "(", "()", " ", or ":".
        finished = [name.rstrip("(): ") for name in names] == [word]
        if not names or (not insert_single and finished):
            popup.hide()
            return
        self._completer.model().setStringList(names)
        self._completer.setCompletionPrefix(word)
        if insert_single and len(names) == 1:
            self._insert_completion(names[0])
            return
        popup.setCurrentIndex(self._completer.completionModel().index(0, 0))
        rect = self.cursorRect()
        rect.setWidth(popup.sizeHintForColumn(0)
                      + popup.verticalScrollBar().sizeHint().width())
        self._completer.complete(rect)

    def _insert_completion(self, name):
        cursor = self.textCursor()
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor,
                            len(self._completer.completionPrefix()))
        cursor.insertText(name)
        self.setTextCursor(cursor)

    def set_text(self, text):
        self.setPlainText(text)
        self.moveCursor(QTextCursor.End)


class ConsoleBlock(QFrame):
    """One submission in the transcript, with its outcome."""

    def __init__(self, kind, text, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("block")
        self.setProperty("kind", kind)
        self._started = time.perf_counter()

        badge = QLabel({PYTHON: "py", AGENT: "agent"}.get(kind, kind))
        badge.setObjectName("badge")
        badge.setProperty("kind", kind)
        badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._status = QLabel("")
        self._status.setObjectName("status")
        self._status.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self._actions = QHBoxLayout()
        self._actions.setSpacing(4)

        header = QHBoxLayout()
        header.addWidget(badge, 0, Qt.AlignTop)
        self._input = _transcript_label(self._input_text(kind, text))
        if kind in (PYTHON, NOTE):
            self._input.setFont(_mono())
        else:
            font = self._input.font()
            font.setBold(True)
            self._input.setFont(font)
        header.addWidget(self._input, 1)
        header.addWidget(self._status, 0, Qt.AlignTop)

        self._body = QVBoxLayout()
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(2)
        self._details = QVBoxLayout()
        self._details.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addLayout(self._body)
        layout.addLayout(self._actions)
        layout.addLayout(self._details)
        self._actions.addStretch(1)

    @staticmethod
    def _input_text(kind, text):
        if kind == NOTE:
            return text.rstrip("\n")
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
        # The running ticker sets the same state ten times a second.
        if self._status.property("state") != state:
            self._status.setProperty("state", state)
            _repolish(self._status)

    def add_output(self, text, error=False, warning=False):
        """Add captured output; long text folds, and huge text keeps only its
        ends."""
        text = _keep_ends(text.rstrip("\n"), KEEP_CHARS)
        if not text:
            return
        label = _transcript_label()
        label.setFont(_mono())
        if error:
            label.setObjectName("error")
        elif warning:
            label.setObjectName("warning")
        self._body.addWidget(label)
        nline = text.count("\n") + 1
        if nline <= FOLD_LINES and len(text) <= FOLD_CHARS:
            label.setText(text)
            return
        head = "\n".join(text.split("\n", FOLD_LINES)[:FOLD_LINES])
        head = head[:FOLD_CHARS]
        more = ("Show all %d lines" % nline if nline > FOLD_LINES
                else "Show all %d characters" % len(text))
        label.setText(head)
        toggle = QToolButton()
        toggle.setObjectName("action")
        toggle.setAutoRaise(True)
        toggle.setText(more)
        toggle.setCheckable(True)

        def flip(on):
            label.setText(text if on else head)
            toggle.setText("Fold" if on else more)
        toggle.toggled.connect(flip)
        self._body.addWidget(toggle, 0, Qt.AlignLeft)

    def add_text(self, text):
        label = _transcript_label(text)
        self._body.addWidget(label)
        return label

    def add_command_row(self, command, result):
        """One row per command: a mark, the call, and its value or error."""
        ok = result is not None and getattr(result, "ok", False)
        if result is None:
            mark, detail = _DOT, "not run"
        elif ok:
            value = getattr(result, "value", None)
            mark, detail = _CHECK, _short_value(value)
        else:
            mark, detail = _CROSS, getattr(result, "error", None) or "failed"
        row = _transcript_label("%s %s%s" % (
            mark, agent.format_command(command),
            "  -> " + detail if detail else ""))
        row.setFont(_mono())
        if not ok:
            row.setObjectName("error")
        self._body.addWidget(row)

    def add_action(self, text, slot=None, tip=None):
        button = QToolButton()
        button.setObjectName("action")
        button.setText(text)
        if tip:
            button.setToolTip(tip)
        if slot is not None:
            button.clicked.connect(slot)
        self._actions.insertWidget(self._actions.count() - 1, button)
        return button

    def add_details(self, title, text):
        body = _transcript_label(text)
        body.setFont(_mono())
        body.setVisible(False)
        button = self.add_action(title)
        button.setCheckable(True)
        button.toggled.connect(body.setVisible)
        self._details.addWidget(body)
        return button


def _keep_ends(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return "%s\n... %d characters omitted ...\n%s" % (
        text[:half], len(text) - 2 * half, text[-half:])


def _short_value(value):
    if not value:
        return ""
    if isinstance(value, dict) and len(value) == 1:
        key, item = next(iter(value.items()))
        return "%s %s" % (key.replace("_id", ""), item)
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


class ConsolePanelWidget(_style.PaletteStyle, QWidget):
    """The dock body: the block transcript over the composer.  It shows
    blocks and emits :attr:`composer`'s signals and
    :attr:`settings_requested`; :class:`ConsolePanel` runs them."""

    settings_requested = Signal()

    #: 192 pixels of transcript over a 40 pixel composer.
    PREFERRED_HEIGHT = 192 + 40

    def __init__(self, backends=(), parent=None):
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

        self.chip = QToolButton()
        self.chip.setObjectName("chip")
        self.chip.setFont(_mono())
        self.chip.setToolTip("Python or agent (%s)" % _keys("I"))
        self.composer = ConsoleComposer()
        self.run = QPushButton("Run")

        self._backend_combo = QComboBox()
        self._backend_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToContents)
        for backend in backends:
            self._backend_combo.addItem(backend.name, backend)
        self.settings = QToolButton()
        self.settings.setObjectName("action")
        self.settings.setText("Settings")
        self.settings.setToolTip("Configure the selected agent backend")
        self._canvas = QLabel("canvas: none")
        # A window title is the user's text, not markup.
        self._canvas.setTextFormat(Qt.PlainText)
        self._hint = QLabel()
        self._hint.setEnabled(False)
        # The texts clip rather than widen the dock.
        for label in (self._canvas, self._hint):
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        entry = QHBoxLayout()
        entry.setContentsMargins(6, 0, 6, 0)
        entry.addWidget(self.chip, 0, Qt.AlignTop)
        entry.addWidget(self.composer, 1)
        entry.addWidget(self.run, 0, Qt.AlignTop)

        context = QHBoxLayout()
        context.setContentsMargins(6, 0, 6, 4)
        context.addWidget(QLabel("backend:"))
        context.addWidget(self._backend_combo)
        context.addWidget(self.settings)
        context.addSpacing(12)
        context.addWidget(self._canvas, 1)
        context.addWidget(self._hint, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._scroll, 1)
        layout.addLayout(entry)
        layout.addLayout(context)

        self.chip.clicked.connect(self._toggle_mode)
        self.composer.mode_changed.connect(self._show_mode)
        self.run.clicked.connect(self._run_clicked)
        self.settings.clicked.connect(self.settings_requested)
        self._running = False
        self._show_mode(self.composer.mode)
        self._apply_style()

    def sizeHint(self):
        # An empty transcript has no height of its own to ask for.
        hint = super().sizeHint()
        hint.setHeight(max(hint.height(), self.PREFERRED_HEIGHT))
        return hint

    def _apply_style(self):
        self.setStyleSheet(_PanelRules.sheet(
            self, "blocks", "outcomes", "controls"))

    def _toggle_mode(self):
        self.composer.set_mode(
            AGENT if self.composer.mode == PYTHON else PYTHON)
        self.composer.setFocus()

    def _show_mode(self, mode):
        self.chip.setText(PROMPTS[mode])
        self.chip.setProperty("mode", mode)
        _repolish(self.chip)
        self.composer.setPlaceholderText(
            "Ask the agent to draw, arrange windows, or aim the view"
            if mode == AGENT else "Python; %s asks the agent" % _keys("I"))
        hints = ["Enter runs", "Shift+Enter line", _keys("I") + " mode",
                 _keys("L") + " clears"]
        hints.insert(2, "Tab completes" if mode == PYTHON else "Esc stops")
        self._hint.setText((" %s " % _DOT).join(hints))
        self._hint.setToolTip(self._hint.text())
        self._show_run()

    def _run_clicked(self):
        if self._running and self.composer.mode == AGENT:
            self.composer.stop_requested.emit()
        else:
            text = self.composer.toPlainText()
            if text.strip():
                self.composer.submitted.emit(text)

    def set_running(self, running):
        """Turn Run into Stop while an agent turn is in flight, in agent
        mode; Python still runs during a turn."""
        self._running = running
        self._show_run()
        self._backend_combo.setEnabled(not running)
        # The call in flight already holds its configuration.
        self.settings.setEnabled(not running)

    def _show_run(self):
        stop = self._running and self.composer.mode == AGENT
        self.run.setText("Stop" if stop else "Run")

    def selected_backend(self):
        return self._backend_combo.currentData()

    def set_canvas(self, title):
        self._canvas.setText("canvas: " + (title or "none"))
        self._canvas.setToolTip(self._canvas.text())

    def _track_bottom(self, value):
        self._follow = value >= self._scroll.verticalScrollBar().maximum()

    def _keep_bottom(self, lo, hi):
        if self._follow:
            self._scroll.verticalScrollBar().setValue(hi)

    def add_block(self, kind, text, keep=None):
        """Add a block, dropping the oldest past :data:`KEEP_BLOCKS` but
        ``keep``, the one a running turn still writes to."""
        # A note is not worth losing the reader's place for.
        if kind != NOTE:
            self._follow = True
        block = ConsoleBlock(kind, text)
        self._blocks.insertWidget(self._blocks.count() - 1, block)
        self._block_list.append(block)
        excess = max(0, len(self._block_list) - KEEP_BLOCKS)
        old = [b for b in self._block_list[:-1] if b is not keep][:excess]
        for gone in old:
            self._block_list.remove(gone)
            gone.setParent(None)
            gone.deleteLater()
        return block

    def blocks(self):
        # Not from the layout: a wrapper of QLayout.itemAt() outlives its
        # item, and a later widget at that address can read as a QWidgetItem.
        return list(self._block_list)

    def clear(self, keep=None):
        """Remove every block but ``keep``, the one a running turn still
        writes to."""
        for block in self.blocks():
            if block is keep:
                continue
            block.setParent(None)
            block.deleteLater()
        self._block_list = [] if keep is None else [keep]


class ConsolePanel(base.PilotFeature):
    """The Console dock.  Python runs in the shared interpreter; an agent
    prompt runs through :class:`~solvcon.pilot.agent.AgentTurnRunner`
    without blocking the window, and a finished turn can be undone or
    replayed as Python."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._panel = None
        self._runner = agent.AgentTurnRunner(self._mgr, parent=self)
        self._runner.payload_composed.connect(self._show_payload)
        self._runner.step_applied.connect(self._show_step)
        self._runner.turn_finished.connect(self._finish)
        self._history = {PYTHON: [], AGENT: []}
        self._history_index = None
        self._draft = ""
        #: The block of the running agent turn, and whether the user was
        #: told it holds off another prompt.
        self._block = None
        self._told_busy = False
        self._ticker = QTimer(self)
        self._ticker.setInterval(100)
        self._ticker.timeout.connect(self._tick)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_all)

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Console",
            "Toggle the Python and agent console", None,
            id="panel.console", weight=30, checkable=True, checked=True)
        self._action.toggled.connect(self._on_toggled)
        self._ensure_panel()
        self._dock.show()
        base.set_message_sink(self.write_message)

    def write_message(self, text):
        """Show ``text`` as a note; its trailing newline is dropped."""
        self._ensure_panel()
        self._panel.add_block(NOTE, text, keep=self._block)

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
        self._panel = ConsolePanelWidget(
            backends=BackendRegistry.available())
        composer = self._panel.composer
        composer.submitted.connect(self.submit)
        composer.navigate.connect(self._navigate)
        composer.mode_changed.connect(self._end_history_walk)
        composer.stop_requested.connect(self.stop)
        composer.clear_requested.connect(self.clear)
        self._panel.settings_requested.connect(self._edit_settings)
        self._dock = QDockWidget("Console")
        self._dock.setWidget(self._panel)
        self._mainWindow.addDockWidget(Qt.BottomDockWidgetArea, self._dock)
        self._dock.visibilityChanged.connect(self._sync_action)
        self._mgr.mdiArea.subWindowActivated.connect(self._show_canvas)
        self._show_canvas()
        apputil.get_current_appenv().seed(agent_run=self.agent_run)

    def _sync_action(self, _visible):
        # The signal is also False for a dock tabbed behind another.
        self._action.setChecked(not self._dock.isHidden())

    def _edit_settings(self):
        backend = self._panel.selected_backend()
        if backend is None:
            return
        error = agent.edit_backend_settings(backend, parent=self._panel)
        if error:
            self.write_message(error)

    def _show_canvas(self, subwin=None):
        title = None
        if self._mgr.currentR2DWidget() is not None:
            try:
                title = self._mgr.mdiArea.currentSubWindow().windowTitle()
            except (AttributeError, RuntimeError):
                title = "2D canvas"
        self._panel.set_canvas(title)

    def _navigate(self, offset):
        """Step ``offset`` entries through the current mode's history; the
        unsent draft comes back past the newest entry."""
        mode = self._panel.composer.mode
        entries = self._history[mode]
        if not entries:
            return
        if self._history_index is None:
            if offset > 0:
                return
            self._draft = self._panel.composer.toPlainText()
            index = len(entries)
        else:
            index = self._history_index
        index += offset
        if index >= len(entries):
            self._history_index = None
            self._panel.composer.set_text(self._draft)
            return
        self._history_index = max(0, index)
        self._panel.composer.set_text(entries[self._history_index])

    def _remember(self, mode, text):
        entries = self._history[mode]
        if not entries or entries[-1] != text:
            entries.append(text)
            del entries[:-KEEP_HISTORY]
        self._end_history_walk()

    def _end_history_walk(self, mode=None):
        """Forget the history position; each mode has its own history, so a
        position in one means nothing in the other."""
        self._history_index = None
        self._draft = ""

    def submit(self, text):
        mode = self._panel.composer.mode
        if mode == AGENT and self._runner.busy:
            # Once per turn: pressing Enter again is no news.
            if not self._told_busy:
                self._told_busy = True
                self.write_message("The agent's turn has not ended, so this "
                                   "prompt stays in the composer. Esc stops "
                                   "the turn.")
            return
        self._remember(mode, text.rstrip("\n"))
        self._panel.composer.clear()
        if mode == PYTHON:
            self.run_python(text)
        else:
            self.ask_agent(text.strip())

    def run_python(self, source):
        """Run ``source`` in the shared interpreter and capture its output.
        It runs on the GUI thread, so a long command freezes the window."""
        source = source.rstrip()
        block = self._panel.add_block(PYTHON, source, keep=self._block)
        runs = run_cell(source)
        for is_error, text in runs:
            block.add_output(text, error=is_error and runs.raised,
                             warning=is_error and not runs.raised)
        block.set_status("error" if runs.raised else "ok")
        return block

    def agent_run(self, commands):
        """Run agent commands from Python and print each outcome; the "As
        Python" action of an agent block writes this call.

        The commands go through the session, so an op it refuses to the
        agent stays refused here, and the run lands in the conversation the
        agent sees next.
        """
        commands = [dict(c) for c in commands]
        results = self._runner.session.apply_commands(commands)
        for command, result in zip(commands, results):
            print("%s %s%s" % (
                _CHECK if result.ok else _CROSS,
                agent.format_command(command),
                "" if result.ok else "  -> %s" % result.error))
        return results

    def ask_agent(self, prompt):
        """Start one agent turn on the active canvas without blocking.
        Returns its block, or None while another turn holds the backend."""
        if self._runner.busy:
            return None
        block = self._panel.add_block(AGENT, prompt)
        self._block, self._told_busy = block, False
        block.set_status("running")
        self._panel.set_running(True)
        self._ticker.start()
        # With no backend the turn finishes before this returns.
        self._runner.start(prompt, self._panel.selected_backend())
        return block

    def _show_payload(self, text):
        self._block.add_details("Payload", text)

    def _tick(self):
        if self._block is not None:
            self._block.set_status("running")

    #: How a turn's stop shows on its block: a status and an optional note.
    _STOPS = {
        StopReason.COMPLETED: ("ok", None),
        StopReason.PROSE: ("ok", None),
        StopReason.BUDGET: ("ok", "Step budget reached; ask again to go "
                                  "on."),
        StopReason.STATE: ("stopped", "The canvas changed while the agent "
                                      "was working; its last reply was not "
                                      "applied."),
        StopReason.STOPPED: ("stopped", "Stopped; a late reply is "
                                        "dropped."),
        StopReason.CANCELLED: ("stopped", None),
        StopReason.NO_BACKEND: ("error", "no backend selected"),
    }

    def _finish(self, reason, records, world):
        """Show why the turn stopped and what can be done."""
        block, self._block = self._block, None
        self._ticker.stop()
        state, note = self._STOPS.get(reason, ("error", None))
        if note is not None:
            block.add_output(note, error=(state != "ok"))
        # A step that failed and a later one that repaired it end well, so
        # the last step that acted says how the turn went.
        acted = [r for r in records if r.commands or r.failed]
        if state == "ok" and acted and (acted[-1].failed or not all(
                getattr(r, "ok", False) for r in acted[-1].results)):
            state = "error"
        block.set_status(state)
        self._add_turn_actions(block, records, world)
        self._panel.set_running(False)
        # Take the focus back from Stop, unless the user moved elsewhere.
        if self._panel.isAncestorOf(QApplication.focusWidget()):
            self._panel.composer.setFocus()

    def _show_step(self, record):
        """One step's reply: the failure, or the log messages or text, then
        one row per command that acts."""
        block = self._block
        if record.failed:
            block.add_output(record.text, error=True)
        logs = [str(c.get("message")) for c, r in zip(record.commands,
                                                      record.results)
                if op_of(c) == "log" and getattr(r, "ok", False)
                and c.get("message") is not None]
        if not record.failed and (logs or record.text):
            block.add_text("\n".join(logs) if logs else record.text)
        for command, result in zip(record.commands, record.results):
            if op_of(command) != "log":
                block.add_command_row(command, result)
        for command in record.commands[len(record.results):]:
            block.add_command_row(command, None)

    def _add_turn_actions(self, block, records, world):
        pairs = [(c, r) for record in records
                 for c, r in zip(record.commands, record.results)
                 if op_of(c) != "log"]
        missing = [c for record in records
                   for c in record.commands[len(record.results):]]
        if not pairs and not missing:
            return
        steps = agent.undo_turn_steps(pairs)
        if world is not None and steps:
            after = world.state_stamp
            button = block.add_action(
                "Undo turn", lambda: self._undo(world, after, steps, button),
                "Revert everything this turn drew")
        applied = [c for c, r in pairs if getattr(r, "ok", False)]
        skipped = [(c, "failed") for c, r in pairs
                   if not getattr(r, "ok", False)]
        skipped += [(c, "not run") for c in missing]
        block.add_action(
            "As Python", lambda: self._as_python(applied, skipped),
            "Put the commands that ran in the composer as Python")

    def _undo(self, world, stamp, steps, button):
        if world.state_stamp != stamp or not world.can_undo:
            button.setEnabled(False)
            button.setToolTip("The canvas changed after this turn")
            return
        for _ in range(steps):
            world.undo()
        widget = self._mgr.currentR2DWidget()
        if widget is not None and widget.world is world:
            widget.requestRepaint()
        button.setEnabled(False)
        button.setText("Undone")

    def _as_python(self, commands, skipped=()):
        composer = self._panel.composer
        composer.set_mode(PYTHON)
        composer.set_text(agent.commands_as_python(commands, skipped))
        composer.setFocus()

    def stop(self):
        if self._runner.stop():
            self._block.set_status("stopping", "stopping...")
            self._ticker.stop()

    def clear(self):
        self._panel.clear(keep=self._block)

    def _stop_all(self):
        """Stop the turn and wait its call out before the app exits."""
        self.stop()
        self._runner.wait()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
