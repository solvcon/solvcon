# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The window of the reflection app: the run strip over the figure."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

from ... import pilot

from . import _column
from . import _viewmodel
from . import _widgets

__all__ = [
    'ReflectionWindow',
    'RunStrip',
]


class RunStrip(QFrame):
    """The strip across the top of the window that drives the run.

    It shows the problem it marches, an action accent that runs and pauses,
    Step and Reset, and the state, the time, and the progress of the march.
    """

    #: The accent action reads by the state of the run.
    RUN_TEXT = {'ready': "Run", 'running': "Pause", 'paused': "Resume",
                'finished': "Run again", 'failed': "Run again"}
    PILL = {'ready': ("Ready", 'muted'), 'running': ("Running", 'accent'),
            'paused': ("Paused", 'muted'), 'finished': ("Finished", 'good'),
            'failed': ("Failed", 'bad')}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("runStrip")
        #: Owner-supplied callback for the accent action, which runs, pauses,
        #: resumes, or runs again by the state shown.
        self.run_clicked = None
        #: Owner-supplied callback for Step.
        self.step_clicked = None
        #: Owner-supplied callback for Reset.
        self.reset_clicked = None
        #: The run state shown, one of :data:`._viewmodel.STATES`.
        self.state = 'ready'

        self._problem = _widgets.ElidedLabel()
        self._problem.setObjectName("stripProblem")
        self._problem.shortest = "Mach 3..."
        self._run = QPushButton("Run")
        self._run.setObjectName("primary")
        self._run.setToolTip("Run or pause the march (Space)")
        self._run.clicked.connect(lambda: self._call(self.run_clicked))
        self._step = QPushButton("Step")
        self._step.setObjectName("secondary")
        self._step.setToolTip("March one frame (Right arrow)")
        self._step.clicked.connect(lambda: self._call(self.step_clicked))
        self._reset = QPushButton("Reset")
        self._reset.setObjectName("secondary")
        self._reset.clicked.connect(lambda: self._call(self.reset_clicked))
        #: The ``'run'``, ``'step'``, and ``'reset'`` buttons.
        self.buttons = {'run': self._run, 'step': self._step,
                        'reset': self._reset}
        for button in self.buttons.values():
            button.setFocusPolicy(Qt.TabFocus)
            button.setCursor(Qt.PointingHandCursor)

        self._pill = QLabel("Ready")
        self._pill.setObjectName("pill")
        self._pill.setAlignment(Qt.AlignCenter)
        self._failure = _widgets.ElidedLabel()
        self._failure.setObjectName("failure")
        self._time = QLabel("")
        self._time.setObjectName("stripTime")
        self._steps = _widgets.Fitted()
        self._steps.setObjectName("stripCaption")
        self._progress = QProgressBar()
        self._progress.setObjectName("progress")
        self._progress.setRange(0, 1000)
        self._progress.setTextVisible(False)
        self._progress.setMaximumWidth(260)
        # The bar repeats the time readout, so it gives up its room first.
        self._progress.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._problem)
        row.addSpacing(6)
        row.addWidget(self._run)
        row.addWidget(self._step)
        row.addWidget(self._reset)
        row.addSpacing(10)
        row.addWidget(self._pill)
        row.addStretch(1)
        row.addWidget(self._time)
        row.addWidget(self._steps)
        row.addWidget(self._progress, 1)
        # A failure takes a line of its own, so it never widens the strip.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 14, 8)
        layout.setSpacing(4)
        layout.addLayout(row)
        layout.addWidget(self._failure)
        self._failure.hide()
        self.restyle()
        self.show_state(_viewmodel.StripState('ready', 0, 0, 0.0, 0.0))

    def _call(self, callback):
        if callback is not None:
            callback()

    def restyle(self):
        """Draw the icons, at the scale of the screen, and the numbers in
        the current palette."""
        shades, ratio = pilot.Shades(self), self.devicePixelRatioF()
        # Drawn once per palette and scale; a frame only picks one.
        self._run_icons = {name: _widgets.icon(name, shades.on_accent, 11,
                                               ratio)
                           for name in ('play', 'pause')}
        self._step.setIcon(_widgets.icon('step', shades.on_base, 11, ratio))
        self._reset.setIcon(_widgets.icon('reset', shades.on_base, 12,
                                          ratio))
        self._time.setFont(pilot.number_font(self.font(), 13,
                                             QFont.DemiBold))
        self._steps.setFont(pilot.number_font(self.font(), 11))
        self._set_run_icon()
        # The widths come from the styled fonts, so they are measured here.
        _widgets.reserve_width(self._run, list(self.RUN_TEXT.values()))
        _widgets.reserve_width(self._pill,
                               [text for text, _ in self.PILL.values()])

    def _set_run_icon(self):
        self._run.setIcon(self._run_icons['pause' if 'running' == self.state
                                          else 'play'])

    def set_problem(self, text):
        """Name the problem the run marches, elided to the room left of the
        buttons, and whole in the tip.

        :param text: The name of the problem.
        """
        self._problem.set_full_text(text)

    def problem(self):
        """Return the whole name of the problem."""
        return self._problem.full_text

    def show_state(self, strip):
        """Read a strip state out into the strip.

        :param strip: The :class:`~._viewmodel.StripState` to show.
        """
        state = strip.state
        self.state = state
        self._run.setText(self.RUN_TEXT[state])
        self._set_run_icon()
        ended = state in ('finished', 'failed')
        self._run.setEnabled(strip.can_reset or not ended)
        self._step.setEnabled(state in ('ready', 'paused'))
        self._reset.setEnabled(strip.can_reset and (
            strip.dirty or 'ready' != state or strip.step > 0))
        pilot.set_style_property(self._reset, "attention",
                                 "true" if strip.dirty else "false")
        if not strip.can_reset:
            tip = "Fix the outlined values in the column to apply the edits"
        elif strip.dirty:
            tip = "Apply the edits and rebuild the run at t = 0 (R)"
        else:
            tip = "Rebuild the run at t = 0 from the inputs in the column (R)"
        self._reset.setToolTip(tip)
        text, tone = self.PILL[state]
        self._pill.setText(text)
        pilot.set_style_property(self._pill, "tone", tone)
        failure = " ".join(strip.failure.split())
        self._failure.set_full_text(failure)
        self._failure.setVisible(bool(failure))
        self._time.setText(_viewmodel.clock(strip.time, strip.t_end))
        nstep = strip.max_steps
        self._steps.set_forms(*((f"step {count} / {nstep}",
                                 f"{count} / {nstep}", f"{count}")
                                for count in (strip.step, nstep)))
        self._progress.setValue(round(1000 * strip.step / nstep) if nstep
                                else 0)

    def pill(self):
        """Return the text and the tone of the state pill."""
        return self._pill.text(), self._pill.property("tone")


class ReflectionWindow(_widgets.Styled, QWidget):
    """The reflection window: the run strip over the figure it is handed.
    The control column docks beside it as a panel of its own.

    :param figure: The widget to show under the strip, any :class:`QWidget`.
    """

    #: The strip button each run key presses.
    KEYS = {Qt.Key_Space: 'run', Qt.Key_Right: 'step', Qt.Key_R: 'reset'}
    ROLES = ("strip", "buttons", "pill", "progress")

    def __init__(self, figure, parent=None):
        super().__init__(parent)
        self.setObjectName("obsrefl")
        # The run keys reach the window through the focus, so a click on
        # its ground has to give it the focus.
        self.setFocusPolicy(Qt.ClickFocus)
        #: The :class:`RunStrip`.
        self.strip = RunStrip()
        #: The widget under the strip, any :class:`QWidget`.
        self.figure = figure
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.strip)
        layout.addWidget(self.figure, 1)
        self._styling = False
        self._apply_style()

    def sizeHint(self):
        return QSize(1000 - _column.ControlColumn.WIDTH, 560)

    def restyle(self):
        """Draw the icons and the numbers of the strip anew."""
        self.strip.restyle()

    def keyPressEvent(self, event):
        if not self.press_key(event):
            super().keyPressEvent(event)

    def press_key(self, event):
        """Press the strip button a run key names.

        :param event: The key event.
        :return: Whether the event holds a run key, which then goes no
            further.
        """
        name = self.KEYS.get(event.key())
        if name is None or event.modifiers() & ~Qt.KeypadModifier:
            return False
        event.accept()
        if event.isAutoRepeat() and 'step' != name:
            return True
        button = self.strip.buttons[name]
        if button.isEnabled():
            button.click()
        return True

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
