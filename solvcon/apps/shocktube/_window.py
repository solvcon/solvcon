# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Shock-tube window."""

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import (QButtonGroup, QHBoxLayout, QLabel,
                               QPushButton, QToolButton, QVBoxLayout,
                               QWidget)

from . import _figure
from . import _style

__all__ = [
    'ShockTubeWindow',
]


class ShockTubeWindow(QWidget):
    """The run controls and the title over the panels, and the chips that
    pick the quantities.  It paints in the colors of the palette, in the
    style of :mod:`._style`, so it follows the theme.

    :ivar plot: The :class:`ProfilePlot`.
    :ivar legend: The :class:`Legend` of its curves, right of the title.
    :ivar chips: The checkable button of each quantity, by name.
    :ivar exact_buttons: The ``'points'`` and ``'cells'`` halves of the
        switch that draws the analytic solution at points or averaged over
        the cells of the run.
    :ivar fronts_chip: The checkable button that shows the wave fronts.
    :ivar option_chips: The ``'log'`` and ``'residual'`` chips of the
        options of :attr:`plot`.
    :ivar xt: The :class:`XtDiagram` under the panels.
    :ivar xt_chip: The checkable button that shows :attr:`xt`.
    :ivar buttons: The ``'run'``, ``'step'``, and ``'reset'`` buttons.
    :ivar fit_button: The button that shows the whole tube again, enabled
        while the panels are zoomed in.
    :ivar run_clicked: Owner-supplied callback for the button that runs,
        pauses, or runs again.
    :ivar step_clicked: Owner-supplied callback for Step.
    :ivar reset_clicked: Owner-supplied callback for Reset.
    :ivar exact_clicked: Owner-supplied callback for
        :attr:`exact_buttons`.
    """

    #: The tone of the state pill by the state of the run.
    TONES = {'ready': 'muted', 'running': 'accent', 'paused': 'muted',
             'finished': 'good'}
    #: What the run button reads by the state of the run.
    RUN_TEXT = {'ready': "Run", 'running': "Pause", 'paused': "Resume",
                'finished': "Run again"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)
        self.plot = _figure.ProfilePlot()
        self.run_clicked = self.step_clicked = self.reset_clicked = None
        self.exact_clicked = None
        self.buttons = {}
        controls = QHBoxLayout()
        controls.setSpacing(8)
        for key, text in (('run', "Run"), ('step', "Step"),
                          ('reset', "Reset")):
            button = QPushButton(text)
            button.setObjectName('primary' if 'run' == key else 'secondary')
            button.clicked.connect(lambda _=False, name=key: self._on(name))
            self.buttons[key] = button
            controls.addWidget(button)
        self._state = QLabel("")
        self._state.setObjectName("pill")
        controls.addSpacing(10)
        controls.addWidget(self._state)
        controls.addStretch(1)
        self.fit_button = QPushButton("Reset zoom")
        self.fit_button.setObjectName("secondary")
        self.fit_button.setToolTip("Show the whole tube again (a "
                                   "double-click on the panels does the "
                                   "same)")
        self.fit_button.setEnabled(False)
        self.fit_button.clicked.connect(lambda: self.plot.set_view(None))
        controls.addWidget(self.fit_button)
        self._title = QLabel("")
        font = QFont(self.font())
        font.setPixelSize(14)
        font.setWeight(QFont.DemiBold)
        self._title.setFont(font)
        self.legend = _figure.Legend()
        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(self._title)
        head.addStretch(1)
        head.addWidget(self.legend)
        self._waves = QLabel("")
        self._waves.setToolTip("The waves of the exact solution, left to "
                               "right (S shock, R rarefaction, C contact), "
                               "and the star state between them")
        self.chips = {}
        chips = QHBoxLayout()
        chips.setSpacing(5)
        for name, text in _figure.QUANTITIES:
            chip = self._chip(text, name in _figure.SHOWN)
            chip.toggled.connect(lambda on, key=name: self._on_chip(key))
            self.chips[name] = chip
            chips.addWidget(chip)
        chips.addStretch(1)
        label = QLabel("exact:")
        label.setObjectName("switchLabel")
        chips.addWidget(label)
        self.exact_buttons = {}
        group = QButtonGroup(self)
        switch = QHBoxLayout()
        switch.setSpacing(0)
        for key, edge, tip in (
                ('points', 'left', "Draw the exact solution at points"),
                ('cells', 'right', "Draw the exact solution averaged over "
                 "the cells of the run, as a conservative scheme holds "
                 "it")):
            button = QToolButton()
            button.setObjectName("switch")
            button.setProperty('edge', edge)
            button.setText(key)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setChecked('points' == key)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda _=False: self._on('exact'))
            group.addButton(button)
            self.exact_buttons[key] = button
            switch.addWidget(button)
        chips.addLayout(switch)
        chips.addSpacing(10)
        self.fronts_chip = self._chip("wave fronts", True)
        self.fronts_chip.toggled.connect(self.plot.set_show_fronts)
        chips.addWidget(self.fronts_chip)
        self.option_chips = {}
        for key, text, tip in (
                ('log', "log y", "A log axis on the panels whose values are "
                                 "all positive"),
                ('residual', "residual", "A strip under each panel with the "
                                         "march minus the exact cell means")):
            chip = self._chip(text, False)
            chip.setToolTip(tip)
            chip.toggled.connect(
                lambda on, name=key: self.plot.set_option(name, on))
            self.option_chips[key] = chip
            chips.addWidget(chip)
        self.xt = _figure.XtDiagram()
        self.plot.view_changed = self._on_view
        self.xt_chip = self._chip("x-t diagram", True)
        self.xt_chip.toggled.connect(self.xt.setVisible)
        chips.addWidget(self.xt_chip)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 10, 14, 8)
        self._layout.setSpacing(6)
        self._layout.addLayout(controls)
        self._layout.addLayout(head)
        self._layout.addWidget(self._waves)
        self._layout.addWidget(self.plot, 2)
        self._layout.addWidget(self.xt, 1)
        self._layout.addLayout(chips)
        self._apply_style()

    def sizeHint(self):
        return QSize(1000, 560)

    def event(self, event):
        # A theme switch changes the palette, and the toggles' shades with
        # it.
        if event.type() in (QEvent.PaletteChange,
                            QEvent.ApplicationPaletteChange):
            self._apply_style()
        return super().event(event)

    def exact_mode(self):
        """Return how the analytic solution is drawn: ``'points'`` or
        ``'cells'``."""
        return next(key for key, button in self.exact_buttons.items()
                    if button.isChecked())

    def set_title(self, title, time):
        """Head the panels with ``title`` at ``time``."""
        self._title.setText(f"{title}    t = {time:.4g}")

    def title(self):
        """Return the text heading the panels."""
        return self._title.text()

    def show_waves(self, exact):
        """Say the wave pattern and the star state of ``exact``, an
        :class:`~._riemann.ExactRiemann`."""
        self._waves.setText(
            f"{exact.pattern}    p* = {exact.p_star:.5g}    "
            f"u* = {exact.u_star:.5g}    "
            f"rho*L = {exact.rho_star_left:.5g}    "
            f"rho*R = {exact.rho_star_right:.5g}")

    def waves_text(self):
        """Return what the window says of the waves."""
        return self._waves.text()

    def show_state(self, state, step, nstep):
        """Show the run state (``'ready'``, ``'running'``, ``'paused'``, or
        ``'finished'``) and its step on the controls."""
        self.buttons['run'].setText(self.RUN_TEXT[state])
        self.buttons['step'].setEnabled(state in ('ready', 'paused'))
        self.buttons['reset'].setEnabled('ready' != state)
        self._state.setText(f"{state}, step {step} / {nstep}")
        if self._state.property("tone") != self.TONES[state]:
            self._state.setProperty("tone", self.TONES[state])
            # A sheet reads a dynamic property only when it polishes.
            self._state.style().unpolish(self._state)
            self._state.style().polish(self._state)

    def state_text(self):
        """Return what the controls say of the run."""
        return self._state.text()

    def _on(self, name):
        callback = getattr(self, f"{name}_clicked")
        if callback is not None:
            callback()

    def _on_view(self, view):
        self.fit_button.setEnabled(view is not None)
        self.xt.set_view(view)

    def _chip(self, text, checked):
        chip = QToolButton()
        chip.setObjectName("chip")
        chip.setText(text)
        chip.setCheckable(True)
        chip.setChecked(checked)
        chip.setFocusPolicy(Qt.NoFocus)
        return chip

    def _apply_style(self):
        self.setStyleSheet(_style.Rules.sheet(self, "buttons", "pill",
                                              "chips"))

    def _on_chip(self, name):
        names = [key for key, _ in _figure.QUANTITIES
                 if self.chips[key].isChecked()]
        if not names:
            # Keep the last panel so the window never goes blank.
            self.chips[name].blockSignals(True)
            self.chips[name].setChecked(True)
            self.chips[name].blockSignals(False)
            return
        self.plot.set_quantities(names)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
