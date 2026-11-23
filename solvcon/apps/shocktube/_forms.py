# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The groups of the shock-tube control column, which edit the run
parameters and what the figure shows."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QComboBox

from ...pilot import panel

from . import _session
from . import _setup
from . import _viewmodel

__all__ = [
    'DisplayGroup',
    'NumericsGroup',
    'ProblemGroup',
]


_SETUP = _setup.ShockTubeSetup


class ProblemGroup(panel.SetupGroup):
    """The two gases the next Reset bursts the diaphragm between."""

    SETUP = _SETUP
    KEYS = _SETUP.PROBLEM
    LABELS = {
        'rho_left': ("left density", "Density of the gas left of the "
                                     "diaphragm"),
        'p_left': ("left pressure", "Pressure of the gas left of the "
                                    "diaphragm"),
        'rho_right': ("right density", "Density of the gas right of the "
                                       "diaphragm"),
        'p_right': ("right pressure", "Pressure of the gas right of the "
                                      "diaphragm"),
        'gamma': ("gamma", "Ratio of the specific heats of both gases"),
    }

    def __init__(self, parent=None):
        super().__init__("Problem", parent)
        self._add_messages()


class NumericsGroup(panel.SetupGroup):
    """The tube and the time step the next Reset builds, and the step cap
    and the steps per frame, which apply to the standing run at once."""

    SETUP = _SETUP
    KEYS = _SETUP.NUMERICS
    LABELS = {
        'xmin': ("x min", "The left end of the tube; the diaphragm "
                          "stands at x = 0"),
        'xmax': ("x max", "The right end of the tube"),
        'ncoord': ("points", "The grid points, an odd count; the run "
                             "holds its solution at every other one"),
        'time_increment': ("time step", "The time a step marches; a finer "
                                        "grid needs a shorter one"),
    }

    def __init__(self, parent=None):
        super().__init__("Numerics", parent)
        self._add_messages()
        self._add_run_settings(_session.ShockTubeSession.MAX_STEPS,
                               _session.ShockTubeSession.STEPS_PER_CHUNK)


class DisplayGroup(panel.ColumnGroup):
    """What the figure shows of the run: the quantity panels, how the exact
    solution is drawn, the guides of the fronts, the axis options, and the
    x-t diagram.  None of it marches the run, so an edit applies at once.
    """

    #: The ways the exact solution is drawn, and what the combo reads.
    EXACT = (('points', "through points"), ('cells', "as cell means"))
    #: The switches of :class:`~._viewmodel.Display`, the text of each
    #: box, and its tip.
    OPTIONS = (
        ('fronts', "wave fronts", "Guide the fronts of the exact solution "
                                  "and name the regions between them"),
        ('xt', "x-t diagram", "The history of the density under the waves "
                              "of the exact solution"),
        ('log', "log y", "A log axis on the panels whose values are all "
                         "positive"),
        ('residual', "residual", "A strip under each panel with the run "
                                 "minus the exact cell means"),
    )

    def __init__(self, parent=None):
        super().__init__("Display", parent)
        #: Owner-supplied callback fired with the
        #: :class:`~._viewmodel.Display` on any change.
        self.display_changed = None
        grid = self._body_grid(labels=panel.SetupGroup.LABEL_WIDTH)
        #: The box of each quantity, by name.
        self.quantity_boxes = {}
        grid.addWidget(panel.form_label("panels", "The quantities drawn, a "
                                        "panel each"), 0, 0)
        for it, (name, label) in enumerate(
                _viewmodel.QUANTITY_LABELS.items()):
            box = QCheckBox(label)
            self.quantity_boxes[name] = box
            grid.addWidget(box, 1 + it // 2, it % 2)
        row = 1 + (len(self.quantity_boxes) + 1) // 2
        self._exact = QComboBox()
        for key, text in self.EXACT:
            self._exact.addItem(text, key)
        self._exact.setToolTip("Draw the exact solution through the points "
                               "of a fine grid, or averaged over the cells "
                               "of the run, as a conservative scheme holds "
                               "it")
        grid.addWidget(panel.form_label("exact", self._exact.toolTip()),
                       row, 0)
        grid.addWidget(self._exact, row, 1)
        #: The box of each switch of :data:`OPTIONS`, by name.
        self.option_boxes = {}
        for it, (key, text, tip) in enumerate(self.OPTIONS):
            box = QCheckBox(text)
            box.setToolTip(tip)
            self.option_boxes[key] = box
            grid.addWidget(box, row + 1 + it // 2, it % 2)
        grid.setColumnStretch(1, 1)
        panel.guard_wheel(self._exact)
        self._controls = ((self._exact,) + tuple(self.quantity_boxes.values())
                          + tuple(self.option_boxes.values()))
        self.show_display(_viewmodel.Display())
        self._exact.currentIndexChanged.connect(self._on_change)
        for name, box in self.quantity_boxes.items():
            box.toggled.connect(lambda on, key=name: self._on_quantity(key))
        for box in self.option_boxes.values():
            box.toggled.connect(self._on_change)

    def display(self):
        """Return the :class:`~._viewmodel.Display` the controls hold."""
        return _viewmodel.Display(
            quantities=tuple(name for name, box
                             in self.quantity_boxes.items()
                             if box.isChecked()),
            exact=self._exact.currentData(),
            **{key: box.isChecked()
               for key, box in self.option_boxes.items()})

    def show_display(self, display):
        """Set the controls from a display without firing the callback.

        :param display: The :class:`~._viewmodel.Display` to show.
        """
        blockers = [QSignalBlocker(widget) for widget in self._controls]
        for name, box in self.quantity_boxes.items():
            box.setChecked(name in display.quantities)
        self._exact.setCurrentIndex(self._exact.findData(display.exact))
        for key, box in self.option_boxes.items():
            box.setChecked(getattr(display, key))
        for blocker in blockers:
            blocker.unblock()

    def _on_quantity(self, name):
        if not any(box.isChecked() for box in self.quantity_boxes.values()):
            # The last panel stays, so the figure never goes blank.
            with QSignalBlocker(self.quantity_boxes[name]):
                self.quantity_boxes[name].setChecked(True)
            return
        self._on_change()

    def _on_change(self, *_args):
        if self.display_changed is not None:
            self.display_changed(self.display())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
