# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import numpy as np

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QDockWidget, QLabel, QPushButton, QTableView,
                               QVBoxLayout, QWidget)

from ...pilot import base

from . import _shocktube
from . import _window


class ShockTubeApp(base.PilotFeature):
    """
    The shock-tube app on App > Shock tube: the numerical solution of Sod's
    problem marched over the analytic one, with the config panel docked
    beside it.
    """

    #: Points the analytic solution is drawn through.
    EXACT_POINTS = 801

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.solver_config = base.SolverConfig([
            ["gamma", 1.4, "The ratio of the specific heats."],
            ["p_left", 1.0, "The pressure of left hand side."],
            ["rho_left", 1.0, "The density of left hand side."],
            ["p_right", 0.1, "The pressure of right hand side."],
            ["rho_right", 0.125, "The density of right hand side."],
            ["xmin", -10, "The most left point of x axis."],
            ["xmax", 10, "The most right point of x axis."],
            ["ncoord", 201, "Number of grid point."],
            ["time_increment", 0.05, "The time increment of a step."],
            ["time_interval", 10, "Qt timer interval"],
            ["max_steps", 50, "Maximum step"],
        ])
        self.st = None
        self.current_step = 0
        self.state = 'ready'
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.window = None
        self.subwindow = None
        self.dock = None
        self._close_filter = None

    @property
    def config(self):
        """The values of the config table by variable name."""
        cfg = self.solver_config
        return {cfg.data(row, 0): cfg.data(row, 1)
                for row in range(cfg.rowCount())}

    def populate_menu(self):
        """
        Set menu item for GUI.
        """
        self.add_action(
            "App",
            text="Shock tube",
            tip="One-dimensional shock-tube problem with Euler solver",
            func=self.open,
            id="app.shocktube",
            weight=10,
        )

    def open(self):
        """Open the window maximized, or bring it back."""
        if self.subwindow is not None:
            self._mgr.mdiArea.setActiveSubWindow(self.subwindow)
            self.subwindow.showMaximized()
            return
        self.window = _window.ShockTubeWindow()
        self.window.run_clicked = self.run_or_pause
        self.window.step_clicked = self.step
        self.window.reset_clicked = self.init_solver
        self.init_solver()
        self.subwindow = self._mgr.addSubWindow(self.window)
        self.subwindow.setWindowTitle("Shock tube: Sod")
        self._close_filter = base.SubWindowCloseFilter(self._on_closed,
                                                       self.subwindow)
        self.subwindow.installEventFilter(self._close_filter)
        self.subwindow.showMaximized()
        self.dock = self._build_dock()
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.dock)

    def _build_dock(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Configuration"))
        table = QTableView()
        table.horizontalHeader().setStretchLastSection(True)
        table.setModel(base.ConfigTableModel(self.solver_config, table))
        layout.addWidget(table)
        set_button = QPushButton("Set")
        set_button.clicked.connect(self.init_solver)
        layout.addWidget(set_button)
        dock = QDockWidget("config")
        dock.setWidget(panel)
        return dock

    def close(self):
        """Close the window."""
        if self.subwindow is not None:
            self.subwindow.close()

    def _on_closed(self):
        # Stop the march before Qt deletes the window it draws in.
        self.timer.stop()
        if self.dock is not None:
            self._mgr.mainWindow.removeDockWidget(self.dock)
            self.dock.deleteLater()
        self.window = None
        self.subwindow = None
        self.dock = None
        self._close_filter = None

    def init_solver(self):
        """Build the problem from the config table and start it over."""
        cfg = self.config
        self.st = _shocktube.ShockTube()
        self.st.build_constant(gamma=cfg["gamma"],
                               pressure1=cfg["p_left"],
                               density1=cfg["rho_left"],
                               pressure5=cfg["p_right"],
                               density5=cfg["rho_right"])
        self.st.build_numerical(xmin=cfg["xmin"], xmax=cfg["xmax"],
                                ncoord=cfg["ncoord"],
                                time_increment=cfg["time_increment"])
        self.timer.stop()
        self.timer.setInterval(cfg["time_interval"])
        self.current_step = 0
        self.state = 'ready'
        self.draw()

    def run_or_pause(self):
        """Run, pause, resume, or run again from t = 0."""
        if 'running' == self.state:
            self.timer.stop()
            self.state = 'paused'
        else:
            if 'finished' == self.state:
                self.init_solver()
            self.state = 'running'
            self.timer.start()
        self.draw()

    def step(self):
        """March one step, and stop the march at the last one."""
        if 'finished' == self.state:
            return
        if 'ready' == self.state:
            self.state = 'paused'
        self.st.svr.march_alpha2(steps=1)
        self.current_step += 1
        if self.current_step >= self.config["max_steps"]:
            self.timer.stop()
            self.state = 'finished'
        self.draw()

    def draw(self):
        """Draw the current step over the analytic solution at its time."""
        cfg = self.config
        time = self.current_step * cfg["time_increment"]
        svr = self.st.svr
        points = (svr.coord[svr.xindices],
                  {name: getattr(svr, name)[svr.xindices]
                   for name, _ in _window.QUANTITIES})
        coord = np.linspace(cfg["xmin"], cfg["xmax"], self.EXACT_POINTS,
                            dtype='float64')
        self.st.build_field(t=time, coord=coord)
        exact = (self.st.coord_field,
                 {name: getattr(self.st, f"{name}_field")
                  for name, _ in _window.QUANTITIES})
        self.window.plot.set_data((cfg["xmin"], cfg["xmax"]), exact, points)
        self.window.set_title("Sod", time)
        self.window.show_state(self.state, self.current_step,
                               cfg["max_steps"])


def build_app(controller):
    """Build the app on the pilot."""
    return ShockTubeApp(mgr=controller.mgr)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
