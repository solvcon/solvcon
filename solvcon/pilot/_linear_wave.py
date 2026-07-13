# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import numpy as np

from . import _base_app
from ..onedim import linear_scalar


class LinearWave:

    def __init__(self):
        self.svr = None
        self.coord_field = None
        self.wave_field = None

    def build_numerical(self, xmin, xmax, ncelm, cfl):

        # Initialize the solver.
        self.svr = linear_scalar.LinearScalarSolver(xmin, xmax, ncelm, cfl)

        # Setup the solver for time-marching.
        self.svr.setup_march()

    def build_field(self):
        self.coord_field = self.svr.xctr() / np.pi
        self.wave_field = self.svr.get_so0(0).ndarray


class LinearWave1DApp(_base_app.OneDimBaseApp):

    TITLE = "Linear Wave"

    def populate_menu(self):
        """
        Set menu item for GUI.
        """
        self.add_action(
            "One",
            text="Linear Wave",
            tip="",
            func=self.run,
            id="one.linear_wave",
            weight=30,
        )

    def init_solver_config(self):
        solver_config_data = [
            ["xmin", 0, "The most left point of x axis."],
            ["xmax", 4 * 2 * np.pi, "The most right point of x axis."],
            ["ncelm", 4 * 64, ""],
            ["cfl", 1, ""],
            ["time_interval", 10, "Qt timer interval"],
            ["max_steps", 0, "Maximum step"],
            ["profiling", False, "Turn on / off solver profiling"],
        ]
        self.solver_config = _base_app.SolverConfig(solver_config_data)

    def set_plot_data(self):
        """
        Set the analytical data for plot.
        """
        self.plot_ana = True
        self.plot_data = []

        wave = _base_app.QuantityLine(
            name="wave",
            color='b',
            y_upper_lim=1.2,
            y_bottom_lim=-1.2)
        setattr(self, wave.name, wave)
        self.plot_data.append([self.wave.name, True])

    def init_solver(self):
        self.st = LinearWave()
        _s = self.solver_config
        self.st.build_numerical(xmin=_s["xmin"]["value"],
                                xmax=_s["xmax"]["value"],
                                ncelm=_s["ncelm"]["value"],
                                cfl=_s["cfl"]["value"])
        self.st.build_field()

    def update_step(self, steps):
        """
        Update data at current step.
        """
        self.st.svr.march_alpha2(steps=steps)
        self.st.build_field()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
