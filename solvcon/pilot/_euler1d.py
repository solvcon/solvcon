# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


from ..onedim import euler1d

from . import _base_app


class Euler1DApp(_base_app.OneDimBaseApp):
    """
    Main application for Euler 1D solver.
    """

    TITLE = "Euler solver"

    def populate_menu(self):
        """
        Set menu item for GUI.
        """
        self.add_action(
            "One",
            text="Euler solver",
            tip="One-dimensional shock-tube problem with Euler solver",
            func=self.run,
            id="one.euler",
            weight=10,
        )

    def init_solver_config(self):
        """
        Set shock tube solver configuration by user' input, and then reset the
        initial condition.
        """
        solver_config_data = [
            ["gamma", 1.4, "The ratio of the specific heats."],
            ["p_left", 1.0, "The pressure of left hand side."],
            ["rho_left", 1.0, "The density of left hand side."],
            ["p_right", 0.1, "The pressure of right hand side."],
            ["rho_right", 0.125, "The density of right hand side."],
            ["xmin", -10, "The most left point of x axis."],
            ["xmax", 10, "The most right point of x axis."],
            ["ncoord", 201, "Number of grid point."],
            ["time_increment", 0.05, "The density of right hand side."],
            ["time_interval", 10, "Qt timer interval"],
            ["max_steps", 50, "Maximum step"],
            ["profiling", False, "Turn on / off solver profiling"],
        ]
        self.solver_config = _base_app.SolverConfig(solver_config_data)

    def set_plot_data(self):
        """
        Set the analytical and numerical data for plot.
            - density
            - pressure
            - velocity
            - temperature
            - internal_energy
            - entropy
        """
        self.plot_ana = True
        self.plot_num = True
        self.plot_data = []

        density = _base_app.QuantityLine(
            name="density",
            unit=r"$\mathrm{kg}/\mathrm{m}^3$",
            color='r',
            y_upper_lim=1.2,
            y_bottom_lim=-0.1)
        setattr(self, density.name, density)
        self.plot_data.append([self.density.name, True])

        pressure = _base_app.QuantityLine(
            name="pressure",
            unit=r"$\mathrm{Pa}$",
            color='g',
            y_upper_lim=1.2,
            y_bottom_lim=-0.1)
        setattr(self, pressure.name, pressure)
        self.plot_data.append([self.pressure.name, True])

        velocity = _base_app.QuantityLine(
            name="velocity",
            unit=r"$\mathrm{m}/\mathrm{s}$",
            color='b',
            y_upper_lim=1.2,
            y_bottom_lim=-0.1)
        setattr(self, velocity.name, velocity)
        self.plot_data.append([self.velocity.name, True])

        temperature = _base_app.QuantityLine(
            name="temperature",
            unit=r"$\mathrm{K}$",
            color='c',
            y_upper_lim=0.15,
            y_bottom_lim=0.0)
        setattr(self, temperature.name, temperature)
        self.plot_data.append([self.temperature.name, False])

        internal_energy = _base_app.QuantityLine(
            name="internal_energy",
            unit=r"$\mathrm{J}/\mathrm{kg}$",
            color='k',
            y_upper_lim=3.0,
            y_bottom_lim=1.5)
        setattr(self, internal_energy.name, internal_energy)
        self.plot_data.append([self.internal_energy.name, False])

        entropy = _base_app.QuantityLine(
            name="entropy",
            unit=r"$\mathrm{J}/\mathrm{K}$",
            color='m',
            y_upper_lim=2.2,
            y_bottom_lim=0.9)
        setattr(self, entropy.name, entropy)
        self.plot_data.append([self.entropy.name, False])

    def init_solver(self):
        """
        Initialize the shock tube solver and set up the initial condition.
        """
        self.st = euler1d.ShockTube()
        _s = self.solver_config
        self.st.build_constant(gamma=_s["gamma"]["value"],
                               pressure1=_s["p_left"]["value"],
                               density1=_s["rho_left"]["value"],
                               pressure5=_s["p_right"]["value"],
                               density5=_s["rho_right"]["value"])
        self.st.build_numerical(xmin=_s["xmin"]["value"],
                                xmax=_s["xmax"]["value"],
                                ncoord=_s["ncoord"]["value"],
                                time_increment=_s["time_increment"]["value"])
        self.st.build_field(t=0)

    def update_step(self, steps):
        """
        Update data at current step.
            - numerical data solved in c++
            - analytical data solved in python
        """
        self.st.svr.march_alpha2(steps=steps)
        self.current_step += steps
        time_current = self.current_step * self.st.svr.time_increment
        self.st.build_field(t=time_current)
        cfl = self.st.svr.cfl
        self.log(f"CFL: min {cfl.min()} max {cfl.max()}")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
