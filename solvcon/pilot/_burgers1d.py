# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import numpy as np

from . import _base_app


class BurgersEquation:
    """
    Burgers' equation problem.
    See the Computational Gasdynamics by Laney, Culbert B.
    """

    def __init__(self):
        self.coord = None
        self.velocity = None

        # Field data of the analytical solution.
        self.coord_field = None
        self.velocity_field = None

    def build_constant(self, x, velocity):
        """
        Set solver initial condition.

        :param x (type: list): the points of x axis.
        :param velocity (type: list): the velocities.
        """
        # Set the given value.
        self.coord = np.array(x, dtype='float64')
        self.velocity = np.array(velocity, dtype='float64')

    def build_field(self, t):
        """
        Populate the field data using the analytical solution.

        :param t: time
        """
        self.coord_field = self.calc_coord_field(t)
        _ = (self.velocity, self.velocity)
        self.velocity_field = np.transpose(_).ravel()

    def calc_coord_field(self, t):
        """
        Calculate the wavefront location after a period of time.

        :param t: time
        :return: wavefront location (type: ndarray)
        """
        # Calculate the left side and right side wavefront location
        # separated by a jump discontinuity.
        left = self.coord[1:-1] + self.velocity[0:-1] * t
        right = self.coord[1:-1] + self.velocity[1:] * t

        # Check shock wave or expansion wave appears or not.
        shock = left > right
        expansion = left <= right

        # Calculate the wavefront location.
        avg = (left + right) / 2   # shcok wave location
        left = shock * avg + expansion * left
        right = shock * avg + expansion * right
        internal = np.transpose((left, right)).ravel()
        return np.hstack((self.coord[0], internal, self.coord[-1]))


class Burgers1DApp(_base_app.OneDimBaseApp):
    """
    Main application for Burgers' equation 1D solver.
    """
    TITLE = "Burgers equation"
    num_region: int = 3
    region_x: list = [-10.0, 0.0, 2.0, 5.0]
    region_velocity: list = [-0.5, 1.0, 0.5]

    def populate_menu(self):
        """
        Set menu item for GUI.
        """
        self.add_action(
            "One",
            text="Burgers equation",
            tip="One-dimensional Burgers equation problem",
            func=self.run,
            id="one.burgers",
            weight=20,
        )

    def get_region_solver_config(self):
        """
        Get the solver configuration data (x & velocity) for region.

        :return: solver configuration data
        """
        region = []
        for i, x in enumerate(self.region_x):
            x_var = f"x{i}"
            x_des = f"The point {i} of x axis"
            region.append([x_var, x, x_des])
        for i, vel in enumerate(self.region_velocity):
            vel_var = f"velocity{i}{i + 1}"
            vel_des = f"The velocity between point {i} and {i + 1}"
            region.append([vel_var, vel, vel_des])
        return region

    def reset_region_properties(self):
        """
        Reset the region properties (x & velocity).
        """
        self.region_x = []
        for i in range(self.num_region + 1):
            x_var = f"x{i}"
            self.region_x.append(self.solver_config[x_var]["value"])
        self.region_velocity = []
        for i in range(self.num_region):
            vel_var = f"velocity{i}{i + 1}"
            self.region_velocity.append(self.solver_config[vel_var]["value"])

    def init_solver_config(self):
        """
        Set Burgers' equation solver configuration by user' input, and then
        reset the initial condition.
        """
        self.adjust_region = True
        solver_config_data = self.get_region_solver_config() + [
            ["time_interval", 0.01, "The amount of time in a time step"],
            ["max_steps", 200, "Maximum time step"],
            ["profiling", False, "Turn on / off solver profiling"],
        ]
        self.solver_config = _base_app.SolverConfig(solver_config_data)

    def set_plot_data(self):
        """
        Set the analytical data for plot.
            - velocity
        """
        self.plot_ana = True
        self.plot_data = []

        velocity = _base_app.QuantityLine(
            name="velocity",
            unit=r"$\mathrm{m}/\mathrm{s}$",
            color='b',
            y_upper_lim=1.2,
            y_bottom_lim=-1.2)
        setattr(self, velocity.name, velocity)
        self.plot_data.append([self.velocity.name, True])

    def init_solver(self):
        """
        Initialize Burgers' equation solver and set up the initial condition.
        """
        self.st = BurgersEquation()
        self.reset_region_properties()
        self.st.build_constant(self.region_x, self.region_velocity)
        self.st.build_field(t=0)

    def update_step(self, steps):
        """
        Update data at current step.
            - analytical data solved in python.
        """
        self.current_step += steps
        time_current = self.current_step * self.time_interval
        self.st.build_field(t=time_current)

    def get_region_add(self):
        """
        Add solver configuration item to the GUI.

        :return: solver configuration data to be added
        """
        self.num_region += 1
        pos_x = self.num_region
        data_x = [f"x{pos_x}",
                  self.region_x[-1],
                  f"The point {pos_x} of x axis"]
        pos_vel = self.num_region * 2 - 1
        data_vel = [f"velocity{pos_x - 1}{pos_x}",
                    self.region_velocity[-1],
                    f"The velocity between point {pos_x - 1} and {pos_x}"]
        return [(pos_x, data_x), (pos_vel, data_vel)]

    def get_region_delete(self):
        """
        Delete solver configuration item to the GUI.

        :return: solver configuration data to be deleted
        """
        pos_x = self.num_region
        pos_vel = self.num_region * 2
        self.num_region -= 1
        return [pos_x, pos_vel]


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
