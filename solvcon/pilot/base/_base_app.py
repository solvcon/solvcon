# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import sys
import dataclasses

import numpy as np

import matplotlib
import matplotlib.pyplot
from matplotlib.backends import backend_qtagg
from matplotlib import figure
from matplotlib.backends import backend_qt5agg

from PySide6.QtCore import QTimer, Slot, Qt, QAbstractTableModel
from PySide6.QtWidgets import (QDockWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QComboBox, QPushButton, QSpacerItem, QMenu,
                               QSizePolicy, QDialog, QWidget, QTableView)

from ... import core as mcore

from . import _gui_common


@dataclasses.dataclass
class QuantityLine(object):
    """
    A class representing a quantity line with associated data.

    This class is a data structure that holds information about a quantity line
    in a plot. It is designed to be used in conjunction with Matplotlib for
    visualizing analytical and numerical data.

    :cvar ana: Line2D for analytical data.
    :vartype ana: matplotlib.lines.Line2D
    :cvar num: Line2D for numerical data.
    :vartype num: matplotlib.lines.Line2D
    :cvar axis: Axis for the plot.
    :vartype axis: matplotlib.pyplot.axis
    :cvar name: Name of the quantity.
    :vartype name: str
    :cvar unit: Unit of measurement.
    :vartype unit: str
    :cvar color: Color of line or symbol.
    :vartype color: str
    :cvar y_upper_lim: y axis upper limit.
    :vartype y_upper_lim: float
    :cvar y_bottom_lim: y axis bottom limit.
    :vartype y_bottom_lim: float

    :Methods:
        - :py:meth:`update_ana`: Update the line data and redraw the plot.
        - :py:meth:`update_num`: Update the line data and redraw the plot.
    """
    ana: matplotlib.lines.Line2D = None
    num: matplotlib.lines.Line2D = None
    axis: matplotlib.pyplot.axis = None
    name: str = ""
    unit: str = ""
    color: str = ""
    y_upper_lim: float = 0.0
    y_bottom_lim: float = 0.0

    def update_ana(self, x=None, y=None):
        """
        Update analytical data and redraw the plot.

        :param x: Analytical data.
        :type x: ndarray
        :param y: Analytical data.
        :type y: ndarray
        :return: None
        """
        if x is not None:
            self.ana.set_xdata(x)
        if y is not None:
            self.ana.set_ydata(y)
        self.ana.figure.canvas.draw()

    def update_num(self, x=None, y=None):
        """
        Update numerical data and redraw the plot.

        :param x: Numerical data.
        :type x: ndarray
        :param y: Numerical data.
        :type y: ndarray
        :return: None
        """
        if x is not None:
            self.num.set_xdata(x)
        if y is not None:
            self.num.set_ydata(y)
        self.num.figure.canvas.draw()


class _Accessor(object):
    """
    Helper calss to access data within the configuration table using
    multiple dimensions, currenlty only support 2-dimensions table.

    This class allows users to access data by chaining multiple [],
    supporting string-based.

    :ivar _data: handle input data that allow user can using
        index or string to access it.
    :ivar _header: list of data's column header name.
    :vartype _header: list
    :ivar _dim_idx: using to indicate which dimension is
        currenlty being accessed by __getitem__.
    :vartype _dim_idx: int
    """

    def __init__(self, data, dim_idx=0, header=None):
        self._data = data
        self._header = header
        self._dim_idx = dim_idx

    def __getitem__(self, key):
        if self._dim_idx == 0:
            for row_idx, row in enumerate(self._data):
                if key == row[0]:
                    return _Accessor(self._data[row_idx],
                                     self._dim_idx + 1,
                                     self._header)
        else:
            for idx, ele in enumerate(self._header):
                if key == ele:
                    return self._data[idx]
        raise KeyError(f'Invaild key: {key} not found in table')


class GUIConfig(object):
    """
    Configuration class for the GUI.

    This class provides a configuration interface for the GUI, allowing
    users to set and retrieve parameters related to the simulation.

    :ivar: _tbl_content (list[list]): The content of the configuration
        table.
    :ivar: _col_header (list): The header for the configuration
        table columns.

    :Methods:
        - :py:meth:`data`: Get the value at a specific row and column
            in the configuration table.
        - :py:meth:`setData`: Set the value at a specific row
            and column in the configuration table.
        - :py:meth:`columnHeader`: Get the header for a specific column
            in the configuration table.
        - :py:meth:`editable`: Check if a cell in the configuration
            table is editable.
        - :py:meth:`rowCount`: Get the number of rows in the configuration
            table.
        - :py:meth:`columnCount`: Get the number of columns in the
            configuration table.
    """

    def __init__(self, input_data, col_headers):
        self._tbl_content = input_data
        self._col_header = col_headers

        expected_length = len(self._col_header)
        for i, row in enumerate(self._tbl_content):
            if len(row) != expected_length:
                raise ValueError(
                    f"Row {i} has length {len(row)}, "
                    f"expected {expected_length}"
                )

    def __getitem__(self, key):
        return _Accessor(self._tbl_content, 0, self._col_header)[key]

    def data(self, row, col):
        """
        Get the value at a specific row and column in the configuration table.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :return:
            The value at the specified location in the configuration table.
        """
        return self._tbl_content[row][col]

    def setData(self, row, col, value):
        """
        Set the value at a specific row and column in the configuration table.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :prarm value: Any
        :return: None
        """
        self._tbl_content[row][col] = value
        return True

    def columnHeader(self, col):
        """
        Get the specific column header in the configuration table.

        :param col: Column index.
        :type col: int
        :return: The header for the specific column.
        :rtype: list[str]
        """
        return self._col_header[col]

    def editable(self, row, col):
        """
        Check if the cell is editable.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :return: True if the cell is editable, false otherwise.
        """
        if col == 1:
            return True
        return False

    def rowCount(self):
        """
        Get the number of rows in the configuration table.

        :return: The number of rows.
        """
        return len(self._tbl_content)

    def columnCount(self):
        """
        Get the number of columns in the configuration table.

        :return: The number of columns.
        """
        return len(self._tbl_content[0])


class SolverConfig(GUIConfig):
    """
    Configuration class for the solver.

    This class provides a configuration interface for the solver, allowing
    users to set and retrieve parameters related to the simulation.
    """

    def __init__(self, input_data):
        super().__init__(input_data, ["variable", "value", "description"])


class PlotConfig(GUIConfig):
    """
    Configuration class for the plot.

    This class provides a configuration interface for the plot, allowing
    users to set and retrieve parameters related to the plotting arae.
    """

    def __init__(self, input_data):
        super().__init__(input_data, ["variable", "line_selection"])

    def editable(self, row, col):
        """
        Check if the cell is editable.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :return: True if the cell is editable, false otherwise.
        """
        if col >= 1:
            return True
        return False


class ConfigOption(GUIConfig):
    """
    Configuration class for the figure.

    This class provides a set of options for the figure, allowing users to set
    and retrieve parameters related to the figure's.
    """

    def __init__(self, input_data):
        super().__init__(input_data, ["variable", "option"])


class DataConfig(GUIConfig):
    """
    Configuration class for the figure.

    This class provides a data configuration structure for the solver.
    """

    def __init__(self, input_data):
        super().__init__(input_data, ["variable", "configuration"])


class OneDimBaseApp(_gui_common.PilotFeature):
    """
    Main application for 1D solver.

    This class provides the main application logic for the 1D solver.
    It includes methods for initializing the solver, configuring parameters,
    setting up timers, and building visualization figures.

    :cvar solver_config: Configuration object for the solver.
    :vartype solver_config: :class:`SolverConfig`
    :cvar plot_data: Data list for the plotting area.
    :vartype plot_data: list[str, bool]
    :cvar plot_config: Configuration object for the plotting area.
    :vartype plot_config: :class:`PlotConfig`
    :cvar plot: State object for the holding plots.
    :vartype plot: :class:`State`
    :cvar plot_ana: Flag indicating whether to plot analytical data.
    :vartype plot_ana: bool
    :cvar plot_num: Flag indicating whether to plot numerical data.
    :vartype plot_num: bool
    :cvar use_grid_layout: Flag indicating whether to use a grid
          layout.
    :vartype use_grid_layout: bool
    :cvar adjust_region: Flag indicating whether to adjust the region.
    :vartype adjust_region: bool

    :Methods:
        - :py:meth:`populate_menu`: Set menu item for GUI.
        - :py:meth:`run`: Create the GUI environment.
        - :py:meth:`setup_app`: Create the window for solver.
        - :py:meth:`init_solver_config`: Initialize solver configuration data.
        - :py:meth:`set_plot_data`: Set the property of st,
            and set list of data.
        - :py:meth:`set_solver_config`: Initialize solver configure by user's
          input, also reset the computational results.
        - :py:meth:`init_solver`: Initialize solver and set up the initial
          conditions.
        - :py:meth:`setup_timer`: Set up the Qt timer for data visualization.
        - :py:meth:`build_single_figure`: Build a single-figure layout for
          visualization.
        - :py:meth:`build_grid_figure`: Build a grid figure for visualization.
        - :py:meth:`init_plot_data`: Initialize analytical and
          numerical data in figure.
        - :py:meth:`step`: Callback function for the step button.
        - :py:meth:`update_step`: Update data at current step.
        - :py:meth:`start`: Start the solver.
        - :py:meth:`set`: Set the solver configurations and update the timer.
        - :py:meth:`update_layout`: Refresh plotting area layout.
        - :py:meth:`stop`: Stop the solver.
        - :py:meth:`single_layout`: Switch plot holder to a single plot layout.
        - :py:meth:`grid_layout`: Switch plot holder to a grid plot layout.
        - :py:meth:`timer_timeout`: Callback function for Qt timer.
        - :py:meth:`log`: Print log messages to the console window and
          standard output.
        - :py:meth:`update_plot`: Updating plot after the solver finishes its
          computation each time.
    """
    st = None
    solver_config: SolverConfig = None
    plot_data: list[list[str, bool]] = None
    plot_config: PlotConfig = None
    config_option: list[list[str, ConfigOption]] = None
    data_config: DataConfig = None
    plot: backend_qtagg.FigureCanvas = None
    plot_ana: bool = False
    plot_num: bool = False
    use_grid_layout: bool = False
    adjust_region: bool = False

    #: Title of the plot sub-window; the Window menu lists windows by it.
    TITLE = "Base Window"

    def populate_menu(self):
        """
        Set menu item for GUI.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def run(self):
        """
        Create the GUI environment.
        """
        self.setup_app()

        # A new dock widget for showing config
        config_window = ConfigWindow(self)
        config_widget = QDockWidget("config")
        config_widget.setWidget(config_window)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea,
                                           config_widget)

        # A new sub-window (`QMdiSubWindow`) for the plot area
        self._subwin = self._mgr.addSubWindow(QWidget())
        self._subwin.setWindowTitle(self.TITLE)
        self._subwin.setWidget(PlotArea(self))
        self._subwin.showMaximized()
        self._subwin.show()

    def setup_app(self):
        """
        Create the window for solver.
        """
        self.init_solver_config()
        self.set_plot_data()
        self.plot_config = PlotConfig(self.plot_data)
        self.set_config_option()
        self.data_config = DataConfig(self.config_option)
        self.set_solver_config()
        self.setup_timer()
        self.plot = self.build_single_figure()

    def set_config_option(self):
        """
        Initialize figure configuration data.
        """
        self.config_option = []
        for name, *_ in self.plot_data:
            data = getattr(self, name)
            option = [["y_min", data.y_bottom_lim],
                      ["y_max", data.y_upper_lim],
                      ["color", data.color],
                      ["unit", data.unit]]
            self.config_option.append([name, ConfigOption(option)])

    def init_solver_config(self):
        """
        Initialize solver configuration data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def set_plot_data(self):
        """
        Set the property of st and set list of data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def set_solver_config(self):
        """
        Initialize solver configure by user's input, also reset the
        computational results.
        """
        self.init_solver()
        self.current_step = 0
        self.time_interval = self.solver_config["time_interval"]["value"]
        self.max_steps = self.solver_config["max_steps"]["value"]
        self.profiling = self.solver_config["profiling"]["value"]

    def init_solver(self):
        """
        Initialize solver and set up the initial conition.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def setup_timer(self):
        """
        Set up the Qt timer for data visualization.
        """
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_timeout)

    def build_single_figure(self):
        """
        Build a single-figure layout for visualization.

        :return: backend_qtagg.FigureCanvas
        """
        fig = figure.Figure()
        canvas = backend_qtagg.FigureCanvas(fig)
        ax = canvas.figure.subplots()
        ax.grid()
        fig.tight_layout()

        y_bottom_lim = sys.float_info.max
        y_upper_lim = 0.0
        for name, *_ in self.plot_data:
            if self.plot_config[name]["line_selection"]:
                data = getattr(self, name)
                data.axis = ax
                self.init_figure_items(data)
                self.init_plot_data(data)
                y_bottom_lim = min(y_bottom_lim, data.y_bottom_lim)
                y_upper_lim = max(y_upper_lim, data.y_upper_lim)

        matplotlib.pyplot.setp(ax, ylim=[y_bottom_lim, y_upper_lim])
        self.update_plot()
        return canvas

    def build_grid_figure(self):
        """
        Build a grid figure for visualization.

        :return: backend_qtagg.FigureCanvas
        """
        fig = figure.Figure()
        canvas = backend_qtagg.FigureCanvas(fig)
        ax = canvas.figure.subplots(3, 2)
        fig.tight_layout()

        for i, [name, *_] in enumerate(self.plot_data):
            if self.plot_config[name]["line_selection"]:
                data = getattr(self, name)
                data.axis = ax[i // 2][i % 2]
                data.axis.grid()
                self.init_figure_items(data)
                self.init_plot_data(data)
                matplotlib.pyplot.setp(
                    data.axis,
                    ylim=[data.y_bottom_lim, data.y_upper_lim])

        self.update_plot()
        return canvas

    def init_plot_data(self, data):
        """
        Initialize analytical and numerical data in figure.

        :param data: property of solver.
        :type data: QuantityLine
        :return: None
        """
        if self.plot_ana:
            x = self.st.coord_field
            data.ana, = data.axis.plot(x, np.zeros_like(x),
                                       f"{data.color}-",
                                       label=f'{data.name}_ana')
        if self.plot_num:
            x = self.st.svr.coord[self.st.svr.xindices]
            data.num, = data.axis.plot(x, np.zeros_like(x),
                                       f"{data.color}x",
                                       label=f'{data.name}_num')
        data.axis.set_xlabel("distance")
        data.axis.legend()

    def init_figure_items(self, data):
        """
        Initialize figure configuration data.

        :param data: property of solver.
        :type data: QuantityLine
        :return: None
        """
        config_option = self.data_config[data.name]["configuration"]
        data.y_bottom_lim = config_option["y_min"]["option"]
        data.y_upper_lim = config_option["y_max"]["option"]
        data.color = config_option["color"]["option"]
        data.unit = config_option["unit"]["option"]

    def step(self, steps=1):
        """
        Callback function for the step button.
        """
        if self.max_steps and self.current_step > self.max_steps:
            self.stop()
            return

        self.update_step(steps)
        self.update_plot()
        if self.profiling:
            self.log(mcore.time_registry.report())

    def update_step(self, steps=1):
        """
        Update data at current step.
        """
        raise NotImplementedError(f"{self.__class__.__name__} not implemented"
                                  f"{sys._getframe().f_code.co_name}")

    def start(self):
        """
        Start the solver.
        """
        self.timer.start(self.time_interval)

    def set(self):
        """
        Set the solver configurations and update the timer.
        """
        self.set_solver_config()
        self.setup_timer()
        self.update_layout()

        # Update PlotArea while click set button
        self._subwin.setWidget(PlotArea(self))

    def update_layout(self):
        """
        Refresh plotting area layout.
        """
        if self.use_grid_layout:
            self.plot = self.build_grid_figure()
        else:
            self.plot = self.build_single_figure()

    def stop(self):
        """
        Stop the solver.
        """
        self.timer.stop()

    def single_layout(self):
        """
        Switch plot holder to a single plot layout.
        """
        self.use_grid_layout = False
        self.plot = self.build_single_figure()

    def grid_layout(self):
        """
        Switch plot holder to a grid plot layout.
        """
        self.use_grid_layout = True
        self.plot = self.build_grid_figure()

    @Slot()
    def timer_timeout(self):
        """
        Callback function for Qt timer timeout.
        """
        self.step()

    def log(self, msg):
        """
        Print log messages to the console window and standard output.
        """
        # stdout can be None under some conditions
        # ref: https://github.com/solvcon/solvcon/issues/334
        if sys.stdout is not None:
            sys.stdout.write(msg)
            sys.stdout.write('\n')
        self._pycon.writeToHistory(msg)
        self._pycon.writeToHistory('\n')

    def update_plot(self):
        """
        Updating plot after the solver finishes its computation each time.
        """
        if self.use_grid_layout:
            for name, *_ in self.plot_data:
                data = getattr(self, name)
                self.update_plot_data(data)
        else:
            for name, *_ in self.plot_data:
                if self.plot_config[name]["line_selection"]:
                    data = getattr(self, name)
                    self.update_plot_data(data)

    def update_plot_data(self, data):
        """
        Update analytical and numerical data.

        :param data: property of solver.
        :type data: QuantityLine
        :return: None
        """
        if self.plot_config[data.name]["line_selection"]:
            if self.plot_ana:
                ana_x = self.st.coord_field
                ana_y = getattr(self.st, data.name + "_field")
                data.update_ana(x=ana_x, y=ana_y)
            if self.plot_num:
                num_y = getattr(self.st.svr, data.name)[self.st.svr.xindices]
                data.update_num(y=num_y)


class PlotConfigDialog(QDialog):
    """
    PlotConfigDialog class for managing plot configuration.

    This class inherits from the QDialog class and provides a GUI for managing
    plot configuration. It will pop up when clicking "Option" button. It
    includes options for selecting the layout and setting data line config.
    """

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Plot Options")
        layout = QVBoxLayout()

        layout_selection_label = QLabel("Layout selection")
        layout.addWidget(layout_selection_label)

        hbox = QHBoxLayout()
        grid_button = QPushButton("Grid")
        grid_button.clicked.connect(self.app.grid_layout)
        hbox.addWidget(grid_button)

        single_button = QPushButton("Single")
        single_button.clicked.connect(self.app.single_layout)
        hbox.addWidget(single_button)

        layout.addLayout(hbox)

        data_line_config_label = QLabel("Data line configuration")
        layout.addWidget(data_line_config_label)

        table = QTableView()
        table.horizontalHeader().setStretchLastSection(True)
        table_model = ConfigTableModel(self.app.plot_config)
        table.setModel(table_model)
        layout.addWidget(table)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.app.update_layout)
        layout.addWidget(save_button)

        self.setLayout(layout)


class FigureConfigDialog(QDialog):
    """
    PlotConfigDialog class for managing plot configuration.

    This class inherits from the QDialog class and provides a GUI for managing
    plot configuration. It will pop up when clicking "Option" button. It
    includes options for selecting the layout and setting data line config.
    """

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Figure Options")
        layout = QVBoxLayout()

        for name, *_ in self.app.config_option:
            layout_data_label = QLabel(name)
            layout.addWidget(layout_data_label)
            table = QTableView()
            table.horizontalHeader().setStretchLastSection(True)
            table_model = ConfigTableModel(
                self.app.data_config[name]["configuration"])
            table.setModel(table_model)
            layout.addWidget(table)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.app.update_layout)
        layout.addWidget(save_button)

        self.setLayout(layout)


class ConfigWindow(QWidget):
    """
    ConfigWindow class for managing solver and plotting area configurations.

    This class inherit from the QWidget class and provides a graphical user
    interface for managing solver configurations. It includes options to set
    the solver type, view and edit configuration parameters, and control the
    solver's behavior.

    :ivar app: The app want to plot something in plotting area.
    :ivar solver_config: Configuration object for the solver.
    :vartype solver_config: :class:`SolverConfig`
    :ivar plot_config: Configuration object for the plotting area.
    :vartype plot_config: :class:`PlotConfig`

    :Methods:
        - :py:meth:`setup`: Setup method to configure the window.
        - :py:meth:`on_open_plot_option`: Callback function
            when plot configure modal window is opened.
        - :py:meth:`on_close_plot_option`: Callback function
            when plot configure modal windows is closed.
        - :py:meth:`add_region`: Add data for a new region.
        - :py:meth:`delete_region`: Delete data for a region.
        - :py:meth:`init_ui`: Define the GUI layout of the window.
    """

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.plot_config_open = False
        self.figure_config_open = False
        self.setup()
        self.init_ui()

    def setup(self):
        """
        Assign the configure object from app

        :return: None
        """
        self.solver_config = self.app.solver_config
        self.plot_config = self.app.plot_config
        self.data_config = self.app.data_config

    def on_open_plot_option(self):
        self.plot_config_open = True
        if not self.plot_config_dialog:
            self.plot_config_dialog = PlotConfigDialog(self.app, self)
        self.plot_config_dialog.exec_()

    def on_close_plot_option(self):
        self.plot_config_open = False
        if self.plot_config_dialog:
            self.plot_config_dialog.close()

    def on_open_figure_option(self):
        self.figure_config_open = True
        if not self.figure_config_dialog:
            self.figure_config_dialog = FigureConfigDialog(self.app, self)
        self.figure_config_dialog.exec_()

    def on_close_figure_option(self):
        self.figure_config_open = False
        if self.figure_config_dialog:
            self.figure_config_dialog.close()

    def add_region(self):
        """
        Add data for a new region to the solver configuration table.
        """
        table_model = ConfigTableModel(self.solver_config)
        list_add = sorted(self.app.get_region_add(),
                          key=lambda x: x[0],
                          reverse=True)
        for pos, data in list_add:
            table_model.insertRow(data, pos)
        self.table.setModel(table_model)

    def delete_region(self):
        """
        Delete data for a region from the solver configuration table.
        """
        table_model = ConfigTableModel(self.solver_config)
        list_del = sorted(self.app.get_region_delete(), reverse=True)
        for pos in list_del:
            table_model.deleteRow(pos)
        self.table.setModel(table_model)

    def init_ui(self):
        """
        Define the GUI layout of the window for this class

        :return: QWidget
        """
        main_layout = QVBoxLayout(self)

        # First VBox, handle operation about solver configuration, weight=4
        vbox1 = QVBoxLayout()
        vbox1.setStretch(0, 4)

        solver_label = QLabel("Solver")
        vbox1.addWidget(solver_label)

        combo_box = QComboBox()
        combo_box.addItem("Euler1D-CESE")
        vbox1.addWidget(combo_box)

        config_label = QLabel("Configuration")
        vbox1.addWidget(config_label)

        self.table = QTableView()
        self.table.horizontalHeader().setStretchLastSection(True)
        table_model = ConfigTableModel(self.solver_config)
        self.table.setModel(table_model)
        vbox1.addWidget(self.table)

        if self.app.adjust_region:
            self.table.setContextMenuPolicy(Qt.CustomContextMenu)
            self.table.customContextMenuRequested.connect(
                self.adjust_region_menu)

        option_button = QPushButton("Figure Options")
        option_button.clicked.connect(self.on_open_figure_option)
        vbox1.addWidget(option_button)

        option_button = QPushButton("Plot Options")
        option_button.clicked.connect(self.on_open_plot_option)
        vbox1.addWidget(option_button)

        set_button = QPushButton("Set")
        set_button.clicked.connect(self.app.set)
        vbox1.addWidget(set_button)

        main_layout.addLayout(vbox1)

        # Second VBox, handle operation about calculation, weight=1
        vbox2 = QVBoxLayout()
        vbox2.setStretch(0, 1)

        spacer = QSpacerItem(
            20, 40, QSizePolicy.Minimum,
            QSizePolicy.Expanding)
        vbox2.addItem(spacer)

        time_label = QLabel("Time Control")
        vbox2.addWidget(time_label)

        start_button = QPushButton("Start")
        start_button.clicked.connect(self.app.start)
        vbox2.addWidget(start_button)

        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self.app.stop)
        vbox2.addWidget(stop_button)

        step_button = QPushButton("Step")
        step_button.clicked.connect(lambda: self.app.step())
        vbox2.addWidget(step_button)

        main_layout.addLayout(vbox2)

        # Modal window for plot configuration
        self.figure_config_dialog = FigureConfigDialog(self.app, self)
        self.plot_config_dialog = PlotConfigDialog(self.app, self)

        self.setLayout(main_layout)

    def adjust_region_menu(self, pos):
        menu = QMenu(self)
        add_action = menu.addAction("Add Region")
        delete_action = menu.addAction("Delete Region")
        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if action == add_action:
            self.add_region()
        elif action == delete_action:
            self.delete_region()


class PlotArea(QWidget):
    """
    Class for displaying the plot area in the application.

    This class inherits from `QWidget` and is responsible for managing the
    display of the plot area in the application.

    :ivar app: The app want to plot something in plotting area.

    :Methods:
        - :py:meth:`init_ui`: Define the GUI layout of the window.
    """

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        toolbar = backend_qt5agg.NavigationToolbar2QT(
            self.app.plot, None)
        layout.addWidget(toolbar)
        layout.addWidget(self.app.plot)

        self.setLayout(layout)


class ConfigTableModel(QAbstractTableModel):
    """
    ConfigTableModel class for displaying configuration data.

    This class inherits from the QAbstractTableModel class and provides a model
    for displaying configuration data using table. It is used in the QTable
    class to display solver and plotting area configurations.

    :ivar config: Configuration object for the model.
    :vartype config: :class:`GUIConfig`

    :Methods:
        - :py:meth:`rowCount`: Get the number of rows in the model.
        - :py:meth:`columnCount`: Get the number of columns in the model.
        - :py:meth:`data`: Get the data at a specific index in the
          model.
        - :py:meth:`setData`: Set the data at a specific index
          in the model.
        - :py:meth:`flags`: Get the flags for a specific index in the
          model.
        - :py:meth:`headerData`: Get the header data
          for a specific section in the model.
        - :py:meth:`insertRow`: Insert a new row of data.
        - :py:meth:`deleteRow`: Delete a row of data.
    """

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def rowCount(self, parent):
        return self.config.rowCount()

    def columnCount(self, parent):
        return self.config.columnCount()

    def data(self, index, role):
        if role in {Qt.DisplayRole, Qt.EditRole}:
            return self.config.data(index.row(), index.column())
        return None

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self.config.setData(index.row(), index.column(), value)
            return True

    def flags(self, index):
        if self.config.editable(index.row(), index.column()):
            return super().flags(index) | Qt.ItemIsEditable
        return super().flags(index)

    def headerData(self, section, orientation, role):
        if (role == Qt.DisplayRole
                and orientation == Qt.Horizontal):
            return self.config.columnHeader(section)
        return None

    def insertRow(self, data, position=None):
        """
        Insert data at specific row.

        :param data: data to be inserted
        :param position: row index of config._tbl_content
        :return: None
        """
        if len(data) != self.config.columnCount():
            raise ValueError(
                f"Row {position} has length {len(data)}, "
                f"expected {self.config.columnCount()}"
            )
        position = self.rowCount(None) if position is None else position
        if position < 0 or position > self.rowCount(None):
            raise IndexError(f"Row {position} out of range")
        self.config._tbl_content.insert(position, data)
        return None

    def deleteRow(self, position):
        """
        Delete data at specific row.

        :param position: row index of config._tbl_content
        :type position: int
        :return: None
        """
        if position < 0 or position >= self.rowCount(None):
            raise IndexError(f"Row {position} out of range")
        self.config._tbl_content.pop(position)
        return None


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
