# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import json
import typing
import functools
import itertools
import importlib.util

from . import _gui_common

from .. import call_profiler
from .. import apputil

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import (
    QAbstractItemView,
    QTreeView,
    QWidget,
    QDockWidget,
    QCheckBox,
    QVBoxLayout,
    QLabel,
    QPushButton
)

from PySide6.QtCore import Qt

from PySide6.QtGui import (
    QStandardItem,
    QStandardItemModel
)

__all__ = ["Profiling"]


class Profiling(_gui_common.PilotFeature):
    """
    Create profiling windows.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._diag = QtWidgets.QFileDialog()
        self._diag.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self._diag.setWindowTitle("Open profiling file")

    def populate_menu(self):
        self.add_action(
            "Profiling",
            text="Open profiling result",
            tip="Open JSON file of profiling result",
            func=self.open_profiling_result,
            id="profiling.open",
            weight=10,
        )

    def open_profiling_result(self):
        self._diag.open(self, QtCore.SLOT("on_finished()"))

    @QtCore.Slot()
    def on_finished(self):
        filenames = []
        for path in self._diag.selectedFiles():
            filenames.append(path)

        result = None
        with open(filenames[0], "r") as f:
            result = json.loads(f.read())

        self._add_result_window(result)

    def _add_result_window(self, result: list[dict[str, typing.Any]]):
        self._table = self._mgr.addSubWindow(QWidget())
        self._table.setWindowTitle("Profiling result")
        self._tree_view = QTreeView(self._table)

        self._model = QStandardItemModel(self._tree_view)
        self._model.setHorizontalHeaderLabels(["Total Time", "Symbol Name"])

        def _recursive_add_item(
                parent: QStandardItem,
                data: dict, tot: float):
            percent = data["total_time"] * 100 / tot

            first_item = QStandardItem(
                f"{data['total_time']} ({percent:2f}%)")
            second_item = QStandardItem(data["name"])

            parent.appendRow([first_item, second_item])

            children = data["children"]
            def key_func(d): d["total_time"]
            sorted_child_data = sorted(children, key=key_func, reverse=True)

            for child_data in sorted_child_data:
                _recursive_add_item(first_item, child_data, data["total_time"])

        for data in result:
            first_item = QStandardItem(f"{data['total_time']} (100%)")
            second_item = QStandardItem(data["name"])

            self._model.appendRow([first_item, second_item])

            for child_data in data["children"]:
                _recursive_add_item(first_item, child_data, data['total_time'])

        self._tree_view.setStyleSheet(
            """
            QTreeView {
                background-color: #2e2e2e;
                color: #dcdcdc;
                border: 1px solid #555555;
            }
            QTreeView::item {
                border: 1px solid #3c3c3c;
                border-right: none;
                border-bottom: none;
            }
            QTreeView::item:selected {
                background-color: #4f4f4f;
                color: white;
            }
            """
        )

        self._tree_view.setModel(self._model)
        self._tree_view.setColumnWidth(0, 400)
        self._tree_view.setColumnWidth(1, 200)
        self._tree_view.setEditTriggers(
            QAbstractItemView.NoEditTriggers)

        self._table.setWidget(self._tree_view)
        self._table.showMaximized()
        self._table.show()


class ProfileConfigWidget(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app

    def init_ui(self, options):
        main_layout = QVBoxLayout(self)

        def on_check_change(check_state, opt, item):
            if check_state == Qt.Checked:
                self.app.add_opt(opt, item)
            else:
                self.app.rm_opt(opt, item)

        for opt in options:
            title = opt[0]
            choices = opt[1]

            vbox = QVBoxLayout()
            vbox.setStretch(0, 4)

            title_label = QLabel(title)
            vbox.addWidget(title_label)

            for c in choices:
                checkbox = QCheckBox(c)
                checkbox.setTristate(False)
                cbfunc = functools.partial(on_check_change, opt=title, item=c)
                checkbox.checkStateChanged.connect(cbfunc)
                vbox.addWidget(checkbox)

            main_layout.addLayout(vbox)

        cmd_vbox = QVBoxLayout()
        cmd_vbox.setStretch(0, 4)

        command_label = QLabel("Command")
        cmd_vbox.addWidget(command_label)

        run_button = QPushButton("run")
        run_button.clicked.connect(self.app.run_profiling)
        cmd_vbox.addWidget(run_button)

        main_layout.addLayout(cmd_vbox)

        self.setLayout(main_layout)


class RunProfiling(_gui_common.PilotFeature):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._diag = QtWidgets.QFileDialog()
        self._diag.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self._diag.setWindowTitle("Open profiling file")
        self.module = None
        self.prof_candidate = {}
        self.prof_opt = {}

    def populate_menu(self):
        self.add_action(
            "Profiling",
            text="Profiling script",
            tip="Run profiling from script",
            func=self.load_profile_util,
            id="profiling.run",
            weight=20,
        )

    def load_profile_util(self):
        self._diag.open(self, QtCore.SLOT("on_finished()"))

    @QtCore.Slot()
    def on_finished(self):
        filenames = []
        for path in self._diag.selectedFiles():
            filenames.append(path)

        self.setup_profile_config(filenames[0])

    def import_profile_module(self, filename):
        spec = importlib.util.spec_from_file_location("profile_mod", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def setup_profile_config(self, filename):
        self.module = self.import_profile_module(filename)
        options = self.module.get_options()

        # Add profiling module to appenv for commamd-line access
        cae = apputil.get_current_appenv()
        cae.locals['profile_module'] = self.module

        self.init_ui(options)

    def add_opt(self, opt, item):
        if self.prof_opt.get(opt, None) is None:
            self.prof_opt[opt] = set()

        self.prof_opt[opt].add(item)

    def rm_opt(self, opt, item):
        if self.prof_opt.get(opt, None) is None:
            return

        self.prof_opt[opt].remove(item)

    def init_ui(self, options):
        config_window = ProfileConfigWidget(self)
        config_window.init_ui(options)
        config_widget = QDockWidget("config")
        config_widget.setWidget(config_window)
        self._mgr.mainWindow.addDockWidget(
            Qt.LeftDockWidgetArea, config_widget)

    def run_profiling(self):

        call_profiler.reset()

        opt_values = [[(k, vv) for vv in v] for k, v in self.prof_opt.items()]

        for case in itertools.product(*opt_values):
            self.module.run({k: v for k, v in case})

        result = call_profiler.result().get("children", None)
        if result is not None:
            self._add_result_window(result)

    def _add_result_window(self, result):
        self._table = self._mgr.addSubWindow(QWidget())
        self._table.setWindowTitle("Profiling result")
        self._tree_view = QTreeView(self._table)

        self._model = QStandardItemModel(self._tree_view)
        self._model.setHorizontalHeaderLabels(
            ["Symbol Name", "Total Time", "Count"])

        def _recursive_add_item(
                parent: QStandardItem,
                data: dict, tot: float):
            percent = data["total_time"] * 100 / tot

            first = QStandardItem(data["name"])
            second = QStandardItem(
                f"{data['total_time']} ({percent:2f}%)")
            third = QStandardItem(f"{data['count']}")

            parent.appendRow([first, second, third])

            children = data["children"]
            def key_func(d): d["total_time"]
            sorted_child_data = sorted(children, key=key_func, reverse=True)

            for child_data in sorted_child_data:
                _recursive_add_item(first, child_data, tot)

        for data in result:
            first = QStandardItem(data["name"])
            second = QStandardItem(f"{data['total_time']} (100%)")
            third = QStandardItem(f"{data['count']}")

            self._model.appendRow([first, second, third])

            for child_data in data["children"]:
                _recursive_add_item(first, child_data, data['total_time'])

        self._tree_view.setStyleSheet(
            """
            QTreeView {
                background-color: #2e2e2e;
                color: #dcdcdc;
                border: 1px solid #555555;
            }
            QTreeView::item {
                border: 1px solid #3c3c3c;
                border-right: none;
                border-bottom: none;
            }
            QTreeView::item:selected {
                background-color: #4f4f4f;
                color: white;
            }
            """
        )

        self._tree_view.setModel(self._model)
        self._tree_view.setColumnWidth(0, 200)
        self._tree_view.setColumnWidth(1, 200)
        self._tree_view.setColumnWidth(1, 200)
        self._tree_view.setEditTriggers(
            QAbstractItemView.NoEditTriggers)

        self._table.setWidget(self._tree_view)
        self._table.show()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
